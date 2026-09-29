// ============================================================================
// my_window — HLS window-based clustering core (final version, v35k)
//
// Input: 600 hits x 12-D embedding (ap_fixed<8,4>). Output: 600 labels.
//
// Pipeline (stages run back to back, each loop pipelined with II=1):
//   PCA_HIST    PCA 12 -> 8 (constants quantised to ap_fixed<8,4>) and a
//               lane-private packed histogram of the coarse sort key
//               bucket = floor(pca0 * 4).
//   BASE        Exclusive prefix sum over the histogram (8 buckets/clock).
//   RANK        Scatter every point to its stable sorted position.
//   EMIT_EDGE   Read the sorted stream back and build edges: for each
//               position and shift 1..12, edge iff the 8-D squared distance
//               is < 0.09.
//   FW_CHAIN    Fixed-step label propagation: forward, backward, forward,
//               backward sweeps as one cascaded, skewed pipeline.
//   JUMP1       Pointer-jumping round 1 (double-pumped: positions k and
//               k+300 per tick).
//   JUMP2       Pointer-jumping round 2, scattering roots back to original
//               order.
//   UNSORT_EM   Emit the labels (32 per tick).
//
// Implementation notes:
//  * Sorted-space loops walk GRAIN=16 consecutive positions per tick, so the
//    array bank (position & 15) is a compile-time constant per unrolled lane
//    and no runtime bank muxes appear. Arbitrary-address writes use mask
//    bank-decode instead of %15 dividers.
//  * Those grain-16 arrays are word-packed (comp_aw / order_w: 160 b per
//    group, edge words em1: 192 b per group). Shift-register chains hand words
//    between the fused FW/BW stages, so no intermediate arrays are needed.
//  * EDGE computes all 8 dimensions per tick on DSP multipliers
//    (1536 15x15 products, latency pinned to 2): d2 += diff*diff.
//  * srt_os and co carry a present flag in bit 10 ({pres, val} words).
//  * EMIT and UNSORT find the one owning lane per position with a 4-bit
//    OR-encoded index and a single indexed tree mux, not a select chain.
//  * JUMP1/JUMP2 gathers read word tables (j1tab: 16 t2p BRAM replicas) plus
//    a lane select.
//  * PCA accumulators are ap_fixed<20,12>: |sum| < 2048, so nothing wraps and
//    the result equals the wider format. emb-MEAN is hoisted out of the
//    dimension loop.
//  * BASE keeps every prefix feed-forward; the only loop-carried path is
//    `run += total`, which is what closes timing.
//  * The histogram is self-cleaning (static; RANK re-zeroes the words it
//    touched), and the present-flag clears are folded into PCA_HIST.
//
// Bit-exact with the earlier, less optimised versions: only storage layout
// and scheduling differ.
// ============================================================================
#include "my_window.h"

static const int PAR = LANES;          // block-space lanes
static const int M   = N / LANES;      // elements per lane (block segment)
static const int HW  = HIST_SIZE / 4;  // packed histogram words (4 buckets each)
static const int G   = GRAIN;          // sorted-space grain (16)
static const int NG  = (N + G - 1) / G;   // sorted groups per pass (38)
static const int NH  = N / 2;             // 300 (double-pump halves)
static const int MJ  = (NH + G - 1) / G;  // double-pump groups (19)

// ---- PCA constants (quantised to ap_fixed<8,4>) ----
static const emb_t MEAN[EMB_DIM] = {
    (emb_t)-2.875,  (emb_t)0.5625, (emb_t)1.875,   (emb_t)0.0625,
    (emb_t)0.0,     (emb_t)1.0,    (emb_t)0.125,   (emb_t)3.125,
    (emb_t)-1.6875, (emb_t)0.1875, (emb_t)3.125,   (emb_t)-1.6875
};

static const emb_t COMPONENTS[PCA_DIM][EMB_DIM] = {
    { (emb_t) 0.4375, (emb_t)-0.0625, (emb_t)-0.1875, (emb_t) 0.0,
      (emb_t) 0.0,    (emb_t)-0.125,  (emb_t) 0.0,    (emb_t)-0.5625,
      (emb_t) 0.375,  (emb_t) 0.0,    (emb_t)-0.5,    (emb_t) 0.25   },
    { (emb_t)-0.1875, (emb_t) 0.0,    (emb_t)-0.25,   (emb_t) 0.125,
      (emb_t)-0.0625, (emb_t) 0.0625, (emb_t) 0.0625, (emb_t)-0.4375,
      (emb_t)-0.6875, (emb_t) 0.0625, (emb_t)-0.3125, (emb_t)-0.3125 },
    { (emb_t) 0.0,    (emb_t)-0.875,  (emb_t) 0.125,  (emb_t)-0.1875,
      (emb_t)-0.125,  (emb_t) 0.1875, (emb_t)-0.0625, (emb_t) 0.0,
      (emb_t) 0.0,    (emb_t) 0.3125, (emb_t) 0.0,    (emb_t) 0.0    },
    { (emb_t)-0.0625, (emb_t)-0.0625, (emb_t)-0.9375, (emb_t) 0.0,
      (emb_t)-0.125,  (emb_t) 0.1875, (emb_t) 0.0625, (emb_t) 0.125,
      (emb_t) 0.125,  (emb_t) 0.0,    (emb_t) 0.25,   (emb_t) 0.0625 },
    { (emb_t)-0.0625, (emb_t)-0.0625, (emb_t) 0.125,  (emb_t)-0.0625,
      (emb_t) 0.1875, (emb_t) 0.0,    (emb_t) 0.1875, (emb_t)-0.6875,
      (emb_t) 0.125,  (emb_t) 0.0,    (emb_t) 0.6875, (emb_t)-0.0625 },
    { (emb_t) 0.8125, (emb_t)-0.0625, (emb_t) 0.0,    (emb_t) 0.0625,
      (emb_t)-0.125,  (emb_t) 0.0625, (emb_t) 0.125,  (emb_t) 0.125,
      (emb_t)-0.1875, (emb_t)-0.125,  (emb_t) 0.1875, (emb_t)-0.375  },
    { (emb_t) 0.125,  (emb_t) 0.0,    (emb_t)-0.0625, (emb_t) 0.0,
      (emb_t)-0.1875, (emb_t)-0.1875, (emb_t)-0.875,  (emb_t)-0.125,
      (emb_t)-0.25,   (emb_t)-0.0625, (emb_t) 0.25,   (emb_t) 0.1875 },
    { (emb_t) 0.0625, (emb_t)-0.1875, (emb_t)-0.0625, (emb_t) 0.0,
      (emb_t)-0.0625, (emb_t)-0.625,  (emb_t) 0.375,  (emb_t) 0.0625,
      (emb_t)-0.375,  (emb_t)-0.0625, (emb_t) 0.125,  (emb_t) 0.5    }
};


void my_window(emb_t emb[N][EMB_DIM], int labels[N]) {
#pragma HLS ARRAY_PARTITION variable=emb block factor=PAR dim=1
#pragma HLS ARRAY_PARTITION variable=emb complete dim=2
    // v35g: labels written 32/clock by the grain-32 UNSORT_EM -> factor 32
    // (bank jj still a compile-time constant per unrolled lane)
#pragma HLS ARRAY_PARTITION variable=labels cyclic factor=32 dim=1

    // ===================== ORIGINAL-SPACE arrays (BLOCK) =====================
    // v35i: pca8w word-pack REVERTED (v35f synthesis: mux -4.4K but
    // PCA_HIST +5.8K LUT/+13K FF from in-module word assembly — net loss).
    pca_t pca8[N][PCA_DIM];
#pragma HLS ARRAY_PARTITION variable=pca8 block factor=PAR dim=1
#pragma HLS ARRAY_PARTITION variable=pca8 complete dim=2
    buck_t buck[N];
#pragma HLS ARRAY_PARTITION variable=buck block factor=PAR dim=1

    // Self-cleaning packed histogram (v33-W1, proven): static => zero at
    // power-on; RANK re-zeroes every word this event touched.
    static histw_t hist[PAR][HW];
#pragma HLS ARRAY_PARTITION variable=hist complete dim=1
#pragma HLS BIND_STORAGE   variable=hist type=ram_t2p

    // v35b: {pres, val} packed tables for the two private-table scatters (T2).
    // v35i: v35h's srt_os->srt_val fold REVERTED (synthesis: BRAM 712->1192
    // — the 139b word doubled every bank's BRAM count — and the 139b-wide
    // EMIT chain cost EMIT_EDGE +22.5K). Separate srt_os LUTRAM returns.
    ap_uint<11> srt_os[PAR][N];
#pragma HLS ARRAY_PARTITION variable=srt_os complete dim=1
#pragma HLS ARRAY_PARTITION variable=srt_os cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE   variable=srt_os type=ram_2p impl=lutram
    ap_uint<128> srt_val[PAR][N];     // 8x pca_t packed, placed at its sorted rank
#pragma HLS ARRAY_PARTITION variable=srt_val complete dim=1
#pragma HLS ARRAY_PARTITION variable=srt_val cyclic factor=8 dim=2
#pragma HLS BIND_STORAGE   variable=srt_val type=ram_t2p impl=bram
    // co = {pres, co_val}: G tables (JUMP2's grain-16 lanes write at
    // arbitrary order[k]); UNSORT_EM reads at grain 16 -> bank j STATIC.
    // Cleared (pres=0) in PCA_HIST.
    ap_uint<11> co[G][N];
#pragma HLS ARRAY_PARTITION variable=co complete dim=1
#pragma HLS ARRAY_PARTITION variable=co cyclic factor=16 dim=2
    // v35g: grain-32 UNSORT reads each bank TWICE per tick (addresses 2t and
    // 2t+1) -> one-write/n-read LUTRAM (writes stay single: PCA_HIST clear /
    // JUMP2 scatter are 1 per table per tick).
#pragma HLS BIND_STORAGE   variable=co type=ram_1wnr impl=lutram

    // =====================================================================
    // STEP 1 (BLOCK): PCA + packed lane-private histogram + pres clears.
    //   Clears run at grain 16 (i = t*16+j): bank j is compile-time constant,
    //   so the fold costs wires, not muxes (v33 paid +33K LUT for this same
    //   fold at 15-grain/runtime-bank). NG=38 <= M for LANES<=15.
    // =====================================================================
    PCA_HIST: for (int t = 0; t < M; t++) {
#pragma HLS PIPELINE II=1
        for (int par = 0; par < PAR; par++) {
            int i = par * M + t;                       // block assignment
            // v35i: hoist emb-MEAN out of the d-loop (exact ops, values
            // identical; v35a report showed 660 subs/lane-group = the tool
            // did NOT share them across the 8 dims)
            ap_fixed<9, 5> de[EMB_DIM];
            for (int j = 0; j < EMB_DIM; j++) de[j] = emb[i][j] - MEAN[j];
            PCA_D: for (int d = 0; d < PCA_DIM; d++) {
                // v35h: (20,12) accumulator — products are (16,8) and
                // |sum| <= 12*128 < 2048, so neither format ever wraps and
                // the final pca_t conversion sees identical bits. The old
                // acc_t(32,16) paid ~12 wasted bits on every adder.
                ap_fixed<20, 12> sum = 0;
                PCA_J: for (int j = 0; j < EMB_DIM; j++) {
                    sum = sum + de[j] * COMPONENTS[d][j];
                }
                pca8[i][d] = sum;
            }
            // fixed-point floor(key*4): AP_TRN truncates toward -inf (np.floor)
            ap_fixed<18, 18, AP_TRN> key4 = pca8[i][0] * 4;
            int bucket = (int)key4;
            int idx    = bucket + BUCKET_OFFSET;       // 0..HIST_SIZE-1
            buck[i]    = (buck_t)idx;
            // packed-word increment: one single-address RMW on one array; an
            // 8-bit field never carries out (a lane count <= M <= 255)
            hist[par][idx >> 2] = hist[par][idx >> 2] + ((histw_t)1 << (8 * (idx & 3)));
        }
        if (t < NG) {                                  // folded pres clears
            for (int j = 0; j < G; j++) {
                int i = t * G + j;
                if (i < N) {
                    for (int q = 0; q < PAR; q++) srt_os[q][i] = 0;
                    for (int q = 0; q < G;   q++) co[q][i]     = 0;
                }
            }
        }
    }

    // =====================================================================
    // STEP 2 (fixed W2): packed exclusive prefix, 8 buckets/clock, ~21+d ticks.
    //   off[p][w] packs base[k]+lane_off[p][k] for the word's 4 buckets — the
    //   only combination RANK needs.
    //   TIMING FIX vs v33: every prefix/total below is feed-forward from
    //   hist[] only; `run` joins each offset with ONE trailing add, and the
    //   loop-carried path is the single final `run += W01`. (v33 seeded the
    //   chain with s=run, putting ~120 serial adds on the carried path ->
    //   70-87ns. v31's own BASE proves this feed-forward shape closes 7ns.)
    // =====================================================================
    offw_t off[PAR][HW];
#pragma HLS ARRAY_PARTITION variable=off complete dim=1
#pragma HLS BIND_STORAGE   variable=off type=ram_t2p
    histw_t lane_cur[PAR][HW];
#pragma HLS ARRAY_PARTITION variable=lane_cur complete dim=1
#pragma HLS BIND_STORAGE   variable=lane_cur type=ram_t2p

    cnt_t run = 0;
    BASE: for (int wp = 0; wp < HW / 2; wp++) {
#pragma HLS PIPELINE II=1
        histw_t h0[PAR], h1[PAR];
#pragma HLS ARRAY_PARTITION variable=h0 complete
#pragma HLS ARRAY_PARTITION variable=h1 complete
        for (int p = 0; p < PAR; p++) {
            h0[p] = hist[p][2 * wp];
            h1[p] = hist[p][2 * wp + 1];
        }
        // per-field exclusive lane prefixes + totals (all feed-forward)
        cnt_t pre0[4][PAR], pre1[4][PAR], T0[4], T1[4];
#pragma HLS ARRAY_PARTITION variable=pre0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=pre1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=T0 complete
#pragma HLS ARRAY_PARTITION variable=T1 complete
        for (int f = 0; f < 4; f++) {
            cnt_t s0 = 0, s1 = 0;
            for (int p = 0; p < PAR; p++) {
                pre0[f][p] = s0; s0 = s0 + (cnt_t)(ap_uint<8>)(h0[p] >> (8 * f));
                pre1[f][p] = s1; s1 = s1 + (cnt_t)(ap_uint<8>)(h1[p] >> (8 * f));
            }
            T0[f] = s0; T1[f] = s1;
        }
        // field bases: exclusive over previous fields (and word0 for word1)
        cnt_t B0[4], B1[4];
        B0[0] = 0;
        for (int f = 1; f < 4; f++) B0[f] = B0[f - 1] + T0[f - 1];
        cnt_t W0 = B0[3] + T0[3];
        B1[0] = W0;
        for (int f = 1; f < 4; f++) B1[f] = B1[f - 1] + T1[f - 1];
        cnt_t W01 = B1[3] + T1[3];
        // assemble packed offsets: run joins only here (one trailing add)
        for (int p = 0; p < PAR; p++) {
            offw_t ow0 = 0, ow1 = 0;
            for (int f = 0; f < 4; f++) {
                ow0 |= ((offw_t)(cnt_t)(run + B0[f] + pre0[f][p])) << (10 * f);
                ow1 |= ((offw_t)(cnt_t)(run + B1[f] + pre1[f][p])) << (10 * f);
            }
            off[p][2 * wp]     = ow0;
            off[p][2 * wp + 1] = ow1;
            lane_cur[p][2 * wp] = 0; lane_cur[p][2 * wp + 1] = 0;
        }
        run = run + W01;                       // the ONLY loop-carried add
    }

    // ==================== PRIVATE TABLES for reorder (T2) ====================
    // (v35h: srt_val = {pres, ord, payload} declared before PCA_HIST above)

    // =====================================================================
    // STEP 3 (BLOCK): rank + private-table scatter + histogram self-clean.
    //   (v33C body verbatim — proven 5.1ns / 7.3K LUT / II=1.)
    // =====================================================================
    RANK: for (int t = 0; t < M; t++) {
#pragma HLS PIPELINE II=1
        for (int par = 0; par < PAR; par++) {
            int i  = par * M + t;
            int k  = buck[i];
            int kw = k >> 2, kf = k & 3;
            cnt_t      o  = (cnt_t)(off[par][kw] >> (10 * kf)); // base+lane_off
            histw_t    cw = lane_cur[par][kw];
            ap_uint<8> c  = (ap_uint<8>)(cw >> (8 * kf));       // lane cursor
            int rank = (int)o + (int)c;
            lane_cur[par][kw] = cw + ((histw_t)1 << (8 * kf));
            hist[par][kw] = 0;             // W1 self-clean: BASE is done with it
            ap_uint<128> w = 0;
            for (int d = 0; d < PCA_DIM; d++)
                w.range(16 * d + 15, 16 * d) = pca8[i][d].range(15, 0);
            srt_val[par][rank] = w;
            srt_os[par][rank]  = ((ap_uint<11>)1 << 10) | (ap_uint<10>)i;
        }
    }

    // ================= SORTED-SPACE arrays (GRAIN-16 CYCLIC) =================
    pca_t pca8_sorted[N][PCA_DIM];
#pragma HLS ARRAY_PARTITION variable=pca8_sorted cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=pca8_sorted complete dim=2
    // v35i: order word-packed (EMIT writes group word, JUMP2 reads word t —
    // both group-sequential; was 16 banks)
    ap_uint<160> order_w[NG];
#pragma HLS BIND_STORAGE variable=order_w type=ram_2p impl=bram
    // v35c: comp_a word-packed (group u = one 160b word, EMIT writes/FW1 reads)
    ap_uint<160> comp_aw[NG];
#pragma HLS BIND_STORAGE variable=comp_aw type=ram_2p impl=bram
    // v35c: edge_mask word-packed (group = 192b word, bit j*K+s), one replica
    // per FW/BW consumer stage so each needs only 1 read port.
    // v35f: ONE edge-word array. FW1 reads word t; every other stage's word
    // (t-2/t-3/t-4/t-5) is something FW1 already read — a 5-deep 192b shift
    // register chain inside FW_CHAIN hands them down. em2/em3/em4 deleted.
    ap_uint<192> em1[NG];
#pragma HLS BIND_STORAGE variable=em1 type=ram_2p impl=bram

    // =====================================================================
    // STEP 4+5: EMIT (grain 16) fused with FULL-WIDTH EDGE (all 8 dims per
    //   tick — with fabric squares there are no DSPs left to phase-split;
    //   v31 split EDGE only to halve DSPs). Skew=2 as in v31: EDGE for
    //   group eu reads positions up to eu*16+27 < (eu+2)*16, i.e. at most
    //   one group ahead, which EMIT wrote at tick eu+1 <= u-1.
    // =====================================================================
    const int ESKEW = 2;
    EMIT_EDGE: for (int u = 0; u < NG + ESKEW; u++) {
#pragma HLS PIPELINE II=1
        if (u < NG) {                                   // EMIT for group u
            ap_uint<160> caw = 0, ow = 0;
            for (int j = 0; j < G; j++) {
                int m = u * G + j;
                caw.range(10 * j + 9, 10 * j) = (ap_uint<10>)m;  // fused LPA_INIT
                if (m < N) {
                    // v35j: encode the (exactly-one) owner lane as a 4-bit
                    // index first (narrow OR is fine — the v35e failure was
                    // OR-ing the 128b payload), then ONE indexed 15:1 tree
                    // mux picks the payload — replaces the 15-deep 128b
                    // select chain (~29K LUT of w_* selects in v35f).
                    ap_uint<4>  own = 0;
                    ap_uint<10> oid_n = 0;
                    for (int q = 0; q < PAR; q++) {
                        ap_uint<11> so = srt_os[q][m];
                        if (so[10]) { own |= (ap_uint<4>)q; oid_n |= (ap_uint<10>)so; }
                    }
                    ap_uint<128> lv[PAR];
#pragma HLS ARRAY_PARTITION variable=lv complete
                    for (int q = 0; q < PAR; q++) lv[q] = srt_val[q][m];
                    int oid = (int)oid_n;
                    ap_uint<128> w = lv[own];      // static bank j%8, 2/bank
                    ow.range(10 * j + 9, 10 * j) = (ap_uint<10>)oid;
                    for (int d = 0; d < PCA_DIM; d++) {
                        pca_t v;
                        v.range(15, 0) = w.range(16 * d + 15, 16 * d);
                        pca8_sorted[m][d] = v;
                    }
                }
            }
            comp_aw[u] = caw;
            order_w[u] = ow;
        }
        int eu = u - ESKEW;
        if (eu >= 0 && eu < NG) {                       // EDGE for group eu
            ap_uint<192> ew = 0;                        // tail lanes stay 0 (=no edge)
            for (int j = 0; j < G; j++) {
                int i = eu * G + j;
                if (i < N)                              // grain-16 tail guard
                for (int s = 0; s < K; s++) {
                    bool inb = (i + s + 1 < N);
                    acc_t d2 = 0;
                    if (inb) {
                        for (int d = 0; d < PCA_DIM; d++) {
                            ap_fixed<15, 7> diff = pca8_sorted[i][d] - pca8_sorted[i + s + 1][d];
                            ap_fixed<30, 14> sq;         // exact 15x15 product
#pragma HLS BIND_OP variable=sq op=mul impl=dsp latency=2
                            sq = diff * diff;            // v31 arithmetic, DSP lat 2
                            d2 += sq;                    // (v35a's lat-3 cost +2 cycles)
                        }
                    }
                    ew[j * K + s] = inb && (d2 < (acc_t)0.09);
                }
            }
            em1[eu] = ew;
        }
    }

    // =====================================================================
    // STEP 6: LPA — FW1/BW1/FW2/BW2, v31's 4-stage cascaded skew at grain 16.
    //   Skews (0,2,3,5) still valid: forward reach s+1 <= K=12 < 16 spans at
    //   most one group ahead, same bound as v31's 12 < 15.
    // =====================================================================
    // v35c: comp_b1/comp_a2/comp_b2 are handed stage-to-stage through 2-deep
    // word shift registers inside FW_CHAIN (see loop below) — no arrays.
    // v35d: comp_r gets a second word replica so JUMP1 can read words t,
    // t+18 (t2p) and t+19 (replica) in one tick — no register prologue, the
    // +1 cycle from v35c comes back.
    ap_uint<160> comp_rw[NG], comp_rw2[NG];
#pragma HLS BIND_STORAGE variable=comp_rw  type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=comp_rw2 type=ram_2p  impl=bram
    // v35d: JUMP1's gather copies word-packed — 32 LUTRAM word tables (16
    // for the k1 gathers, 16 for k2) instead of 2x256 single-position banks.
    // BW2 writes each table with ONE whole-word write per tick (was 16
    // per-bank writes); JUMP1 reads one runtime word + a 16:1 lane select.
    // v35f: 16 t2p BRAM replicas (2 gather reads each) instead of 32 LUTRAM
    // word tables — trades ~160 BRAM18 (headroom is large) for the LUTRAM.
    ap_uint<160> j1tab[G][NG];
#pragma HLS ARRAY_PARTITION variable=j1tab complete dim=1
#pragma HLS BIND_STORAGE   variable=j1tab type=ram_t2p impl=bram

    // Stage-to-stage word registers. Shifted unconditionally every tick, so at
    // tick t: w1_d1 = FW1's word t-1, w1_d2 = word t-2; w2_d1 = BW1's word
    // t-3, w2_d2 = word t-4; w3_d1 = FW2's word t-4, w3_d2 = word t-5.
    // ca_prev/e1_prev/e3_prev carry the previous word of the backward-reading
    // stages' second source. Garbage lanes (beyond N or from inactive ticks)
    // are masked by the same jj>=0 / jj<N / k<N guards as before.
    ap_uint<160> ca_prev = 0, w1_d1 = 0, w1_d2 = 0, w2_d1 = 0, w2_d2 = 0,
                 w3_d1 = 0, w3_d2 = 0;
    // v35f: edge words handed down a 5-deep shift chain: e1_dN = em1 word t-N.
    ap_uint<192> e1_d1 = 0, e1_d2 = 0, e1_d3 = 0, e1_d4 = 0, e1_d5 = 0;
    FW_CHAIN: for (int t = 0; t <= NG + 5; t++) {
#pragma HLS PIPELINE II=1
        ap_uint<160> ca_cur = 0;
        ap_uint<192> e1_cur = 0;
        if (t < NG)              { ca_cur = comp_aw[t]; e1_cur = em1[t]; }
        ap_uint<160> w1_new = 0, w2_new = 0, w3_new = 0, wr_new = 0;
        if (t < NG) {                                   // FW1 for group t
            for (int j = 0; j < G; j++) {
                int k = t * G + j;
                ap_uint<10> best = ca_cur.range(10 * j + 9, 10 * j);
                for (int s = 0; s < K; s++) {
                    int jj = k - s - 1;
                    int lj = j - s - 1;                 // lane: word t if >=0, else word t-1
                    bool eb;
                    ap_uint<10> v;
                    if (lj >= 0) { eb = e1_cur[lj * K + s];
                                   v  = ca_cur.range(10 * lj + 9, 10 * lj); }
                    else         { eb = e1_d1[(lj + G) * K + s];
                                   v  = ca_prev.range(10 * (lj + G) + 9, 10 * (lj + G)); }
                    if (jj >= 0 && eb && v < best) best = v;
                }
                w1_new.range(10 * j + 9, 10 * j) = best;
            }
        }
        if (t >= 2 && t - 2 < NG) {                     // BW1 for group tt=t-2
            int tt = t - 2;                             // word tt = w1_d2, word tt+1 = w1_d1
            for (int j = 0; j < G; j++) {
                int k = tt * G + j;
                ap_uint<10> best = w1_d2.range(10 * j + 9, 10 * j);
                for (int s = 0; s < K; s++) {
                    int jj = k + s + 1;
                    int lj = j + s + 1;
                    bool eb = e1_d2[j * K + s];
                    ap_uint<10> v = (lj < G)
                        ? (ap_uint<10>)w1_d2.range(10 * lj + 9, 10 * lj)
                        : (ap_uint<10>)w1_d1.range(10 * (lj - G) + 9, 10 * (lj - G));
                    if (jj < N && eb && v < best) best = v;
                }
                w2_new.range(10 * j + 9, 10 * j) = best;
            }
        }
        if (t >= 3 && t - 3 < NG) {                     // FW2 for group tt=t-3
            int tt = t - 3;                             // word tt = w2_d1, word tt-1 = w2_d2
            for (int j = 0; j < G; j++) {
                int k = tt * G + j;
                ap_uint<10> best = w2_d1.range(10 * j + 9, 10 * j);
                for (int s = 0; s < K; s++) {
                    int jj = k - s - 1;
                    int lj = j - s - 1;
                    bool eb;
                    ap_uint<10> v;
                    if (lj >= 0) { eb = e1_d3[lj * K + s];
                                   v  = w2_d1.range(10 * lj + 9, 10 * lj); }
                    else         { eb = e1_d4[(lj + G) * K + s];
                                   v  = w2_d2.range(10 * (lj + G) + 9, 10 * (lj + G)); }
                    if (jj >= 0 && eb && v < best) best = v;
                }
                w3_new.range(10 * j + 9, 10 * j) = best;
            }
        }
        if (t >= 5 && t - 5 < NG) {                     // BW2 for group tt=t-5
            int tt = t - 5;                             // word tt = w3_d2, word tt+1 = w3_d1
            for (int j = 0; j < G; j++) {
                int k = tt * G + j;
                ap_uint<10> best = w3_d2.range(10 * j + 9, 10 * j);
                for (int s = 0; s < K; s++) {
                    int jj = k + s + 1;
                    int lj = j + s + 1;
                    bool eb = e1_d5[j * K + s];
                    ap_uint<10> v = (lj < G)
                        ? (ap_uint<10>)w3_d2.range(10 * lj + 9, 10 * lj)
                        : (ap_uint<10>)w3_d1.range(10 * (lj - G) + 9, 10 * (lj - G));
                    if (jj < N && eb && v < best) best = v;
                }
                wr_new.range(10 * j + 9, 10 * j) = best;
            }
            comp_rw[tt] = wr_new;
            comp_rw2[tt] = wr_new;
            for (int q = 0; q < G; q++) j1tab[q][tt] = wr_new;
        }
        ca_prev = ca_cur;
        e1_d5 = e1_d4;  e1_d4 = e1_d3;  e1_d3 = e1_d2;  e1_d2 = e1_d1;  e1_d1 = e1_cur;
        w1_d2 = w1_d1;  w1_d1 = w1_new;
        w2_d2 = w2_d1;  w2_d1 = w2_new;
        w3_d2 = w3_d1;  w3_d1 = w3_new;
    }

    // =====================================================================
    // STEP 6b: JUMP1 double-pumped (v33-B's proven S2 trick, grain-16 form):
    //   each grain lane j handles positions k1 = t*16+j and k2 = k1+300 in
    //   the same tick. comp_b and the JUMP2 gather table are LO/HI split so
    //   every write hits its own static bank j (300 % 16 = 12 would
    //   otherwise fold two writers onto one bank).
    // =====================================================================
    // v35d: comp_b halves and JUMP2's gather copies word-packed. JUMP1
    // assembles its 16 v1/v2 results into two 160b words and writes them
    // once per tick (was 16+16 per-bank writes into 512 banks).
    ap_uint<160> cbLw[MJ], cbHw[MJ];  // comp_b for k<300 / k>=300, word t = k1 group
#pragma HLS BIND_STORAGE variable=cbLw type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=cbHw type=ram_2p impl=bram
    ap_uint<160> j2tabL[G][MJ];       // JUMP2 gather word tables, address halves
#pragma HLS ARRAY_PARTITION variable=j2tabL complete dim=1
#pragma HLS BIND_STORAGE   variable=j2tabL type=ram_2p impl=lutram
    ap_uint<160> j2tabH[G][MJ];
#pragma HLS ARRAY_PARTITION variable=j2tabH complete dim=1
#pragma HLS BIND_STORAGE   variable=j2tabH type=ram_2p impl=lutram

    // Position k1 = t*16+j sits in comp_r word t lane j; k2 = k1+300 sits in
    // word t+18 lane j+12 (j<4) or word t+19 lane j-4 (j>=4) — compile-time
    // slices. Words t and t+18 come from comp_rw's two ports, t+19 from the
    // comp_rw2 replica: 3 word reads/tick, no prologue (trip stays 19).
    JUMP1: for (int t = 0; t < MJ; t++) {           // comp_b[i] = comp_r[comp_r[i]]
#pragma HLS PIPELINE II=1
        ap_uint<160> lo   = comp_rw[t];
        ap_uint<160> hi18 = comp_rw[t + 18];        // words 18..36
        ap_uint<160> hi19 = comp_rw2[t + 19];       // words 19..37
        ap_uint<160> v1w = 0, v2w = 0;
        for (int j = 0; j < G; j++) {
            int k1 = t * G + j;
            if (k1 < NH) {
                ap_uint<10> a1 = lo.range(10 * j + 9, 10 * j);
                ap_uint<10> a2 = (j < 4)
                    ? (ap_uint<10>)hi18.range(10 * (j + 12) + 9, 10 * (j + 12))
                    : (ap_uint<10>)hi19.range(10 * (j - 4) + 9, 10 * (j - 4));
                ap_uint<160> wA = j1tab[j][a1 >> 4];      // t2p port 0
                ap_uint<160> wB = j1tab[j][a2 >> 4];      // t2p port 1
                ap_uint<4> sA = a1 & 15, sB = a2 & 15;
                // v35j: lane picks as indexed reads of a fully-partitioned
                // local view -> one 16:1 tree mux each (the v35d if(s==q)
                // chain synthesized 2x255 twelve-LUT icmps in JUMP1 alone)
                ap_uint<10> lnA[G], lnB[G];
#pragma HLS ARRAY_PARTITION variable=lnA complete
#pragma HLS ARRAY_PARTITION variable=lnB complete
                for (int q = 0; q < G; q++) {
                    lnA[q] = wA.range(10 * q + 9, 10 * q);
                    lnB[q] = wB.range(10 * q + 9, 10 * q);
                }
                ap_uint<10> v1 = lnA[sA], v2 = lnB[sB];
                v1w.range(10 * j + 9, 10 * j) = v1;
                v2w.range(10 * j + 9, 10 * j) = v2;
            }
        }
        cbLw[t] = v1w;
        cbHw[t] = v2w;
        for (int q = 0; q < G; q++) { j2tabL[q][t] = v1w; j2tabH[q][t] = v2w; }
    }

    // =====================================================================
    // STEP 7: JUMP2 (comp_b[comp_b[i]]) fused with the unsort scatter (v31).
    //   Gather reads both halves clamped, then selects (v33-B's trick).
    // =====================================================================
    // (v35b: co_val lives inside co = {pres, val}, declared above)
    // v35d word reads: position k = t*16+j. For k<300, b sits in cbLw word t
    // lane j. For k>=300, rel = k-300 sits in cbHw word t-19 lane j+4 (j<12)
    // or word t-18 lane j-12 (j>=12) — bH_prev carries word t-19 (last
    // tick's t-18 read; first H user is t=18/j>=12 which only needs t-18).
    ap_uint<160> bH_prev = 0;
    JUMP2: for (int t = 0; t < NG; t++) {           // comp_a[i] = comp_b[comp_b[i]]
#pragma HLS PIPELINE II=1
        ap_uint<160> bLw = (t < MJ) ? (ap_uint<160>)cbLw[t] : (ap_uint<160>)0;
        ap_uint<160> bHc = (t >= 18) ? (ap_uint<160>)cbHw[t - 18] : (ap_uint<160>)0;
        ap_uint<160> ordw = order_w[t];
        for (int j = 0; j < G; j++) {
            int k = t * G + j;
            if (k < N) {
                ap_uint<10> b;
                if (k < NH)      b = bLw.range(10 * j + 9, 10 * j);
                else if (j < 12) b = bH_prev.range(10 * (j + 4) + 9, 10 * (j + 4));
                else             b = bHc.range(10 * (j - 12) + 9, 10 * (j - 12));
                int a  = (int)b;
                int aL = (a < NH) ? a : (NH - 1);           // clamp
                int aH = (a >= NH) ? (a - NH) : 0;          // clamp
                ap_uint<160> wL = j2tabL[j][aL >> 4];
                ap_uint<160> wH = j2tabH[j][aH >> 4];
                ap_uint<4> sL = aL & 15, sH = aH & 15;
                ap_uint<10> lnL[G], lnH[G];   // v35j: indexed 16:1 tree muxes
#pragma HLS ARRAY_PARTITION variable=lnL complete
#pragma HLS ARRAY_PARTITION variable=lnH complete
                for (int q = 0; q < G; q++) {
                    lnL[q] = wL.range(10 * q + 9, 10 * q);
                    lnH[q] = wH.range(10 * q + 9, 10 * q);
                }
                ap_uint<10> vL = lnL[sL], vH = lnH[sH];
                idx_t v = (a < NH) ? (idx_t)vL : (idx_t)vH;
                int oi = (int)(ap_uint<10>)ordw.range(10 * j + 9, 10 * j);
                co[j][oi] = ((ap_uint<11>)1 << 10) | v;  // mask bank-decode write
            }
        }
        bH_prev = bHc;
    }

    // =====================================================================
    // STEP 8: emit labels = unsorted connected-component root.
    //   v35g: GRAIN-32 — UNSORT is pure read+emit (no DSP, tiny logic), so
    //   doubling its width is the cheapest -19 cycles in the design. Bank
    //   (i&15) stays a compile-time constant per lane; each co bank now
    //   serves two addresses (2t / 2t+1) per tick via ram_1wnr's read ports;
    //   labels is factor-32 partitioned so all 32 writes are static-bank.
    // =====================================================================
    UNSORT_EM: for (int t = 0; t < (N + 31) / 32; t++) {   // 19 ticks (was 38)
#pragma HLS PIPELINE II=1
        for (int j = 0; j < 32; j++) {
            int i = t * 32 + j;
            if (i < N) {
                // v35k: encoded-owner pick (same shape as v35j's EMIT fix):
                // OR the 4-bit owner index, then one indexed 16:1 tree mux.
                ap_uint<4> own = 0;
                ap_uint<10> lnv[G];
#pragma HLS ARRAY_PARTITION variable=lnv complete
                for (int q = 0; q < G; q++) {
                    ap_uint<11> cw = co[q][i];
                    lnv[q] = (ap_uint<10>)cw;
                    if (cw[10]) own |= (ap_uint<4>)q;
                }
                labels[i] = (int)lnv[own];
            }
        }
    }
}

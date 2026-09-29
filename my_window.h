#ifndef MY_WINDOW_H
#define MY_WINDOW_H

#include "ap_fixed.h"
#include "ap_int.h"

// ---- Problem dimensions (from the golden reference) ----
static const int N        = 600;   // hits per event
static const int EMB_DIM  = 12;    // NN embedding dimension (input)
static const int PCA_DIM  = 8;     // PCA output dimension
static const int K        = 12;    // sliding-window max shift

// ---- Algorithm constants (from golden_reference.py) ----
static const int   BUCKET_OFFSET = 60;
// Principled worst-case bucket range (see main change.txt v15/v24): the index
// floor(key*4)+60 lies in [4,164] for ALL inputs; HIST_SIZE=168 covers it.
// Must stay divisible by 8 (BASE handles two 4-bucket words per clock).
static const int   HIST_SIZE     = 168;

// ---- Parallelism ----
// LANES: block-space lanes (PCA/RANK/private sort tables). 15 here.
#ifndef LANES
#define LANES 15
#endif
#if (N % LANES) != 0
#error "LANES must divide N (=600) for the block-distributed bucket sort"
#endif
#if (N / LANES) > 255
#error "packed histogram uses 8-bit per-lane counters: need M = N/LANES <= 255"
#endif
// GRAIN: sorted-space grain, fixed at 16 INDEPENDENT of LANES (v34 core idea).
// Every sorted-space loop iterates 16 consecutive positions per tick, so with
// cyclic factor=16 partitions the bank index (i & 15) is a compile-time
// constant per unrolled lane -> ZERO runtime bank muxes (the v33 LUT killer),
// while arbitrary-address writes still decode banks with a mask (no %15
// dividers, the v31 RANK killer).
static const int GRAIN = 16;

// ---- Fixed-point types ----
typedef ap_fixed<8, 4, AP_RND_CONV, AP_SAT> emb_t;
typedef ap_fixed<32, 16> acc_t;
typedef ap_fixed<16, 8> pca_t;
typedef ap_uint<10> idx_t;
typedef ap_uint<10> cnt_t;
typedef ap_uint<9> buck_t;
// histw_t: FOUR neighbouring histogram buckets packed into one 32-bit word
//   (8-bit lane-private counts, bound M=N/LANES<=255). Packing is NOT banking:
//   the PCA_HIST increment stays one single-address RMW on one array, so the
//   II=1 recurrence is untouched (banking was proven to force II=2 — change.txt
//   v3 + v27).
typedef ap_uint<32> histw_t;
// offw_t: FOUR packed 10-bit "base[k]+lane_off[par][k]" offsets (<= N=600).
typedef ap_uint<40> offw_t;

// ---- DUT top function ----
void my_window(emb_t emb[N][EMB_DIM], int labels[N]);

#endif

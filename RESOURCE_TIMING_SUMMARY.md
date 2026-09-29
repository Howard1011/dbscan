# my_window (v35k) — Resource and Timing Summary

**Platform** xcu250-figd2104-2L-e (Alveo U250) ｜ **Target clock** 10 ns (100 MHz) ｜ **Critical path** 7.289 ns → up to **137 MHz** ｜ Source: v35k synthesis report (`my_window_proj/solution1/syn/report/csynth.rpt`)

---

## 1. Overview

| Metric | Value |
|---|---|
| **Total latency** | **308 cycles** |
| Execution time @ 100 MHz (target) | 3.08 µs |
| Execution time @ 137 MHz (achievable) | 2.25 µs |
| **Initiation Interval (II)** | **1** (all loops, fully pipelined) |
| Timing | Critical path 7.289 ns (timing met, slack > 0) |
| Correctness | **Partition identical** to golden (verified on 3 events); labels are component-root values |

### Resource usage (U250)

| Resource | Used | Share | Available |
|---|---:|---:|---:|
| **BRAM** | 720 | **13 %** | 5,376 (18K) |
| **DSP** | 1,566 | **12 %** | 12,288 |
| **FF** | 40,663 | **1 %** | 3,456,000 |
| **LUT** | 224,200 | **12 %** | 1,728,000 |

> All four resources are ≤ 15%, which was one of the original design goals.

---

## 2. Per-stage timing (how the 308 cycles add up)

Stages run **back to back (no overlap)** because of data dependencies between them; inside each stage the loop is **pipelined with II=1**. Total latency is therefore roughly the sum of the stages.

| # | Stage | What it does | Trip count | Iteration latency | II | **Stage latency (cyc)** | Share of 308 |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | PCA_HIST | PCA (12→8) + build histogram | 40 | 5 | 1 | **45** | 14.6 % |
| 2 | BASE | Counting-sort prefix sum (rank start offsets) | 21 | 4 | 1 | **25** | 8.1 % |
| 3 | RANK | Compute ranks + scatter to sorted positions | 40 | 3 | 1 | **43** | 14.0 % |
| 4 | EMIT_EDGE | Read back sorted data + build edges (distance test) | 40 | 6 | 1 | **46** | 14.9 % |
| 5 | FW_CHAIN | Label propagation (4 sweeps) | 44 | 6 | 1 | **50** | 16.2 % |
| 6 | JUMP1 | Pointer jumping #1 (double-pumped) | 19 | 3 | 1 | **22** | 7.1 % |
| 7 | JUMP2 | Pointer jumping #2 + un-sort | 38 | 3 | 1 | **41** | 13.3 % |
| 8 | UNSORT_EM | Emit labels (grain 32) | 19 | 2 | 1 | **21** | 6.8 % |
| — | Stage hand-off / entry-exit | Function entry/exit + stage transitions | — | — | — | **≈15** | 4.9 % |
| | | | | | | **Total 308** | 100 % |

- "Iteration latency" is the number of cycles one batch of data takes from entering to leaving the stage.
- "Stage latency" = iteration latency + (trip count − 1) × II, i.e. the cycles the pipeline needs to push the whole stage through.
- The three most expensive stages are **FW_CHAIN (50) + EMIT_EDGE (46) + PCA_HIST (45)**, about 46% of the total.

---

## 3. Per-stage resources (which stage uses what)

| # | Stage | DSP | FF | LUT | LUT share |
|---|---|---:|---:|---:|---:|
| 1 | PCA_HIST | 30 | 6,835 | 33,078 | 14.8 % |
| 2 | BASE | 0 | 1,596 | 4,210 | 1.9 % |
| 3 | RANK | 0 | 3,293 | 7,285 | 3.2 % |
| 4 | **EMIT_EDGE** | **1,536** | 5,068 | **69,153** | **30.8 %** |
| 5 | FW_CHAIN | 0 | 5,180 | 28,008 | 12.5 % |
| 6 | JUMP1 | 0 | 150 | 3,297 | 1.5 % |
| 7 | JUMP2 | 0 | 335 | 4,291 | 1.9 % |
| 8 | UNSORT_EM | 0 | 14 | 4,671 | 2.1 % |
| — | Top-level mux / glue | 0 | ~18,000 | ~70,000 | ~31 % |
| | **Total** | **1,566** | **40,663** | **224,200** | 100 % |

**Key points**
- **Almost all DSPs are in EMIT_EDGE (1,536 of 1,566).** Edge building computes variable × variable squared distances (16 positions × 12 neighbors × 8 dimensions = 1,536 multiplications), which need real multipliers.
- **PCA uses only 30 DSPs.** PCA multiplies by fixed constants, so the tool implements it with LUT shift-and-add instead of DSPs.
- **The largest LUT consumer is also EMIT_EDGE (69K)**, followed by PCA_HIST (33K) and FW_CHAIN (28K).
- BRAM (720) is mainly the large arrays shared at the top level (`srt_val`, `pca8_sorted`, `co`, the various word arrays) and is not broken down per stage.

---

## 4. Optimization results (before → after)

From the fast15 starting point to v35k, **speed is almost unchanged while area drops sharply**. The core strategy is trading idle DSPs for LUTs.

| Metric | fast15 (start) | **v35k (final)** | Change |
|---|---:|---:|---:|
| Latency | 325 cyc | **308 cyc** | −5 % |
| LUT | 539,067 | **224,200** | **−58 %** |
| FF | 108,617 | **40,663** | **−63 %** |
| DSP | 30 | **1,566** | idle DSPs used in place of LUTs |
| BRAM | 540 | 720 | +33 % (still 13%, within target) |

**Latency history (cycles):** scalar 7,672 → HEPT-style 2,154 → v31 575 → fast15 325 → **v35k 308** (about **25× faster** than the original).

All numbers come from `my_window_proj/solution1/syn/report/csynth.rpt`. For the algorithm itself see [ALGORITHM.md](ALGORITHM.md).

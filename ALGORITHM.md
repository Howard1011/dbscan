# my_window — Clustering Algorithm Specification and Stability Analysis

This document describes the clustering method that the HLS core `my_window` (v35k) **actually computes**. All hardware parallelization tricks are stripped out; only the computed values are specified. It is meant for anyone who wants to reimplement the method in plain software, test it at scale, and judge **whether it is sound and where it can become unstable**.

The Python reference in §3 was run against the bundled `event500/501/502` data: all three events give `same_partition=True`.

The hardware optimizations in the v35 series (latency 308 cycles, II=1) did **not** change the algorithm. Output is bit-identical to earlier versions, so the method below is the only thing you need to validate.

---

## 1. What the method does

It clusters 600 hits (each a 12-dimensional embedding). Sorting plus a sliding window turns the O(N²) pairwise comparison into an O(N·K) neighborhood comparison. A fixed number of label-propagation steps then approximates connected components.

**In one line:**
`PCA (12→8) → coarse sort on the 1st principal component → build edges with a width-12 sliding window in sorted space (8-D squared distance < 0.09) → fixed 2×(forward+backward) sweeps + 2 pointer-jumping rounds to approximate connected components → component root = label`

**Input / output**
- Input: `emb` (600×12 float, already quantized), `MEAN` (12,), `COMPONENTS` (8×12). The last two are fixed, pre-trained PCA parameters.
- Output: `labels` (600 ints), the clustering result.

---

## 2. The algorithm (5 steps + parameters)

| Parameter | Value | Meaning |
|---|---|---|
| N | 600 | number of hits |
| EMB_DIM | 12 | original dimension |
| PCA_DIM | 8 | dimension after PCA |
| K | 12 | window width (each point looks at the next 12 points in sorted order) |
| Q | 0.25 | bucket width for sorting, `bucket = floor(key / Q)` |
| EPS | 0.30 | distance threshold (compared as EPS² = 0.09) |
| SWEEPS | 2 | number of (forward + backward) sweep pairs |
| JUMPS | 2 | number of pointer-jumping rounds |

- **STEP 1 — PCA:** `pca8 = (emb - MEAN) @ COMPONENTS.T`, shape (600, 8).
- **STEP 2 — Coarse sort:** `bucket = floor(pca8[:,0] / 0.25)`; `order = argsort(bucket, kind="stable")`. Within a bucket, the smaller original index comes first.
  - Only the bucket is sorted, **not** the actual value. This is part of the method's definition (the original golden reference does the same), not a hardware shortcut.
- **STEP 3 — Edges (sorted space):** for each position `k` and `shift = 1..12`, `edge(k, k+shift) ⇔ Σ_{d=0..7} (s[k][d] − s[k+shift][d])² < 0.09`.
- **STEP 4 — Approximate connected components (sorted space):** initialize `comp[k] = k`. Repeat SWEEPS times a forward sweep followed by a backward sweep. Each sweep reads the whole `comp` array as it was *before* the sweep and replaces it in one batch (not in-place):
  - forward: `comp'[k] = min(comp[k], min{ comp[j] : j = k−12..k−1, edge(j,k) })`
  - backward: `comp'[k] = min(comp[k], min{ comp[j] : j = k+1..k+12, edge(k,j) })`

  Then do JUMPS rounds of pointer jumping: `comp = comp[comp]`.
- **STEP 5 — Back to original order:** `labels[order[k]] = comp[k]`.

### Important: label values are not `0,1,2,…`

The output label of a point is the **root of its connected component (a sorted-position index)**, not a renumbered `0,1,2,…`. All points in one cluster share the same value, but the value itself is an implementation detail.

> **Always compare clusterings with `same_partition()` (who is grouped with whom). Never compare label arrays directly with `(a == b).all()`.**

The original golden reference runs STEP 4 to convergence and renumbers labels by first appearance. This method uses fixed step counts and outputs roots directly, which suits hardware parallelization. The clustering is intended to be the same; only the label numbering differs. So against the golden files this method gives `same_partition=True` but `exact_match=False`. The bundled testbench reports the same thing.

---

## 3. Python reference implementation (verified)

```python
import numpy as np

N, PCA_DIM, K = 600, 8, 12
Q      = 0.25   # bucket = floor(key / Q)
EPS2   = 0.09   # squared-distance threshold (EPS = 0.30)
SWEEPS = 2      # 2 forward+backward sweep pairs
JUMPS  = 2      # 2 pointer-jumping rounds

def cluster_as_hw(emb, MEAN, COMPONENTS):
    """
    Mirrors what the hardware core computes (NOT the golden
    "run to convergence + renumber" version).
    emb (600,12) / MEAN (12,) / COMPONENTS (8,12), float.
    Returns labels (600,) int64. Values are component roots, not 0,1,2,...;
    compare clusterings with same_partition().
    """
    # STEP 1: PCA
    pca8 = (emb - MEAN) @ COMPONENTS.T                      # (600, 8)

    # STEP 2: approximate counting sort (stable; ties broken by original index)
    bucket = np.floor(pca8[:, 0] / Q).astype(np.int64)
    order  = np.argsort(bucket, kind="stable")             # order[k] = original index at sorted position k
    s      = pca8[order]

    # STEP 3: sliding-window edges (sorted space, K=12)
    edge = np.zeros((N, K), dtype=bool)
    for sh in range(1, K + 1):
        n  = N - sh
        d2 = np.sum((s[:n] - s[sh:sh + n]) ** 2, axis=1)
        edge[:n, sh - 1] = d2 < EPS2

    # STEP 4: fixed SWEEPS x (forward+backward) sweeps + JUMPS pointer-jumping rounds
    comp = np.arange(N)
    for _ in range(SWEEPS):
        nxt = comp.copy()                                  # forward: read comp from before the sweep
        for k in range(N):
            best = comp[k]
            for sh in range(1, K + 1):
                j = k - sh
                if j >= 0 and edge[j, sh - 1]:
                    best = min(best, comp[j])
            nxt[k] = best
        comp = nxt
        nxt = comp.copy()                                  # backward
        for k in range(N):
            best = comp[k]
            for sh in range(1, K + 1):
                j = k + sh
                if j < N and edge[k, sh - 1]:
                    best = min(best, comp[j])
            nxt[k] = best
        comp = nxt
    for _ in range(JUMPS):
        comp = comp[comp]                                  # pointer jumping

    # STEP 5: sorted space -> original index space (no renumbering)
    labels = np.zeros(N, dtype=np.int64)
    labels[order] = comp
    return labels


def same_partition(a, b):
    """True iff two labelings induce the same clustering (label values ignored).
    Use this instead of (a == b).all()."""
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal((a[:, None] == a[None, :]), (b[:, None] == b[None, :]))
```

**Run the check** from this directory (the `data/` folder contains `pca_mean.dat`, `pca_components.dat`, `emb_event*.dat` and `golden_event*.dat`):

```python
MEAN = np.loadtxt("data/pca_mean.dat")          # (12,)
COMP = np.loadtxt("data/pca_components.dat")    # (8,12)
for ev in (500, 501, 502):
    emb    = np.loadtxt(f"data/emb_event{ev}.dat")            # (600,12)
    golden = np.loadtxt(f"data/golden_event{ev}.dat", dtype=int)
    out    = cluster_as_hw(emb, MEAN, COMP)
    print(ev, "same_partition =", same_partition(out, golden))   # expect True for all three
```

---

## 4. Stability analysis: where the method can fail

The method has **two layers of approximation**, both introduced for hardware parallelism, plus fixed-point effects. These are the risk points.

### A. Coarse sort (bucket = 0.25, ties broken by original index)
- **Failure mode:** points inside one bucket are ordered by original index, not by geometry. If a bucket is crowded (more than about 12 points), the window K=12 cannot cover it. Two geometrically close points can end up more than 12 positions apart in sorted order, so the edge is missed and a cluster that should be merged gets split. Two keys that are very close but fall on opposite sides of a bucket boundary can also change relative order under small perturbations.
- **Safe when:** each 0.25-wide strip of `pca0` generally holds ≤ 12 points.
- **How to test:** measure the per-bucket point count distribution; the P99 bucket occupancy is the key indicator. Also check how many points sit close to a bucket boundary, and whether the clustering changes under small perturbations.

### B. Window width K = 12
- **Failure mode:** coupled with A. A true neighbor more than K positions away in sorted order is never seen. Clusters are also linked only through chains in sorted space, so a cluster that is geometrically continuous but folds back along `pca0` (non-monotonic) can be split into several segments after sorting.

### C. Fixed-step propagation (SWEEPS = 2, JUMPS = 2) — the most suspicious point in theory
- **Failure mode:** each sweep is a batch update, so the minimum label travels only one hop (≤ 12 positions) along edges per sweep. Four sweeps plus two pointer-jumping rounds reach a limited depth. If a component's "diameter" in sorted space (shortest path length measured in edge hops) exceeds that depth, the component outputs two or more roots (it should be connected but is not).
- **Evidence so far:** on `event500/501/502` the result matches the golden run-to-convergence version exactly as a partition. That is empirical evidence from 3 events, not a proof.
- **How to test (the most important experiment):** increase SWEEPS/JUMPS (e.g. 8/4, or run to convergence) and compare the partitions. If they change, the fixed-step approximation does hurt on your data. If they do not change, the method is stable for your data.

### D. Fixed-point quantization (affects hardware-vs-float agreement, not the method itself)
- The hardware input `emb` is `ap_fixed<8,4>`: step 0.0625, range [−8, 7.9375], saturating with rounding. PCA outputs are stored as 16-bit (truncate, wrap), differences are clamped to 15-bit (wrap), squares and accumulation are exact, and the threshold comparison is exact. The threshold 0.09 is held as `ap_fixed<32,16>`, so truncation error is about 2⁻¹⁶.
- **Failure mode:** a pair whose d² is right at the 0.09 boundary, or a point right at a bucket boundary, can be judged differently by a float implementation and the fixed-point one, flipping a cluster. This is a precision issue, not a flaw in the method. You can estimate the flip rate as the fraction of pairs with `|d² − 0.09|` below 1 ulp of the fixed-point format.
- Corner case: with 15-bit wrap, `|diff| ≥ 64` aliases and could in theory create a false edge. Real PCA values are far from this range, but keep it in mind if your data is much wider.
- `data/emb_event*.dat` already contain the quantized inputs.

### Suggested feasibility checklist
- [ ] Run the §3 reference on event500/501/502 and confirm `same_partition=True` (baseline).
- [ ] Run it on your own data and compare against your ground truth or model with `same_partition()`.
- [ ] Sweep `SWEEPS ∈ {2, 3, 4, converged}` and `JUMPS ∈ {2, 3, 4}`, and check whether the clustering is stable. This tells you whether the fixed step count is enough.
- [ ] Measure the fraction of boundary points (key within ε of a bucket boundary, or d² within ε of 0.09) to estimate quantization and threshold sensitivity.
- [ ] To reproduce the hardware bit for bit, quantize the input to `ap_fixed<8,4>`.

---

## 5. Related files

| Purpose | File |
|---|---|
| Hardware source (this document is derived from it) | `my_window.cpp` / `my_window.h` |
| PCA parameters | `data/pca_mean.dat` (12,), `data/pca_components.dat` (8,12) |
| Test inputs | `data/emb_event500/501/502.dat` (600×12 each) |
| Golden clustering | `data/golden_event500/501/502.dat` (600 ints each) |
| Resource and timing results | `RESOURCE_TIMING_SUMMARY.md` |

The hardware figures (latency 308 cycles, II=1, all resources ≤ 15% at 137 MHz) are irrelevant to software validation. You only need to check whether the method above clusters correctly and stably.

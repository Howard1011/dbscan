# my_window — HLS Window-Based Clustering Core (v35k)

A window-based clustering (DBSCAN-like) accelerator using `ap_fixed<8,4>` arithmetic, implemented in Vitis HLS 2023.2.

- Target platform: xcu250-figd2104-2L-e (Alveo U250), 10 ns clock
- Results: latency 308 cycles, II=1 on all loops, critical path 7.289 ns (about 137 MHz achievable)
- Resource and timing details: [RESOURCE_TIMING_SUMMARY.md](RESOURCE_TIMING_SUMMARY.md)
- Algorithm specification and stability analysis: [ALGORITHM.md](ALGORITHM.md)

## Files

| File | Description |
|---|---|
| `my_window.cpp` / `my_window.h` | HLS core (top function: `my_window`) |
| `my_window_tb.cpp` | Testbench: reads each event, compares against golden, writes labels |
| `run_hls.tcl` | One-shot csim + csynth script |
| `data/emb_event500~502.dat` | Input: 600 rows × 12 columns of embeddings per event |
| `data/golden_event500~502.dat` | Expected cluster labels (one integer per line) |
| `data/labels_event500~502.dat` | Reference output (overwritten when the testbench runs) |
| `data/pca_mean.dat` / `data/pca_components.dat` | PCA mean vector (12) and projection matrix (8×12). Used only by the Python reference; the same values are hard-coded in `my_window.cpp` |
| `clustering_window_emb_apfix.ipynb` | Software reference implementation |
| `ALGORITHM.md` | Algorithm specification and stability analysis |
| `RESOURCE_TIMING_SUMMARY.md` | Resource usage and timing breakdown |
| `my_window_proj/` | Vitis HLS project output (csim, synthesis reports, generated RTL) |

## Running

Load the Vitis HLS 2023.2 environment first (`source <Vitis_HLS install path>/settings64.sh`).

```bash
# From this directory: csim + synthesis
vitis_hls -f run_hls.tcl
```

For a quick functional check without launching Vitis, compile the C simulation with g++. Point `INC` at the Vitis HLS `include` directory:

pass criterion is `partition_match` (identical cluster partition).
`exact_match` will show FAIL because the hardware and the golden files
number the clusters differently; this is expected. The output should
end with `ALL EVENTS PASS (partition_match)`.

The testbench reads `data/emb_event*.dat` and `data/golden_event*.dat` relative to the current directory and writes `data/labels_event*.dat`, so run it from this directory.

Note: `run_hls.tcl` starts with `open_project -reset`, which deletes and regenerates `my_window_proj/`.

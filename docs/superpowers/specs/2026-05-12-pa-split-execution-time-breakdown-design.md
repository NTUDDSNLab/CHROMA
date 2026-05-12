# Execution-Time Breakdown Plot: CA / PA-scan / PA-decrement

**Date:** 2026-05-12
**Author:** PunchShadow + Claude
**Status:** Draft for review

## Goal

Produce a single figure that highlights the benefit of **dynamic workload
balancing** in CHROMA's PA decrement phase, by decomposing per-graph execution
time into three components (CA, PA scan, PA decrement) for three frameworks:

| Framework label              | Algo id | What's new vs the previous one        |
|------------------------------|---------|---------------------------------------|
| `cuSL_ELS_SDC` (baseline)    |   1     | Warp-per-vertex Phase 2               |
| `cuSL_ELS_SDC_CTA`           |  ≈3a    | BlockScan + CTA-balanced Phase 2      |
| `cuSL_ELS_SDC_CTA_S`         |  ≈3b    | Dispatched SDC-warp / CTA Phase 2     |

All three are exercised in **SPLIT mode** (separate `scan` / `decrement` /
`advance` kernel launches per outer iteration) so per-phase CUDA-event timing
isolates PA scan vs PA decrement. Theta is held fixed at **e=0** for all runs.

The figure spans **19 EGR datasets** in `Datasets/EGR/`, sorted by edge count
ascending on the x-axis.

## Non-goals

- Comparing CHROMA against JP-Series, CPU, or non-SDC variants.
- Tuning theta or comparing predicted-theta.
- Plotting wall-clock total time (the breakdown's three slices already sum to
  PA + CA wall time; reduction is excluded — the question is about PA cost).
- Showing absolute speedup numbers; the bar heights speak for themselves.

## Source-code changes (CHROMA module)

### `CHROMA/PA_split.cu`

Add two new Phase-2 split kernels. Phase 1 is reused from the existing
`P_SL_ELS_SDC_split_scan` because the scan + enqueue logic in
`P_SL_ELS_SDC`, `P_SL_ELS_SDC_CTA`, and `P_SL_ELS_SDC_CTA_S` is byte-for-byte
identical (verified by reading `PA.cu:103-280` and `PA.cu:528-720`).

1. `P_SL_ELS_SDC_CTA_split_decrement`
   - Mirrors `P_SL_ELS_SDC_CTA`'s Phase 2 only:
     - `cub::BlockScan` over `node_buf[BLOCK_SIZE]` for prefix sums
     - CTA-balanced distribution via `cursor_remove` atomicAdd
     - Per-work-unit `atomicSub(&degree_list[neighbor], 1)`
   - Removes the persistent `do { ... grid.sync(); } while (worker != nodes)`
     wrapper because the SPLIT loop is host-side.
2. `P_SL_ELS_SDC_CTA_S_split_decrement`
   - Mirrors `P_SL_ELS_SDC_CTA_S`'s Phase 2 only:
     - Reads `remove_size` once at entry
     - Dispatches per-block to SDC-warp path or CTA-balanced path depending on
       `remove_size < CTA_S_THRESHOLD`
     - Same `atomicSub` decrement
3. Extend `P_SL_ELS_SDC_split_advance` to also reset `cursor_remove = 0`.
   Harmless for the existing SDC SPLIT (which doesn't use the cursor);
   required for the two new variants.

### `CHROMA/chroma_utils.cuh` / `chroma_utils.cu`

Add a return struct + two new host launchers; refactor the existing one:

```cpp
struct PaSplitStats {
    float scan_ms;        // sum across all outer iterations
    float decrement_ms;   // sum across all outer iterations
};

PaSplitStats run_sdc_split    (int blocks, const ECLgraph& g, DevPtr& d);
PaSplitStats run_sdc_cta_split(int blocks, const ECLgraph& g, DevPtr& d);
PaSplitStats run_sdc_cta_s_split(int blocks, const ECLgraph& g, DevPtr& d);
```

Each launcher wraps every `_split_scan` / `_split_decrement` call with a
`cudaEvent` pair (start before launch, stop after `cudaEventSynchronize`)
and accumulates the elapsed ms into `scan_ms` / `decrement_ms`. The `_advance`
kernel is NOT timed separately — its cost is negligible (single-thread
housekeeping) and folds into wall-clock overhead.

### `CHROMA/CHROMA.cu`

1. Help banner: add lines for algo ids `8` (`cuSL_ELS_SDC_CTA_SPLIT`) and
   `9` (`cuSL_ELS_SDC_CTA_S_SPLIT`); update the SPLIT description for `7`.
2. Algo-string parsing: accept `8`/`cuSL_ELS_SDC_CTA_SPLIT` and
   `9`/`cuSL_ELS_SDC_CTA_S_SPLIT`.
3. Run-loop dispatch: when `algo_name == "cuSL_ELS_SDC_CTA_SPLIT"` call
   `run_sdc_cta_split(...)`; when `cuSL_ELS_SDC_CTA_S_SPLIT` call
   `run_sdc_cta_s_split(...)`. Both replace the existing `timer_PA` block for
   that algo (the existing GPUTimer no longer needs to wrap the kernel
   launches; we use the per-phase ms returned by the launcher).
4. For ALL three SPLIT variants (7, 8, 9):
   - Replace single `runtime_PA` with `runtime_PA = scan_ms + decrement_ms`.
   - Per-run line gets two extra fields:
     `PA scan: X.XXX ms  PA dec: Y.YYY ms`
   - Multi-run summary adds two lines after `PA time     : ...`:
     `PA scan     : avg=...  min=...  max=...`
     `PA decrement: avg=...  min=...  max=...`
   - The new fields are gated on `algo_name.endswith("_SPLIT")` so non-SPLIT
     runs are byte-for-byte unchanged in their output.

## `scripts/batch_profile.py`

New top-level batch driver. Modeled after `run_pa_sweep.py` but simpler.

```text
usage: batch_profile.py [-h]
                        [--binary BINARY]
                        [--dataset-dir DATASET_DIR]
                        [--runs RUNS]
                        [--timeout TIMEOUT]
                        [--frameworks FW [FW ...]]
                        [--only NAME [NAME ...]] [--skip NAME [NAME ...]]
                        [--out OUT]
```

Defaults:
- `--binary` → `CHROMA/CHROMA`
- `--dataset-dir` → `Datasets/EGR`
- `--runs` → `5`
- `--timeout` → `1200` s per (framework, dataset) cell
- `--frameworks` → `cuSL_ELS_SDC_SPLIT cuSL_ELS_SDC_CTA_SPLIT cuSL_ELS_SDC_CTA_S_SPLIT`
- `--out` → `scripts/batch_profile_results.json`

For each (framework, dataset):
1. Run `CHROMA -f <dataset> -a <framework> -e 0 --runs 5` (no `--predict`).
2. Parse stdout. Required regex matches:
   - `nodes = (\d+)` / `edges = (\d+)` (from the binary's pre-loop banner).
     Falls back to reading the `.egr` header via `lib/io/ECLgraph.h` shape if
     the binary doesn't print these — verified via existing print sites.
   - `=== Statistics over N runs (ms) ===` block:
     - `CA time     : avg=([0-9.]+)`
     - `PA scan     : avg=([0-9.]+)`
     - `PA decrement: avg=([0-9.]+)`
     - `Total time  : avg=([0-9.]+)`
     - `colors used : avg=([0-9.]+)`
3. On timeout or non-zero exit code, record `{"error": "..."}` for that cell.

Output JSON shape:

```json
{
  "frameworks": [...],
  "datasets":   [{"name": "facebook", "nodes": 4039, "edges": 88234}, ...],
  "rows": [
    {
      "framework": "cuSL_ELS_SDC_SPLIT",
      "dataset":   "facebook",
      "nodes":     4039,
      "edges":     88234,
      "ca_ms":           6.10,
      "pa_scan_ms":      1.20,
      "pa_decrement_ms": 0.80,
      "total_ms":        8.40,
      "colors_used":     72,
      "runs":            5
    },
    ...
  ]
}
```

Stderr emits per-cell progress lines like `run_pa_sweep.py` does.

## `scripts/plots/plot_execution_breakdown.py`

New plotting script. Reads `scripts/batch_profile_results.json` (path is
configurable via `--in`) and writes
`scripts/plots/execution_time_breakdown.{pdf,png}`.

Layout:
- One axes; no title.
- X axis: 19 dataset groups, sorted by `edges` ascending; labels rotated 35°.
- Per group: 3 grouped bars (`width=0.27` each, in-group gap 0).
- Per bar: 3-segment stack — bottom **CA** (color #1), middle **PA scan**
  (color #2), top **PA decrement** (color #3).
- Hatches for the 3 frameworks: `''`, `'//'`, `'xx'` (configurable).
- Edge color black, `linewidth=0.8`.
- Y axis: linear by default. `--log` switches to symlog (with a small linthresh
  to keep zeros visible).
- Two horizontal legends placed above the axes via `fig.legend(...)`:
  1. Top legend: 3 colored patches → `CA`, `PA scan`, `PA decrement`.
  2. Second legend: 3 hatched patches (neutral grey fill) → framework labels.

CLI:

```text
usage: plot_execution_breakdown.py [-h]
                                   [--in PATH]         (default: scripts/batch_profile_results.json)
                                   [--out-prefix PATH] (default: scripts/plots/execution_time_breakdown)
                                   [--log]
                                   [--figsize W H]     (default: 14 5)
```

## Unit-of-work isolation

| Unit                              | What it does                                | Depends on                          |
|-----------------------------------|---------------------------------------------|--------------------------------------|
| PA_split.cu kernels               | Per-phase device kernels                    | globals.cuh, cub                     |
| chroma_utils launchers            | Host-side scan→decrement→advance loop + timing | PA_split kernels                   |
| CHROMA.cu dispatch + printing     | Wire SPLIT algos and emit per-phase ms      | launchers                            |
| batch_profile.py                  | Sweep (framework, dataset) and write JSON   | the binary above                     |
| plot_execution_breakdown.py       | Render figure from JSON                     | JSON only — no binary, no datasets   |

Plotting is decoupled from the binary by the JSON contract, so the script can
be iterated without rerunning the sweep.

## Risks / failure modes

- **SPLIT mode is slow.** The non-cooperative host loop pays
  `cudaMemcpyFromSymbol(worker)` every outer iteration; large graphs may take
  10× longer than the cooperative version. The 1200 s timeout absorbs this for
  all 19 EGR graphs based on prior `pa_sweep_*.log` traces (worst case
  `twitter_combined` ≈ 0.45 s for `cuSL_ELS_SDC_CTA` non-split → expect
  ≤ 30 s split).
- **`europe_osm` is the largest by edges (~33M) and slowest historically.** If
  any cell does time out, batch_profile.py records the error and the plot
  script will draw an empty bar group for that dataset.
- **`atomicSub` semantics**: the existing `P_SL_ELS_SDC_split_decrement`
  passes `iteration_list` to read the `0x40000000u` "removed" bit. The new CTA
  variants must do the same — confirmed in PA.cu Phase 2 paths.

## Testing strategy

After the source patches:

1. Build with `make ARCH=sm_89 PROF=1`; verify a clean build.
2. Smoke run: `CHROMA -f Datasets/test/facebook.egr -a cuSL_ELS_SDC_CTA_SPLIT
   -e 0 --runs 3` — must print `result verification passed` and the new
   `PA scan / PA decrement` lines.
3. Sanity check: `pa_scan_ms + pa_decrement_ms ≈ pa_ms` from the same run of
   `cuSL_ELS_SDC_CTA` (non-split), to within ~10% (extra cost is launch
   overhead). If they diverge by an order of magnitude, the split kernels are
   producing wrong work and need re-inspection.
4. Run `batch_profile.py --only facebook le450_25d school1 --runs 2` end-to-end
   and confirm JSON is well-formed.
5. Run `plot_execution_breakdown.py --in scripts/batch_profile_results.json`
   and verify the figure opens (or saves) cleanly.

Final figure is regenerated only after the full 19×3 sweep completes.

## Out of scope

- BB split variant (algo 6) and JP-Series — not part of this comparison.
- Predicted-theta mode (`--predict`) — orthogonal to this ablation.
- Color-reduction time — excluded from the breakdown by design.

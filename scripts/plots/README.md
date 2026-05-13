# Execution-Time Breakdown Plot

Tooling for measuring and visualizing CHROMA's CA / PA execution time
across the EGR dataset suite for three SDC algorithm variants:

| CLI algo string         | What it measures                                  |
|-------------------------|---------------------------------------------------|
| `cuSL_ELS_SDC`          | Baseline SDC, warp-per-vertex decrement           |
| `cuSL_ELS_SDC_CTA`      | CTA-balanced decrement (BlockScan)                |
| `cuSL_ELS_SDC_CTA_S`    | Dispatched SDC-warp / CTA (the recommended)       |

By default the sweep uses these **unified (cooperative) kernels** and
the figure renders each bar as a 2-segment stack: CA (bottom) + PA
(top, scan and decrement fused inside the cooperative kernel).

For finer diagnostics, swap in the SPLIT-mode variants
(`cuSL_ELS_SDC_SPLIT`, `cuSL_ELS_SDC_CTA_SPLIT`,
`cuSL_ELS_SDC_CTA_S_SPLIT`) and the figure auto-renders a 3-segment
stack with `PA scan` and `PA decrement` separated. SPLIT mode runs
each phase as its own kernel launch + `cudaMemcpyFromSymbol(worker)`
roundtrip per outer iteration, so it is several × slower than the
cooperative kernels — use it when you need the breakdown, not for
production timing.

The workflow has two stages — drive the sweep with
`scripts/batch_exe_breakdown_profile.py` to collect timings, then render
the figure with `scripts/plots/plot_execution_breakdown.py`.

## Prerequisites

1. Build the CHROMA binary for your GPU. On the lab RTX A4000 (sm_86) just
   run `make` from `CHROMA/`:

   ```
   cd CHROMA && make -j4
   ```

   The default `ARCH=sm_86` is correct for that hardware. Override with
   `ARCH=sm_89` (or similar) only when targeting newer GPUs — a mismatched
   arch silently miscompiles the cooperative kernels.

2. Python 3 with `matplotlib` and `numpy`. No virtualenv needed; the
   project's existing toolchain is sufficient.

## Step 1 — Run the sweep (`scripts/batch_exe_breakdown_profile.py`)

For each `(framework, dataset)` cell the script runs:

```
CHROMA/CHROMA -f <dataset.egr> -a <framework> --no-reduce --runs N <theta-flag>
```

where `<theta-flag>` is either `-e <N>` (default) or `--predict`
(optionally `--predict --v2-model`). The script then parses the multi-
run statistics block to extract per-phase ms.

### Usage

```
python3 scripts/batch_exe_breakdown_profile.py [options]
```

| Flag                  | Default                                | Meaning |
|-----------------------|----------------------------------------|---------|
| `--binary`            | `CHROMA/CHROMA`                        | Path to the CHROMA binary. |
| `--dataset-dir`       | `Datasets/EGR`                         | Directory glob'd for `*.egr` files. |
| `--runs`              | `5`                                    | Repeated runs per cell; must be `>= 2`. |
| `--timeout`           | `1200`                                 | Seconds per cell before aborting that cell. |
| `--frameworks`        | `SDC  CTA  CTA_S`                      | One or more CHROMA algorithm names. Defaults to the unified cooperative kernels; pass the `_SPLIT` variants to capture the 3-segment breakdown. |
| `--only`              | (all `.egr` files)                     | Restrict to specific dataset stems. |
| `--skip`              | (none)                                 | Drop specific dataset stems. |
| `--out`               | `scripts/batch_profile_results.json`   | Path for the JSON output. |
| `-e`, `--elastic`     | `0`                                    | Set theta via CHROMA's `-e`. Mutually exclusive with `--predict`. |
| `-p`, `--predict`     | (off)                                  | Use CHROMA's `--predict` (linked ML model) instead of a fixed `-e`. |
| `--predict-model`     | `v3`                                   | When `--predict` is set, picks which CHROMA predictor to use. `v3` (9-feature random forest, CHROMA's default) or `v0_paper` (paper-era RF on V/E only). Forwarded as `--predict-model <name>`. Ignored when `--predict` is not set. |

### Examples

```
# Full default sweep (19 datasets x 3 frameworks x 5 runs, theta = 0).
python3 scripts/batch_exe_breakdown_profile.py

# Smoke test on two small graphs with fewer runs.
python3 scripts/batch_exe_breakdown_profile.py --only facebook le450_25d --runs 3

# Profile theta = 10 instead of the default 0; write to a separate JSON.
python3 scripts/batch_exe_breakdown_profile.py -e 10 \
    --out scripts/breakdown_e10.json

# Profile the v3 ML predictor (CHROMA's default --predict path).
python3 scripts/batch_exe_breakdown_profile.py --predict \
    --out scripts/breakdown_predict_v3.json

# Profile the paper-era predictor (--predict-model v0_paper).
python3 scripts/batch_exe_breakdown_profile.py --predict --predict-model v0_paper \
    --out scripts/breakdown_predict_v0.json

# Restrict to the two CTA variants (unified kernels).
python3 scripts/batch_exe_breakdown_profile.py \
    --frameworks cuSL_ELS_SDC_CTA cuSL_ELS_SDC_CTA_S

# Three-segment breakdown via SPLIT kernels (slower, diagnostic).
python3 scripts/batch_exe_breakdown_profile.py \
    --frameworks cuSL_ELS_SDC_SPLIT cuSL_ELS_SDC_CTA_SPLIT cuSL_ELS_SDC_CTA_S_SPLIT \
    --out scripts/breakdown_split.json

# Partial sweep, custom output path.
python3 scripts/batch_exe_breakdown_profile.py \
    --skip europe_osm soc-pokec-relationships.col \
    --out /tmp/quick.json
```

### Output JSON shape

```
{
  "config": {
    "elastic":       0,         // null when "predict" is true
    "predict":       false,
    "predict_model": null,      // "default" or "v2" when "predict" is true
    "runs":          5,
    "no_reduce":     true
  },
  "frameworks": ["cuSL_ELS_SDC_SPLIT", ...],
  "datasets":   [{"name": "facebook", "nodes": 4039, "edges": 176468}, ...],
  "rows": [
    {
      "framework":       "cuSL_ELS_SDC_SPLIT",
      "dataset":         "facebook",
      "nodes":           4039,
      "edges":           176468,
      "runs":            5,
      "wall_s":          0.4,
      "elastic":         0,
      "predict":         false,
      "predict_model":   null,
      "ca_ms":           3.50,
      "pa_scan_ms":     13.82,
      "pa_decrement_ms": 8.88,
      "pa_ms":          22.70,
      "total_ms":       26.20,
      "colors_used":    72.0
    },
    ...
  ]
}
```

Failed cells (timeouts, non-zero exits, unparseable stdout) get an
`"error"` field instead of the timing fields.

## Step 2 — Render the figure (`scripts/plots/plot_execution_breakdown.py`)

Reads the JSON produced by step 1 and emits a stacked grouped-bar PDF +
PNG.

By default the 19 datasets are split into 4 horizontal panels by max
total execution time so each panel has its own y-axis scale and small-
graph bars stay readable alongside the largest graphs. Within each panel,
datasets are sorted by edge count ascending. Each bar is a stack of CA
(bottom) / PA scan / PA decrement (top); each framework gets a distinct
hatch (`''`, `'//'`, `'xx'`). Two horizontal legends sit above the figure
(stack colours, framework hatches). No title.

### Usage

```
python3 scripts/plots/plot_execution_breakdown.py [options]
```

| Flag           | Default                                       | Meaning |
|----------------|-----------------------------------------------|---------|
| `--in`         | `scripts/batch_profile_results.json`          | JSON produced by `batch_profile.py`. |
| `--out-prefix` | `scripts/plots/execution_time_breakdown`      | Writes `<prefix>.pdf` and `<prefix>.png`. |
| `--bins`       | `20,100,300,1000`                             | Comma-separated ms thresholds (max total) for splitting datasets across panels. Pass `--bins ""` for a single panel. |
| `--log`        | (off)                                         | Use symlog y-axis on every panel. Usually unnecessary once panels are split by magnitude. |
| `--figsize`    | `16 5`                                        | Figure width and height in inches. |

### Examples

```
# Default 5-panel layout at 20 / 100 / 300 / 1000 ms (empty bins are dropped).
python3 scripts/plots/plot_execution_breakdown.py

# Single-panel (legacy) layout.
python3 scripts/plots/plot_execution_breakdown.py --bins ""

# 4-panel layout at the previous default boundaries.
python3 scripts/plots/plot_execution_breakdown.py --bins "100,300,1000"

# 3-panel layout, one wider boundary.
python3 scripts/plots/plot_execution_breakdown.py --bins "100,500"

# Custom size and output prefix (useful for paper figures).
python3 scripts/plots/plot_execution_breakdown.py \
    --figsize 18 4.5 \
    --out-prefix /tmp/breakdown
```

## Output artifacts

After running both steps the following land in `scripts/`:

- `scripts/batch_profile_results.json` — raw timings (gitignored under
  the project's `*.json` rule; regenerable from step 1).
- `scripts/batch_profile_full.log` — stderr of step 1, one line per cell
  (committed for reproducibility).
- `scripts/plots/execution_time_breakdown.pdf`
- `scripts/plots/execution_time_breakdown.png`

## Notes and gotchas

- **GPU arch matters.** On sm_86 hardware, building CHROMA with sm_89
  silently miscompiles the cooperative kernels — `cuSL_ELS` finishes in
  ~10 µs with no actual coloring and `cuSL_ELS_SDC_CTA_SPLIT` hangs.
  Use the Makefile default (`ARCH=sm_86`) unless you know your GPU is
  newer.
- **`--runs >= 2`.** CHROMA only prints the multi-run statistics block
  when `num_runs > 1`. The sweep enforces this with an early exit.
- **SPLIT-only.** Non-SPLIT algos don't emit the per-phase `PA scan` /
  `PA decrement` lines. The sweep rejects non-SPLIT names up front.
- **Color reduction is disabled.** The sweep passes `--no-reduce` to
  CHROMA so the breakdown measures pure CA + PA scan + PA decrement.
- **SPLIT mode is slow.** The per-iteration host-side
  `cudaMemcpyFromSymbol(worker)` imposes 5-15 µs overhead per outer
  iteration. On this dataset suite the slowest cell
  (`europe_osm` × `SDC_SPLIT`) takes ~20 s; the full sweep finishes in
  about 5 minutes wall time on the lab RTX A4000.

## Related source

- `CHROMA/PA_split.cu` — split-mode Phase 1 (scan), Phase 2 (decrement),
  and Phase 3 (advance) kernels.
- `CHROMA/chroma_utils.{cu,cuh}` — `run_sdc_split` / `run_sdc_cta_split`
  / `run_sdc_cta_s_split` host launchers and the `PaSplitStats` return
  struct.
- `CHROMA/CHROMA.cu` — algo IDs 7, 11, 12 dispatch and per-phase
  printing (`PA scan` / `PA decrement` lines in the multi-run summary).
- `docs/superpowers/specs/2026-05-12-pa-split-execution-time-breakdown-design.md`
  — design spec.
- `docs/superpowers/plans/2026-05-12-pa-split-execution-time-breakdown.md`
  — implementation plan.

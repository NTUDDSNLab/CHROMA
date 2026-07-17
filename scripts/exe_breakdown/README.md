# Execution-Time Breakdown Plot

Tooling for measuring and visualizing CHROMA's CA / PA execution time
across the EGR dataset suite for the paper's configuration table. One
sweep records every config into a single JSON; the plot script then
picks any subset with `--configs`.

| Config name       | `-a` algo            | EGC theta          | AWD | Bumping |
|-------------------|----------------------|--------------------|-----|---------|
| `CHROMA`          | `cuSL_ELS`           | `-e <N>`           |     |         |
| `CHROMA+`         | `cuSL_ELS_SDC`       | `-e <N>`           |     |         |
| `CHROMA_star`     | `cuSL_ELS_SDC`       | predict `v0_paper` |     |         |
| `CHROMA_star_awd` | `cuSL_ELS_SDC_CTA_S` | predict `v0_paper` | v   |         |
| `CHROMA_v2-b-awd` | `cuSL_ELS_SDC`       | predict `3feat`    |     |         |
| `CHROMA_v2-b`     | `cuSL_ELS_SDC_CTA_S` | predict `3feat`    | v   |         |
| `CHROMA_v2`       | `cuSL_ELS_SDC_CTA_S` | predict `3feat`    | v   | v       |

(`_star` stands in for the paper's superscript-* — shell-safe. All
static-theta configs pass `--no-dynamic-theta`; only `CHROMA_v2` leaves
the on-device bumping controller enabled.)

By default the sweep uses the **unified (cooperative) kernels** and
the figure renders each bar as a 2-segment stack: CA (bottom) + PA
(top, scan and decrement fused inside the cooperative kernel).

For finer diagnostics, pass the SPLIT-mode algo names as configs
(`cuSL_ELS_SDC_SPLIT`, `cuSL_ELS_SDC_CTA_SPLIT`,
`cuSL_ELS_SDC_CTA_S_SPLIT` — unknown config names pass through as raw
`-a` algos with the `-e` theta) and the figure auto-renders a 3-segment
stack with `PA scan` and `PA decrement` separated. SPLIT mode runs
each phase as its own kernel launch + `cudaMemcpyFromSymbol(worker)`
roundtrip per outer iteration, so it is several × slower than the
cooperative kernels — use it when you need the breakdown, not for
production timing.

The workflow has two stages — drive the sweep with
`scripts/exe_breakdown/batch_exe_breakdown_profile.py` to collect
timings, then render the figure with
`scripts/exe_breakdown/plot_execution_breakdown.py`.

## Prerequisites

1. Build the CHROMA binary for your GPU with the predictor linked in
   (the predict-based configs need it; `DYNAMIC_THETA=1` is the default
   and covers the `CHROMA_v2` bumping config). On the lab RTX A4000
   (sm_86):

   ```
   cd CHROMA && make PRE_MODEL=1 -j4
   ```

   The default `ARCH=sm_86` is correct for that hardware. Override with
   `ARCH=sm_89` (or similar) only when targeting newer GPUs — a mismatched
   arch silently miscompiles the cooperative kernels.

2. Python 3 with `matplotlib` and `numpy`. No virtualenv needed; the
   project's existing toolchain is sufficient.

## Step 1 — Run the sweep (`scripts/exe_breakdown/batch_exe_breakdown_profile.py`)

For each `(config, dataset)` cell the script runs:

```
CHROMA/CHROMA -f <dataset.egr> -a <algo> --no-reduce --runs N <config extras> <theta-flag>
```

where the algo, extras and theta flag come from the config's row in the
table above (`-e <N>` for the fixed-theta configs, `--predict
--predict-model <model>` for the rest). The script then parses the
multi-run statistics block to extract per-phase ms. All configs land in
one JSON.

### Usage

```
python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py [options]
```

| Flag              | Default                                            | Meaning |
|-------------------|----------------------------------------------------|---------|
| `--binary`        | `CHROMA/CHROMA`                                    | Path to the CHROMA binary. |
| `--dataset-dir`   | `Datasets/EGR`                                     | Directory glob'd for `*.egr` files. |
| `--runs`          | `5`                                                | Repeated runs per cell; must be `>= 2`. |
| `--timeout`       | `1200`                                             | Seconds per cell before aborting that cell. |
| `--configs`       | all 7 paper configs                                | Config names from the table above, and/or raw CHROMA `-a` algo names (pass the `_SPLIT` variants for the 3-segment breakdown). |
| `--only`          | (all `.egr` files)                                 | Restrict to specific dataset stems. |
| `--skip`          | (none)                                             | Drop specific dataset stems. |
| `--out`           | `scripts/exe_breakdown/batch_profile_results.json` | Path for the JSON output. |
| `-e`, `--elastic` | `0`                                                | Theta for the fixed-theta configs (`CHROMA`, `CHROMA+`, raw algo names). Predict-based configs ignore it. |

### Examples

```
# Full default sweep (19 datasets x 7 configs x 5 runs).
python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py

# Smoke test on two small graphs with fewer runs.
python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py \
    --only facebook le450_25d --runs 3

# Just the journal-version trio.
python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py \
    --configs CHROMA_v2-b-awd CHROMA_v2-b CHROMA_v2

# Fixed-theta configs at theta = 10 instead of 0.
python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py -e 10 \
    --configs CHROMA CHROMA+ --out scripts/exe_breakdown/breakdown_e10.json

# Three-segment breakdown via SPLIT kernels (slower, diagnostic).
python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py \
    --configs cuSL_ELS_SDC_SPLIT cuSL_ELS_SDC_CTA_SPLIT cuSL_ELS_SDC_CTA_S_SPLIT \
    --out scripts/exe_breakdown/breakdown_split.json

# Partial sweep, custom output path.
python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py \
    --skip europe_osm soc-pokec-relationships.col \
    --out /tmp/quick.json
```

### Output JSON shape

```
{
  "config": {
    "elastic":   0,      // the -e value used by fixed-theta configs
    "runs":      5,
    "no_reduce": true
  },
  "configs":  ["CHROMA", "CHROMA+", "CHROMA_star", ...],
  "datasets": [{"name": "facebook", "nodes": 4039, "edges": 176468}, ...],
  "rows": [
    {
      "config":          "CHROMA_v2",
      "algo":            "cuSL_ELS_SDC_CTA_S",
      "dataset":         "facebook",
      "nodes":           4039,
      "edges":           176468,
      "runs":            5,
      "wall_s":          0.4,
      "elastic":         null,     // 0 for fixed-theta configs
      "predict":         true,
      "predict_model":   "3feat",  // null for fixed-theta configs
      "ca_ms":           1.63,
      "pa_ms":           0.89,
      "total_ms":        2.52,
      "colors_used":    74.0
      // SPLIT-mode rows additionally carry pa_scan_ms / pa_decrement_ms
    },
    ...
  ]
}
```

Failed cells (timeouts, non-zero exits, unparseable stdout) get an
`"error"` field instead of the timing fields.

## Step 2 — Render the figure (`scripts/exe_breakdown/plot_execution_breakdown.py`)

Reads the JSON produced by step 1 and emits a stacked grouped-bar PDF +
PNG for any subset of the recorded configs.

By default the 19 datasets are split into 4 horizontal panels by max
total execution time so each panel has its own y-axis scale and small-
graph bars stay readable alongside the largest graphs. Within each panel,
datasets are sorted by edge count ascending. Each bar is a stack of CA
(bottom) / PA scan / PA decrement (top); each config gets a distinct
hatch. A combined horizontal legend sits above the figure (stack
colours, config hatches; wraps to two rows beyond 6 entries). No title.

### Usage

```
python3 scripts/exe_breakdown/plot_execution_breakdown.py [options]
```

| Flag           | Default                                            | Meaning |
|----------------|----------------------------------------------------|---------|
| `--in`         | `scripts/exe_breakdown/batch_profile_results.json` | JSON produced by step 1. |
| `--out-prefix` | `scripts/exe_breakdown/execution_time_breakdown`   | Writes `<prefix>.pdf` and `<prefix>.png`. |
| `--configs`    | (all configs in the JSON)                          | Which configs to plot, in legend/bar order. Errors out on names missing from the JSON. |
| `--bins`       | `20,100,300,1000`                                  | Comma-separated ms thresholds (max total) for splitting datasets across panels. Pass `--bins ""` for a single panel. |
| `--log`        | (off)                                              | Use symlog y-axis on every panel. Usually unnecessary once panels are split by magnitude. |
| `--figsize`    | `16 4`                                             | Figure width and height in inches. |

### Examples

```
# All recorded configs, default 5-panel layout (empty bins are dropped).
python3 scripts/exe_breakdown/plot_execution_breakdown.py

# Conference pair vs journal trio, from the same sweep JSON.
python3 scripts/exe_breakdown/plot_execution_breakdown.py \
    --configs CHROMA_star CHROMA_star_awd --out-prefix /tmp/conf
python3 scripts/exe_breakdown/plot_execution_breakdown.py \
    --configs CHROMA_v2-b-awd CHROMA_v2-b CHROMA_v2 --out-prefix /tmp/journal

# Single-panel (legacy) layout.
python3 scripts/exe_breakdown/plot_execution_breakdown.py --bins ""

# Custom size and output prefix (useful for paper figures).
python3 scripts/exe_breakdown/plot_execution_breakdown.py \
    --figsize 18 4.5 \
    --out-prefix /tmp/breakdown
```

## Output artifacts

After running both steps the following land in `scripts/exe_breakdown/`:

- `batch_profile_results.json` — raw timings (gitignored under the
  project's `*.json` rule; regenerable from step 1).
- `execution_time_breakdown.pdf` / `.png` — the figure.

## Notes and gotchas

- **GPU arch matters.** On sm_86 hardware, building CHROMA with sm_89
  silently miscompiles the cooperative kernels — `cuSL_ELS` finishes in
  ~10 µs with no actual coloring and `cuSL_ELS_SDC_CTA_SPLIT` hangs.
  Use the Makefile default (`ARCH=sm_86`) unless you know your GPU is
  newer.
- **`--runs >= 2`.** CHROMA only prints the multi-run statistics block
  when `num_runs > 1`. The sweep enforces this with an early exit.
- **SPLIT-only.** Non-SPLIT algos don't emit the per-phase `PA scan` /
  `PA decrement` lines; the plot falls back to the 2-segment CA + PA
  stack unless every selected config has the per-phase fields.
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

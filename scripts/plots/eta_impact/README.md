# η-Impact Plot

Sweeps η (`cta_s_threshold`, CLI `--eta`) — the `P_SL_ELS_SDC_CTA_S`
phase-2 dispatch threshold: `remove_size < η` takes the SDC
warp-per-vertex path, `remove_size >= η` the CTA-balanced path. Only
runtime varies with η; the coloring result is η-invariant.

Default η = 2048 (4 × block_size, block_size = 512).

## Prerequisites

- `CHROMA/CHROMA` built with `--eta` support:
  `cd CHROMA && make ARCH=sm_86` (set `ARCH` to your GPU).
- `.egr` graphs under `Datasets/EGR/` (override with `--dataset-dir`;
  default sweeps every `.egr` there, subset via `--datasets`).

## Step 1 — Sweep

```
python3 scripts/plots/eta_impact/eta_impact.py
```

For each dataset (default: every `.egr` in `--dataset-dir`) and each
η (default: ×2 from 128 to 64K, then ×4 out to 64M — past the largest
graph's node count, i.e. the pure-SDC limit) runs
`CHROMA -a cuSL_ELS_SDC_CTA_S -e 10 --eta <η>` 5×, keeps the best run
(min colors, tie → min runtime).
Key flags: `--datasets`, `--algo` (default `cuSL_ELS_SDC_CTA_S`),
`--elastic` (fixed θ, default 10), `--etas`, `--runs`, `--timeout`,
`--binary`, `--dataset-dir`, `--out`.
Writes `eta_impact_results.json` (regenerable). A failed η cell is
recorded with an `error` field and drawn as a 0-height gap.

## Step 2 — Plot

```
python3 scripts/plots/eta_impact/plot_eta_impact.py
```

Draws every dataset found in the results JSON; pick a subset with
`--datasets`. Flags: `--in`, `--out-prefix`, `--ncols` (subplots per
row, default 3), `--figsize` (default 5×3 per subplot). Writes
`eta_impact.{pdf,png}`. Per subplot: x = η; left y = runtime (ms)
line + markers (not zero-based, so the turnover stays visible);
right y = warp-centric (SDC-path) share in %, one line for
iterations and one for peeled nodes (CTA share = 100 − shown);
dashed vertical line = default η=2048, ★ = best (min-runtime) η.
Legend appears once, in the first subplot.

## Notes

- Smoke a single dataset fast:
  `python3 scripts/plots/eta_impact/eta_impact.py --datasets facebook
  --etas 512 2048 8192 --runs 2 --out /tmp/ei.json --dataset-dir Datasets/test`

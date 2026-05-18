# θ-Impact Plot (paper Fig. 6 redraw)

Redraws Fig. 6 of `CHROMA_IPDPSW_26.pdf` — `cuSL_ELS_SDC` runtime, color
count, and iteration count across θ — for **as-skitter**, **cit-Patents**,
**europe_osm**. The single paper "Predicted θ" star becomes two markers:

| Marker | Meaning | How it is obtained |
|--------|---------|--------------------|
| ★ CEP theta (v0_paper) | paper-era predictor θ | `CHROMA -a cuSL_ELS_SDC --predict --predict-model v0_paper` |
| ◆ AEP theta (v3_raw)   | v3 predictor θ, online bumping off | `CHROMA -a cuSL_ELS_SDC --predict --predict-model v3 --no-dynamic-theta` |

`EGC θ: N (Predicted)` reports the predictor's *initial* θ (before any
online bumping), so each predicted-θ is a single deterministic run.

## Prerequisites

- `CHROMA/CHROMA` built with `PRE_MODEL=1` (needed for `--predict`;
  supports `--predict-model {v3,v0_paper}` and `--no-dynamic-theta`):
  `cd CHROMA && make ARCH=sm_89 PRE_MODEL=1` (set `ARCH` to your GPU).
- `Datasets/EGR/{as-skitter,cit-Patents,europe_osm}.egr` present.

## Step 1 — Sweep

```
python3 scripts/plots/theta_impact/theta_impact.py
```

For each dataset: θ = 0…20 runs `CHROMA -a cuSL_ELS_SDC -e <θ>` 5×,
keeps the best run (min colors, tie → min runtime), then one
deterministic CEP and one AEP predicted-θ run. Key flags: `--datasets`,
`--algo` (default `cuSL_ELS_SDC`), `--theta-max` (default 20), `--runs`
(default 5), `--timeout` (per-invocation seconds, default 1200),
`--binary`, `--dataset-dir`, `--out`. Writes
`scripts/plots/theta_impact/theta_impact_results.json` (gitignored under
the project `*.json` rule; regenerable). A failed θ cell is recorded
with an `error` field and drawn as a 0-height gap; a failed predicted-θ
run stores `null` and that marker is omitted.

## Step 2 — Plot

```
python3 scripts/plots/theta_impact/plot_theta_impact.py
```

Flags: `--in`, `--out-prefix`, `--figsize` (default 15×4). Writes
`theta_impact.{pdf,png}`. Each subplot: x = θ, left y = runtime (ms)
bars coloured by #colors (per-subplot `color = N` legend), right y =
iteration-count line, ★ CEP / ◆ AEP near y=0.

## Notes

- `europe_osm` is the slow part of the sweep (θ=0 / small-θ runs on the
  largest graph dominate wall time); raise `--timeout` if cells fail.
  The per-cell timeout absorbs hangs — a timed-out cell becomes a gap.
- Predicted-θ is deterministic, so CEP/AEP are 1 run each (not `--runs`).
- Smoke a single dataset fast:
  `python3 scripts/plots/theta_impact/theta_impact.py --datasets
  cit-Patents --theta-max 3 --runs 2 --out /tmp/ti.json`.

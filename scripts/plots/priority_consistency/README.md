# Priority Consistency Ratio Plot

Redraws Fig. 5 of the CHROMA paper — vertex-priority-ordering
consistency of each framework with respect to the **JP-SL^A**
reference (paper Eq. 1–3: `C / T` concordant ordered pairs).

Reference: **JP-SL^A** via `CPU/Parallel/cpu_SL --dump`. The 8 bars
(the first is the reference compared against itself):

| Bar | Source |
|-----|--------|
| `JP-SL^A` (self) | `cpu_SL --dump` vs itself — the per-dataset consistency ceiling (no framework can exceed it; it's < 100% because JP-SL^A is a tie-heavy partial order) |
| `JP-SLL` | `pa_dumper -a JP_SLL` |
| `JP-ADG` | `cpu_ADG --dump` (pa_dumper's JP_ADG kernel emits no per-vertex order — all-zeros) |
| `CHROMA` | `pa_dumper -a cuSL_ELS -e 0` (ELS) |
| `CHROMA+` | `pa_dumper -a cuSL_ELS_SDC -e 0` (ELS+SDC) |
| `CHROMA*` | `CHROMA -a cuSL_ELS_SDC --no-reduce --predict --predict-model v0_paper --no-dynamic-theta` (paper-era predictor) |
| `CHROMA_v2 (v3_raw)` | `CHROMA -a cuSL_ELS_SDC --no-reduce --predict --predict-model v3 --no-dynamic-theta` (v3 offline predictor, no online bumping) |
| `CHROMA_v2 (v3_bump)` | `CHROMA -a cuSL_ELS_SDC --no-reduce --predict --predict-model v3` (v3 predictor + on-device dynamic-θ bumping) |

## Prerequisites

- `cd CPU/Parallel && make`  — builds `cpu_SL` and `cpu_ADG`, both with
  the `--dump <path>` flag (JP-SL^A reference and JP-ADG source).
- `tools/pa_dumper/pa_dumper` built (JP-SLL / CHROMA / CHROMA+).
- `CHROMA/CHROMA` built with `PRE_MODEL=1` (needed for `--predict`;
  supports `--predict-model {v3,v0_paper}` and `--no-dynamic-theta`).
- `scripts/consistency_metric` built
  (`g++ -O3 -std=c++17 -Ilib/io scripts/consistency_metric.cpp -o scripts/consistency_metric`).

## Step 1 — Sweep

```
python3 scripts/plots/priority_consistency/sweep_priority_consistency.py
```

Key flags: `--only/--skip <stems>`, `--threads N` (cpu_SL / cpu_ADG
OpenMP, default 32), `--frameworks ...`, `--timeout SECS` (default
1800), `--keep-dumps`, `--out PATH`, plus binary overrides
`--binary --pa-dumper --cpu-sl --cpu-adg --metric-bin`. Writes
`scripts/plots/priority_consistency/consistency_results.json`
(gitignored under the project `*.json` rule; regenerable).

## Step 2 — Plot

```
python3 scripts/plots/priority_consistency/plot_priority_consistency.py
```

Flags: `--in`, `--out-prefix`, `--ymin` (percent floor, default 10 —
low enough that no bar is clipped; global min ≈ 14.7%),
`--figsize` (default 14×3.2). Writes `priority_consistency.{pdf,png}`.

## Notes

- `cpu_SL` (JP-SL^A) and `cpu_ADG` (JP-ADG) priority allocation is the
  slow phase on large graphs (paper Fig. 1) — `europe_osm`,
  `as-skitter` can take minutes. The per-cell `--timeout` absorbs
  this; a failed cell is recorded with an `error` field and drawn as a
  0-height gap.
- `consistency_metric` dense-ranks each priority list independently,
  so only within-list ordering matters. JP-SL^A is a *partial order*
  (batch removal gives every vertex peeled in the same round the same
  priority), so a ref-vs-ref check yields `1 − tie-fraction` (≈ 0.99
  on facebook), not exactly 1.0; the dump is still deterministic and
  correctly ordered (validated during development).
- `pa_dumper`'s `JP_ADG` kernel does not write a per-vertex iteration
  (every vertex stays at 0 → all-zeros dump → consistency 0.0), which
  is why JP-ADG is sourced from `cpu_ADG --dump` instead, mirroring how
  JP-SL^A is sourced from `cpu_SL --dump`.

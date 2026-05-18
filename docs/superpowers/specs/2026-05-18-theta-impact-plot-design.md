# θ-Impact Plot (paper Fig. 6 redraw)

**Date:** 2026-05-18
**Author:** PunchShadow + Claude
**Status:** Draft for review

## Goal

Reproduce Fig. 6 of `CHROMA_IPDPSW_26.pdf` ("CHROMA* runtime, color
count, and iteration count across different values of θ") with one
change to the predicted-θ annotation: the original single "Predicted θ"
star becomes **two** markers —

- **★ CEP theta (v0_paper)** — the paper-era predictor's θ (the
  existing star, just relabelled).
- **◆ AEP theta (v3_raw)** — the v3 predictor's θ with online dynamic-θ
  bumping disabled, drawn with a distinct glyph/colour.

Self-contained sweep + plot pipeline under
`scripts/plots/theta_impact/`, structured like
`scripts/plots/priority_consistency/`.

## Figure layout (unchanged from the paper)

Three subplots in one row: **as-skitter**, **cit-Patents**,
**europe_osm**. Per subplot:

- x-axis: θ, integers 0–20.
- left y-axis: runtime (ms) — one bar per θ; **bar fill colour = number
  of colors used** at that θ (discrete per-subplot legend `color = N`,
  exactly as the paper).
- right y-axis: iteration count — line with markers across θ.
- **★** at `(cep_theta, ≈0)` labelled `CEP theta (v0_paper)`.
- **◆** at `(aep_theta, ≈0)` labelled `AEP theta (v3_raw)`, distinct
  colour.
- No title. Subplot captions `(a) as-skitter` / `(b) cit-Patents` /
  `(c) europe_osm`.

## Components

### 1. `scripts/plots/theta_impact/theta_impact.py` (sweep driver)

For each dataset in `as-skitter`, `cit-Patents`, `europe_osm`:

1. **θ sweep**, θ = 0…20: run
   `CHROMA -f <egr> -a cuSL_ELS_SDC -e <θ>` **5 times**; keep the best
   run (min `colors used`; tie → min runtime). Record
   `{color, runtime_ms, iter_count}`.
   - Parse the verbose single-run output with the same regexes
     `scripts/grid_elastic.py` uses:
     - runtime: `re.compile(r"Total\s+runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.I)`
       (anchored — CHROMA prints "PA runtime:"/"CA runtime:" before
       the "Total runtime:" line, so an unanchored "runtime:" would
       match PA time first)
     - colors: `re.compile(r"colors\s+used:\s*(\d+)", re.I)`
     - iter: `re.compile(r"Iter\s+count:\s*(\d+)", re.I)`
   - Reuse grid_elastic's keep-best rule:
     `sorted(valid, key=lambda r: (colors_used, runtime_ms))[0]`.
2. **CEP θ** (one run, deterministic):
   `CHROMA -f <egr> -a cuSL_ELS_SDC --predict --predict-model v0_paper`
   → parse the predicted θ from the line `EGC θ: <N> (Predicted)` with
   `re.compile(r"EGC[^:]*:\s*(\d+)\s*\(Predicted\)")`.
3. **AEP θ** (one run): `CHROMA -f <egr> -a cuSL_ELS_SDC --predict
   --predict-model v3 --no-dynamic-theta` → parse the same
   `EGC θ: <N> (Predicted)` line.

   Note: `EGC θ: N (Predicted)` reports the predictor's *initial* θ
   (printed before any online bumping), so the parsed value is
   independent of whether the dynamic-θ controller is on. CEP follows
   the paper's v0_paper predict path; AEP adds `--no-dynamic-theta`
   per the "v3_raw" definition. Marker values are unaffected by this
   asymmetry.

Output JSON (`--out`, default
`scripts/plots/theta_impact/theta_impact_results.json`):

```json
{
  "datasets": ["as-skitter", "cit-Patents", "europe_osm"],
  "algo": "cuSL_ELS_SDC",
  "theta_max": 20,
  "runs": 5,
  "data": {
    "as-skitter": {
      "nodes": 1696415, "edges": 22190596,
      "sweep": {
        "0":  {"color": 71, "runtime_ms": 451.9, "iter_count": 854},
        "1":  {"color": 70, "runtime_ms": 120.3, "iter_count": 612},
        "...": "...",
        "20": {"color": 73, "runtime_ms": 8.1,   "iter_count": 41}
      },
      "cep_theta": 2,
      "aep_theta": 4
    },
    "cit-Patents": { "..." : "..." },
    "europe_osm":  { "..." : "..." }
  }
}
```

A θ cell that fails/timing-outs records
`{"error": "<reason>"}` instead of the metric triple (drawn as a gap).
A failed CEP/AEP run sets `cep_theta`/`aep_theta` to `null` (marker
omitted for that subplot). Stderr prints one progress line per θ.

CLI flags (argparse, RawDescriptionHelpFormatter):

| Flag | Default | Meaning |
|------|---------|---------|
| `--binary`       | `CHROMA/CHROMA`        | CHROMA binary |
| `--dataset-dir`  | `Datasets/EGR`         | `.egr` location |
| `--datasets`     | `as-skitter cit-Patents europe_osm` | subplot datasets (stems) |
| `--algo`         | `cuSL_ELS_SDC`         | swept CHROMA algorithm |
| `--theta-max`    | `20`                   | sweep θ = 0…N |
| `--runs`         | `5`                    | runs per θ (keep-best) |
| `--timeout`      | `1200`                 | per-CHROMA-invocation seconds |
| `--out`          | `scripts/plots/theta_impact/theta_impact_results.json` | JSON path |

Dataset-stem → `.egr` resolution mirrors `run_pa_sweep.py` (`.col`
double-suffix aware): `cit-Patents` → `Datasets/EGR/cit-Patents.egr`,
`europe_osm` → `europe_osm.egr`, etc.

### 2. `scripts/plots/theta_impact/plot_theta_impact.py`

Reads the JSON, renders the 3-subplot figure described above:

- `fig, axes = plt.subplots(1, len(datasets), …)`.
- Per subplot, twin y-axis (`ax.twinx()`): bars on the left axis
  coloured by #colors via a discrete map (distinct colour per unique
  color count present in that subplot; per-subplot legend `color = N`);
  iteration-count line on the right axis.
- `ax.scatter([cep_theta],[~0], marker='*', s=…, label='CEP theta (v0_paper)')`
  and `marker='D'` for `AEP theta (v3_raw)` in a distinct colour;
  omit a marker if its θ is `null`.
- x ticks 0…20; left y "runtime (ms)", right y "iteration count" (only
  on the rightmost subplot's right axis label, matching the paper).
- One combined legend per subplot for the color-count entries + the two
  θ markers (the paper uses a per-subplot legend box).
- CLI: `--in` (default the JSON above), `--out-prefix`
  (default `scripts/plots/theta_impact/theta_impact`), `--figsize`
  (default 15 4). Writes `<prefix>.{pdf,png}`. Missing input → error,
  exit 1.

### 3. `scripts/plots/theta_impact/README.md`

Prerequisites (CHROMA built with `PRE_MODEL=1` for `--predict`;
supports `--predict-model {v3,v0_paper}` + `--no-dynamic-theta`),
the 2-step workflow, CLI tables, JSON shape, gotchas (europe_osm is the
slow part of the sweep; predicted-θ is deterministic so 1 run each;
figure/JSON gitignored on this branch — regenerable).

## Unit-of-work isolation

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| theta_impact.py  | θ-sweep + CEP/AEP θ → JSON | CHROMA binary |
| plot_theta_impact.py | JSON → 3-subplot figure | JSON only |
| README.md        | usage | — |

Plot is decoupled from the binary by the JSON contract.

## Testing strategy

1. CHROMA built with `PRE_MODEL=1` (else `--predict` fails — surfaced
   as a clear error). Confirm `CHROMA --help` lists `--predict-model`.
2. Sweep smoke: `theta_impact.py --datasets cit-Patents --theta-max 3
   --runs 2 --out /tmp/ti.json` → JSON has `sweep` keys "0".."3" each
   with color/runtime_ms/iter_count, plus integer `cep_theta`/`aep_theta`,
   no errors.
3. Predicted-θ parse check: assert `cep_theta` and `aep_theta` are
   ints ≥ 0 for cit-Patents (both models are linked in the
   `PRE_MODEL=1` build).
4. Plot smoke: synthetic JSON (3 datasets, θ 0–5, fabricated colors/
   iters/cep/aep) → `plot_theta_impact.py` writes non-empty PDF+PNG;
   missing-input → exit 1.
5. Full sweep (3 datasets × 21 θ × 5 + 6 predicted runs), then final
   figure; eyeball vs paper Fig. 6 (runtime drops sharply then
   flattens with θ; iteration-count line monotone-ish down; ★/◆ near
   small θ).

## Out of scope

- Datasets other than the paper's three (configurable via `--datasets`
  but not the default).
- Multi-GPU / RGP, color-reduction timing, dynamic-θ trajectory.
- Changing the consistency or breakdown pipelines.

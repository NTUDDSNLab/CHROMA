# Priority Consistency Ratio Plot (paper Fig. 5 redraw)

**Date:** 2026-05-17
**Author:** PunchShadow + Claude
**Status:** Draft for review

## Goal

Reproduce Fig. 5 of `CHROMA_IPDPSW_26.pdf` ("Ordering consistency of
approximate ordering heuristics and CHROMA with respect to JP-SL^A")
with an expanded framework set, as a self-contained sweep + plot
pipeline under `scripts/plots/priority_consistency/`.

The figure quantifies how closely each framework's vertex priority
ordering agrees with the JP-SL^A reference ordering, per the paper's
consistency-ratio metric (Eq. 1–3): `C / T` where `T = n(n-1)` ordered
pairs and `C` counts pairs whose relative order agrees in both lists.

## Frameworks

Baseline / reference (NOT drawn as a bar — it is the 100% anchor):

- **JP-SL^A** — produced by `CPU/Parallel/cpu_SL` (the OpenMP parallel
  Smallest-Last = Allwright batch-removal SL).

Comparison bars (in this left-to-right order within each dataset group):

| Bar label              | CHROMA config (paper Table III) | Priority source |
|------------------------|---------------------------------|-----------------|
| `JP-SLL`               | —                               | `pa_dumper -a JP_SLL` |
| `JP-ADG`               | —                               | `cpu_ADG <egr> <threads> --dump` (CPU/Parallel; pa_dumper's JP_ADG kernel emits no per-vertex order — all-zeros) |
| `ECL-GC`               | largest-degree-first ordering   | synthesized in Python |
| `CHROMA`               | ELS                             | `pa_dumper -a cuSL_ELS` (θ=0) |
| `CHROMA⁺`              | ELS + SDC                       | `pa_dumper -a cuSL_ELS_SDC` (θ=0) |
| `CHROMA*` (v0_paper)   | ELS + SDC + EGC (predicted)     | `CHROMA -a cuSL_ELS_SDC --predict --predict-model v0_paper --no-dynamic-theta` |
| `CHROMA_v2` (v3_raw)   | ELS + SDC + EGC (predicted, no online bumping) | `CHROMA -a cuSL_ELS_SDC --predict --predict-model v3 --no-dynamic-theta` |
| `CHROMA_v2` (v3_bump)  | ELS + SDC + EGC (predicted + online θ bumping) | `CHROMA -a cuSL_ELS_SDC --predict --predict-model v3` (dynamic-θ default ON) |

**v3_raw vs v3_bump:** the v3 offline predictor sets the initial θ; the
on-device dynamic-θ controller ("online bumping") is CHROMA's default in
`DYNAMIC_THETA=1` builds. `v3_raw` disables it with `--no-dynamic-theta`
(predictor's θ only); `v3_bump` keeps it on (predictor θ + online
booster). `CHROMA*` (v0_paper) is the paper-era predictor with no online
bumping, so it also takes `--no-dynamic-theta`. Internal keys:
`CHROMA*`, `CHROMA_v2_raw`, `CHROMA_v2_bump`.

## Datasets

The 19 EGR graphs in `Datasets/EGR/*.egr` (same set used by the
breakdown sweep and the paper). Display names reuse the breakdown
plot's `DATASET_LABELS` mapping (`wiki-Vote.col → wiki-Vote`,
`twitter_combined → twitter`, `soc-pokec-relationships.col → soc-Pokec`,
etc.).

## Non-goals

- No new consistency metric — reuse the built `scripts/consistency_metric`
  binary verbatim (`<graph.egr> <ref.bin> <test.bin>` → one JSON line).
- No multi-GPU / RGP, no θ sweeps. θ is fixed: 0 for CHROMA/CHROMA⁺;
  predictor-chosen for CHROMA*/CHROMA_v2_raw/CHROMA_v2_bump.
- No change to `pa_dumper`, the metric, or any GPU kernel.

## Components

### 1. `CPU/Parallel/SL.cpp` — add `--dump <path>`

`cpu_SL`'s `main()` currently accepts only `<graph.egr> <threads>` and
always runs the full coloring + verification after computing the
JP-SL^A `priority[]`. Add an optional 3rd positional/flag argument:

- Accept `<graph.egr> <threads> --dump <path>` (also tolerate
  `--dump=<path>`).
- When `--dump` is given: after `compute_SL(g, threads, priority)`
  returns, write `priority[]` as exactly `g.nodes` little-endian
  `uint32` values to `<path>` (reinterpret the `int` vector — the bit
  pattern is what `consistency_metric::read_priority` reads), print a
  one-line `dumped <n> priorities to <path>` confirmation, then
  `return 0` BEFORE the init/runLarge/runSmall coloring stage (skip the
  wasted CA work and its potential failure modes when only dumping).
- When `--dump` is absent: behaviour is byte-for-byte unchanged.
- Rebuild via `CPU/Parallel/Makefile` (`make` in `CPU/Parallel`).

### 1b. `CPU/Parallel/ADG.cpp` — add `--dump <path>` (same pattern)

`cpu_ADG` is structurally identical to `cpu_SL` (`compute_ADG(g,
threads, priority, 0)` builds `priority[v] =
(deg_v<<30)|(round_id<<16)|(deg&0xffff)`, then the same
`new[]`/init/runLarge/runSmall stage). pa_dumper's `JP_ADG` kernel
does not emit a per-vertex order (every vertex ends at iteration 0 →
all-zeros dump → consistency 0.0), so JP-ADG's priority list must come
from `cpu_ADG` instead — exactly mirroring how JP-SL^A comes from
`cpu_SL`. Apply the identical `--dump` change (arg-parse loop;
dump-and-exit block freeing the five `new[]` arrays before each
`return`; short-write removes the partial file; usage error returns
1) to `CPU/Parallel/ADG.cpp`, then rebuild.

Correctness note: `consistency_metric` densifies each list
independently (dense-ranks by value) before counting concordant
pairs, so only the *within-list ordering* matters, not absolute
values or cross-dumper encoding. `priority[v] =
(flag<<30)|((iter+1)<<16)|(deg&0xffff)` already sorts vertices into
JP-SL^A order by ascending value, identical in spirit to the
`iteration_list` that `pa_dumper`/`CHROMA --dump-priority` emit (which
the metric already consumes). Dumping it raw as `uint32` is therefore
correct. A ref-vs-ref smoke check validates this: JP-SL^A is a
**partial order** (batch removal gives every vertex removed in the same
round the same priority), so the metric's strict-concordance definition
makes a dump's self-consistency equal to `1 − tied_pair_fraction`
(≈ 0.993 on facebook, `tie_ratio_ref ≈ 0.057`), **not** 1.0. The real
gate is: two `cpu_SL` runs produce byte-identical dumps (determinism)
and self-comparison hits that maximum-achievable ratio with no
encoding inversion (a cross-check vs an all-distinct SL dump such as
`pa_dumper -a SDL` stays high, ≈ 0.97, not near 0).

### 2. `scripts/plots/priority_consistency/sweep_priority_consistency.py`

Sweep driver. For each dataset:

1. Dump the JP-SL^A reference: `cpu_SL <egr> <threads> --dump <ref.bin>`.
2. For each comparison framework, produce its `<test.bin>`:
   - `JP-SLL`, `CHROMA`, `CHROMA⁺` →
     `pa_dumper -f <egr> -a <algo> -e 0 --dump-priority <test.bin>`
     (`JP_SLL`, `cuSL_ELS`, `cuSL_ELS_SDC`).
   - `JP-ADG` → `cpu_ADG <egr> <threads> --dump <test.bin>` (CPU
     binary with the new `--dump` flag; pa_dumper's JP_ADG kernel
     can't produce a rankable order).
   - `CHROMA*` (v0_paper), `CHROMA_v2_raw` (v3, no bump),
     `CHROMA_v2_bump` (v3 + online bump) →
     `CHROMA -f <egr> -a cuSL_ELS_SDC --no-reduce --predict
     --predict-model {v0_paper,v3} [--no-dynamic-theta] --dump-priority
     <test.bin>`. `--no-dynamic-theta` is added for `CHROMA*` and
     `CHROMA_v2_raw`; omitted (controller default ON) for
     `CHROMA_v2_bump`.
   - `ECL-GC` → synthesized: read the `.egr` CSR (header gives
     `nodes`,`edges`; then the `nindex` array of `nodes+1` int32), set
     `degree[v] = nindex[v+1] - nindex[v]`, produce a total order by
     sorting vertex ids by `(-degree, id)`, and write
     `priority[v] = position_of_v_in_that_order` as `uint32`. Ascending
     priority therefore = "processed earlier" = largest-degree-first,
     matching the directional convention of the other dumps.
3. Run `scripts/consistency_metric <egr> <ref.bin> <test.bin>`, parse
   the single JSON line, collect `consistency_ratio` (plus
   `tie_ratio_test`, `distinct_test`, `concordant_unord_pairs` for
   completeness).
4. Clean up per-dataset temp dumps unless `--keep-dumps`.

CLI flags (argparse, RawDescriptionHelpFormatter):

| Flag | Default | Meaning |
|------|---------|---------|
| `--repo`         | auto (`parents[3]`) | repo root |
| `--binary`       | `CHROMA/CHROMA` | CHROMA binary (for the predict-model bars) |
| `--pa-dumper`    | `tools/pa_dumper/pa_dumper` | priority dumper |
| `--cpu-sl`       | `CPU/Parallel/cpu_SL` | JP-SL^A reference |
| `--cpu-adg`      | `CPU/Parallel/cpu_ADG` | JP-ADG priority source |
| `--metric-bin`   | `scripts/consistency_metric` | consistency metric |
| `--dataset-dir`  | `Datasets/EGR` | `*.egr` glob |
| `--threads`      | `32` | OpenMP threads for cpu_SL |
| `--frameworks`   | the 8 above | subset/override |
| `--only`/`--skip`| — | dataset-stem filter (`.col`-aware like run_pa_sweep) |
| `--timeout`      | `1800` | per (framework,dataset) cell seconds |
| `--keep-dumps`   | off | keep temp priority dumps |
| `--out`          | `scripts/plots/priority_consistency/consistency_results.json` | JSON output |

Output JSON:

```json
{
  "baseline": "JP-SL^A",
  "frameworks": ["JP-SLL","JP-ADG","ECL-GC","CHROMA","CHROMA+",
                 "CHROMA*","CHROMA_v2_raw","CHROMA_v2_bump"],
  "datasets": [{"name":"facebook","nodes":4039,"edges":176468}, ...],
  "rows": [
    {"framework":"JP-SLL","dataset":"facebook","nodes":4039,
     "edges":176468,"consistency_ratio":0.98,"tie_ratio_test":0.05,
     "wall_s":1.2}, ...
  ]
}
```

Failed cells get an `"error"` field instead of the metric fields
(same convention as the breakdown sweep). Stderr prints one progress
line per cell.

Internal framework keys are filesystem/CLI-safe (`CHROMA+`,
`CHROMA*`, `CHROMA_v2_raw`, `CHROMA_v2_bump`); the plot maps them to
display labels (e.g. `CHROMA_v2_bump` → `CHROMA_v2 (v3_bump)`).

### 3. `scripts/plots/priority_consistency/plot_priority_consistency.py`

Reads the JSON, renders paper-Fig.5 style:

- Single axes (no panels), `figsize` default `14 4`.
- One grouped cluster per dataset; one bar per framework
  (`bar_w = 0.85 / n_frameworks`), framework order as listed above.
- Distinct fill colour per framework (a fixed 7-colour palette);
  black edge, `linewidth=0.6`. No hatching (paper Fig.5 uses solid
  colours).
- Y-axis label `Consistency`, formatted as percent, limits
  `[50%, 100%]` (paper convention; `--ymin` overridable, `--ymax`
  default 100).
- X-axis: 19 dataset display names, rotated 35°, sorted alphabetically
  by display name (deterministic across servers, consistent with the
  breakdown plot's recent change).
- One horizontal `fig.legend` on top (framework names), `frameon=False`,
  no title.
- Writes `<out-prefix>.pdf` + `.png` (default prefix
  `scripts/plots/priority_consistency/priority_consistency`).
- CLI: `--in`, `--out-prefix`, `--ymin` (default 50), `--figsize`.

### 4. `scripts/plots/priority_consistency/README.md`

Usage guide mirroring `scripts/plots/README.md`: prerequisites
(build `cpu_SL`, `pa_dumper`, `CHROMA` with `PRE_MODEL=1`), the two
steps (sweep then plot), CLI tables, JSON shape, gotchas (cpu_SL
slowness on big graphs, predict-model needs PRED_MODEL build).

## Unit-of-work isolation

| Unit | Responsibility | Depends on |
|------|----------------|-----------|
| `SL.cpp --dump` | emit JP-SL^A priority as uint32 | ECLgraph.h |
| sweep script | orchestrate dumps + metric → JSON | cpu_SL, pa_dumper, CHROMA, consistency_metric |
| ECL-GC synth (in sweep) | LDF priority from .egr CSR | pure Python |
| plot script | JSON → figure | JSON only (no binaries) |

Plot is decoupled from data generation by the JSON contract.

## Testing strategy

1. Build: `cd CPU/Parallel && make` (cpu_SL); ensure `pa_dumper`,
   `CHROMA`, `scripts/consistency_metric` exist.
2. cpu_SL dump smoke: `cpu_SL Datasets/test/facebook.egr 32 --dump
   /tmp/ref.bin`; file size == `nodes*4` bytes; non-dump invocation
   still prints `result verification passed`.
3. **Ref-vs-ref check**: `consistency_metric facebook.egr ref.bin
   ref.bin` → `consistency_ratio == 1 − tie-fraction` (≈ 0.993 on
   facebook, NOT 1.0, because JP-SL^A is a partial order with tied
   priorities). Validate instead that (a) two cpu_SL runs are
   byte-identical, (b) self-comparison equals that
   maximum-achievable ratio, (c) a cross-check vs an all-distinct SL
   dump (`pa_dumper -a SDL`) stays high (≈ 0.97), confirming no
   encoding inversion.
4. Sweep smoke: `sweep_priority_consistency.py --only facebook
   le450_25d` → JSON well-formed, 2×7 rows, all `consistency_ratio`
   in (0,1], no errors.
5. ECL-GC sanity: its synthesized list on a known tiny graph orders
   highest-degree vertex first.
6. Plot smoke: render from the 2-dataset JSON; PDF+PNG non-empty.
7. Full 19×8 sweep, then final figure; eyeball against paper Fig.5
   (CHROMA⁺/CHROMA* should be high ≈ JP-SL^A; JP-ADG/ECL-GC lower on
   skewed graphs).

## Out of scope

- Adjacent-pair tie-ratio figure (collected in JSON but not plotted).
- Any change to `tools/pa_dumper` or `consistency_metric`.
- uk-2002 / non-EGR datasets.

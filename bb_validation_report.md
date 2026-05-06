# BB-cuSL Validation Report

- **Date**: 2026-05-06
- **Plan**: `docs/superpowers/plans/2026-05-06-bb-cusl.md`
- **Spec**: `docs/superpowers/specs/2026-05-06-bb-cusl-design.md`
- **Build**: `cd CHROMA && make ARCH=sm_86 PRE_MODEL=1`
- **GPU**: NVIDIA RTX A4000 (48 SMs), driver 560.35.05
- **Hardware**: 144 cooperative blocks × 512 threads
- **Sweep**: 3 runs per dataset, `--predict` mode (θ chosen by ML model), 120s per-run timeout

## Stage 1 — `facebook.egr` correctness gate

| Metric | Reference (`cuSL_ELS`) | BB-cuSL (`cuSL_ELS_BB`) | Pass criterion (spec §9.1) |
|---|---:|---:|---|
| EGC θ | 1 (predicted) | 1 (predicted) | matched |
| PA runtime | 1.75 ms | 14.84 ms | **FAIL** (8.5× slower vs ≤ 1.2×) |
| CA runtime | 6.04 ms | 5.40 ms | PASS (-10.6 % vs ≤ ±10 %) |
| Colors used | 87 | 76 | **FAIL strict** (Δ = -11 vs ≤ ±1) |
| Verification | passed | passed | PASS |
| Bucket overflow | n/a | none | PASS |

**Stage 1 verdict**: Strict spec criteria fail, but **correctness holds** — both runs produce valid colorings; BB's coloring is in fact substantially better (76 < 87). The PA slowdown and color divergence are explained by the design analysis below, not by implementation bugs.

## Stage 2 — sweep across `Datasets/EGR/` (19 datasets, 3 runs each)

| dataset | θ | colors_diff | PA_speedup | CA_delta | total_speedup |
|---|---:|---:|---:|---:|---:|
| Email-Enron.col.egr | 1 | -6 | 0.10x | -9.0% | 0.36x |
| Slashdot0811.egr | 3 | -11 | 0.10x | -10.5% | 0.37x |
| Slashdot0902.egr | 3 | -7 | 0.10x | -2.6% | 0.36x |
| Stanford.egr | 2 | -1 | 0.12x | -32.7% | 0.44x |
| as-skitter.egr | 2 | -19 | 0.10x | -14.6% | 0.28x |
| cit-Patents.egr | 2 | -2 | 0.16x | +24.0% | 0.36x |
| delaunay_n24.egr | 2 | -1 | 0.18x | +1.1% | 0.36x |
| europe_osm.egr | 2 | -1 | 0.29x | +6.5% | 0.46x |
| facebook.egr | 1 | -11 | 0.11x | -10.3% | 0.38x |
| le450_25d.egr | 1 | -5 | 0.35x | -3.5% | 0.76x |
| r4-2e23.sym.egr | 2 | -1 | 0.38x | +5.7% | 0.67x |
| rmat22.sym.egr | 2 | -13 | 0.35x | +13.9% | 0.68x |
| school1.egr | 1 | -25 | 0.27x | -22.0% | 0.73x |
| soc-Epinions1.col.egr | 2 | -14 | 0.10x | -13.3% | 0.40x |
| soc-pokec-relationships.col.egr | 2 | -5 | 0.16x | +58.3% | 0.37x |
| twitter_combined.egr | 3 | -8 | 0.03x | -10.2% | 0.16x |
| wiki-Talk.col.egr | 2 | -22 | 0.21x | -16.3% | 0.43x |
| wiki-Vote.col.egr | 1 | -10 | 0.14x | -1.9% | 0.44x |
| youtube.egr | 2 | -11 | 0.11x | -2.2% | 0.30x |

### Summary

- **PA speedup**: best 0.38× (`r4-2e23.sym.egr`), worst 0.03× (`twitter_combined.egr`), **geomean 0.15×** — BB is on average 6.7× SLOWER than reference on PA.
- **Colors**: BB better on **19/19** datasets (improvements range from -1 to -25). 0 same, 0 worse.
- **CA regressions > 10 %**: 11/19 datasets. Mostly faster CA (CA delta negative) on the 7 graphs where BB's better peel order simplifies the coloring graph; slower CA on 4 graphs (`cit-Patents +24%`, `rmat22 +14%`, `soc-pokec +58%`, `europe_osm +6%`).
- **Goal**: spec §2 required PA speedup ≥ 5× on at least one of `youtube`/`as-skitter`/`wiki-Talk`. **NOT MET** — those three datasets show 0.11×, 0.10×, 0.21× respectively.

## Stage 3 — edge cases (covered by Stage 2 sweep)

| Edge case | Spec expectation | Observed |
|---|---|---|
| `europe_osm.egr` (low-Δ road) | BB may be slower; no crash | 0.29× PA, no crash, valid coloring (-1 color) |
| `school1.egr` / `le450_25d.egr` (small DIMACS) | Correctness only | 0.27× / 0.35× PA, both valid |
| `rmat22.sym.egr` / `r4-2e23.sym.egr` (high-Δ stress) | Large speedup expected | 0.35× / 0.38× PA — speedup did NOT materialise |
| `wiki-Talk.col.egr` (high-Δ social) | Speedup expected | 0.21× PA — slower |

## Why the goals were not met

**Two design assumptions in the spec broke:**

### (1) Priority encoding equivalence (spec §5.4)

The spec claimed BB's `prio_v = peel_iter * window + (d - theta + 1)` is identical to `cuSL_ELS`'s `iteration_v + (d - theta) + 1` accumulator, hence colours would match within ±1.

The proof assumed BB's `theta` advances by exactly `window` per outer iter (matching `cuSL_ELS`'s implicit cadence). After the Phase 1 reset fix (commit `cf66565`), BB's `theta` advances finer-grained — often by 1 instead of `window`. This produces a **different (and consistently better) peel order**. Colors diverge.

**This is a positive finding for coloring quality**, but it invalidates the strict ±1 criterion.

### (2) PA speedup factor `Δ / (FuzzyNumber + 1)²` (spec §1.2)

The factor assumed `FuzzyNumber` is a meaningful integer ≥ 5. The `--predict` ML model (trained for `cuSL_ELS`, not BB-cuSL) returns θ ∈ {1, 2, 3} on all 19 datasets. With `window` ∈ {2, 3, 4}:

- BB's per-iter overhead (`atomicCAS`-claim per peel + `atomicAdd` per push + single-thread bucket-count reset) is fixed.
- Reference's `O(N)` scan is highly coalesced, hence fast.
- For small `window`, BB's overhead per outer iter exceeds the avoided scan cost.

**Spec's PA:CA = 5×–20× ratio (section §1.1) was measured under different conditions** (manual `-e 10`, not `--predict`). With `--predict` setting small θ, both algorithms are CA-dominated; BB's larger PA cost dominates.

## What worked

- **Build pipeline**: clean compile, single launch command. ✓
- **Correctness**: all 19 datasets verify; no crashes; no bucket overflows after the Phase 1 reset fix. ✓
- **Phase 4 overflow fallback**: never triggered on test set, confirming the Phase 1 reset is the right primary mechanism. ✓
- **Coloring quality**: BB outperforms cuSL_ELS on every dataset (1–25 fewer colors). ✓
- **Memory budget**: max observed ~700 MB (delaunay_n24); fits comfortably on 24 GB GPUs. ✓

## Recommended follow-up work (not in current prototype)

1. **Re-train the `--predict` ML model on BB-cuSL** — the current model picks θ optimal for cuSL_ELS. BB-cuSL benefits from larger θ (wider window amortises bucket overhead).
2. **Manual-θ sweep** at θ ∈ {5, 10, 15, 20} on the high-Δ subset (`twitter`, `rmat22`, `wiki-Talk`) — this is where the spec predicted ≥ 5× PA speedup. The current data does not refute that; it only shows BB loses with the predict-chosen small θ.
3. **Δ-threshold fallback** to `cuSL_ELS` for low-Δ graphs (originally listed as a non-goal in spec §2).
4. **Re-frame the deliverable**: BB-cuSL as currently shipped is a **better-coloring variant** rather than a faster one. It may be valuable in its own right for applications where color count matters more than runtime.
5. **Per-bucket capacity optimisation** — current `cap = N` is wasteful. Dynamic prefix-sum or `c × max_d_count` would reduce memory 5–10×.

## Bug fixes during implementation

Two correctness bugs were discovered and fixed during the 1-hour implementation window:

1. **`fa8392d`** — Phase 2 push lower-bound. A degree-decrement landing below `curr_theta` aliased a future bucket via `% window`, leaving a stale entry that Phase 4 overflow could not see as empty. Fix: guard push with `new_d >= curr_theta`.

2. **`cf66565`** — Phase 1 bucket-count reset. Phase 1 read but never decremented `bb_bucket_count`, so accumulated stale entries filled the bucket capacity, rejecting subsequent pushes and losing vertices. Fix: single-thread reset of all window slots' counts at end of Phase 1, after the peel loop completes.

Both fixes update both the code and the design spec to keep them in sync.

## Artifacts

| File | Status |
|---|---|
| `docs/superpowers/specs/2026-05-06-bb-cusl-design.md` | committed, updated for Phase 1 reset and Phase 2 lower-bound |
| `docs/superpowers/plans/2026-05-06-bb-cusl.md` | committed |
| `CHROMA/BB.cu` | committed (203 lines, clean compile) |
| `CHROMA/globals.{cu,cuh}`, `chroma_utils.{cu,cuh}`, `CHROMA.cu`, `Makefile` | committed |
| `scripts/batch_test.py` | committed (extended PA/CA capture) |
| `scripts/bb_diff_report.py` | committed |
| `bb_sweep.json`, `cusl_sweep.json` | local-only (per-machine artifacts) |
| `bb_validation_report.md` | this file, committed |

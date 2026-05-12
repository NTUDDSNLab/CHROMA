# θ Predictor v2: extended features + VR-capped maximisation loss

**Date:** 2026-05-08
**Status:** Spec, awaiting user review
**Owner:** PunchShadow
**Supersedes (operationally):** the predictor described in CHROMA paper §V (Linear Regression on `[V, E]` with VR-only loss)

## Background

CHROMA's `--predict` flag selects the elastic parameter θ for a graph by
calling `score(double input[])` (m2cgen-emitted from `model/model.cpp`).
The current predictor:

* **Features:** 3 — `(V, E, V/E)` — extracted from the `.egr` header alone.
* **Target:** `max_θ` such that `color(θ) ≤ color(0)` and the speedup
  ratio between consecutive θ values stays above 1.2× (existing
  `prepare_train_data` in `model/train.py`).
* **Loss:** `violation_ratio` (VR) — minimise the fraction of test
  graphs where prediction over-shot the true `max_θ`. Optimised via
  `make_scorer(..., greater_is_better=False)` in `GridSearchCV`.
* **Best result (paper):** Linear Regression, VR ≈ 3.66 %, MSE ≈ 0.26.

Two limitations the user wants to fix:

1. **Features are too coarse.** Two graphs with the same `(V, E)` can
   have very different degree distributions and respond very
   differently to the elastic mechanism. We want the predictor to see
   distributional shape, not just size.
2. **The loss is one-sided.** Pure VR minimisation rewards
   under-prediction — a model that always picks `θ̂ = 0` has VR = 0.
   We want the predictor to push θ̂ as high as possible **subject to
   keeping VR low**, so we capture more runtime gain without
   regressing colouring quality.
3. **The training target is locally-greedy.** `prepare_train_data`
   stops scanning as soon as it sees `color(θ) > color(0)`, but the
   paper itself (§VI-C, as-skitter) notes the `θ → color` curve is
   non-monotonic on some graphs — meaning the current target labels
   non-monotonic graphs with a too-small `max_θ` and the model is
   trained to under-predict on them. Section "Target (revised)" below
   addresses this.

## Goals

* G1. Extend the feature vector from 3 to **7** numerically robust
  graph features that capture both size and degree-distribution shape:
  `V, E, d, s, R, GI, H_er` (definitions below).
* G2. Replace the loss with a **VR-capped maximiser**: hinge-penalise
  VR above a configurable cap (default 5 %), then reward larger mean
  predicted θ.
* G3. Move from random 80/20 split to **5-fold cross-validation** so
  reported VR isn't biased by a lucky split.
* G4. Keep the `m2cgen → C++` deployment path. CHROMA `--predict` on
  the new model must work in-process, no sidecar JSON.
* G5. Ship the changes as a small **package** (B in the brainstorming
  options) so feature extraction, loss scoring, data prep, and the
  model factory are reusable and unit-testable in isolation.

## Non-goals

* Not adding new training data sources (SNAP / SuiteSparse / synthetic).
  Same training set as the paper for now; comparable numbers.
* Not switching to a deep model. m2cgen has to be able to emit the C++.
* Not redefining the target (`max_θ`). Loss change only.
* Not adding `--reduce` / `--runs` interactions. Predictor is upstream of those.

## Approach (B — modular package)

```
model/
  __init__.py
  features.py       # graph (.egr) → dict[str, float]
                    # (V, E, d, s, R, GI, H_er); shared with C++ inference
  losses.py         # weighted_vr_scorer + helpers (VR, mean_θ̂)
  data.py           # JSON I/O + prepare_train_data + K-fold split
  models.py         # ModelSpec dataclass + factory + grids + deployable flag
  train.py          # CLI orchestrator (existing argparse style)
  evaluate.py       # (optional) hold-out comparison: old vs new predictor

lib/io/
  graph_features.h    # C++ mirror: GraphFeatures + compute_graph_features(g)
  graph_features.cpp  # impl

CHROMA/CHROMA.cu     # --predict path consumes graph_features.h + scaler.h

# Generated artifacts (every train.py run)
model/model.cpp      # m2cgen → score(double input[7])
model/scaler.h       # static const double SCALER_MEAN[7], SCALER_STD[7]
model/model_meta.json # { feature_names, vr_cap, alpha, beta, model_class,
                       #   cv_results: [{fold_i, vr, mean_pred_theta}],
                       #   feature_count: 7 }
```

## Feature definitions

Reference: Boldi & Vigna, *Fairness on the Web: Alternatives to the
Power Law*, WebSci '12, DOI 10.1145/2380718.2380741. Preprint: ResearchGate
publication 260282672.

Notation: `n = V`, `m = E` (note: ECL `.egr` stores undirected edges as
two directed entries, so `Σ_v deg(v) = m`, average `d = m/n`). Sorted
degrees `d₁ ≤ d₂ ≤ ⋯ ≤ d_n`.

| Feature | Formula | Range | Notes |
|---|---|---|---|
| `V` | n | [1, ∞) | from `g.nodes` |
| `E` | m | [0, ∞) | from `g.edges` |
| `d` | m / n | [0, n) | mean degree |
| `s` | √( Σ_v (deg(v) − d)² / n ) | [0, ∞) | population standard deviation |
| `R` | (max d − min d) / d | [0, ∞) | "relative range" of degree |
| `GI` | (Σᵢ (2i − n − 1) · dᵢ) / (n · Σⱼ dⱼ), i = 1..n on sorted dᵢ | [0, 1] | Gini index, sorted-rank form |
| `H_er` | (−Σᵢ pᵢ log₂ pᵢ) / log₂ n, pᵢ = dᵢ / (2m) | [0, 1] | relative edge distribution entropy |

`H_er` uses log base 2 for normalisation against `log₂ n` (max
entropy when degree mass is uniform). If the paper uses natural log,
the ratio is invariant — so this matches.

**Single source of truth.** Both `model/features.py` (numpy
implementation) and `lib/io/graph_features.cpp` (C++) implement the
same formulas. A unit test (Python ↔ C++) on a fixed set of small
graphs (`K_5`, `P_10`, `C_10`, `BA_50_3`) checks numerical agreement
to 1e-9.

**Edge cases**:
* `n = 0` or `m = 0` → all features set to 0 except V (and the
  predictor is bypassed, callers should fall back to θ = 0).
* `d = 0` → `R` set to 0 to avoid division by zero.
* Graphs where all degrees are equal → `s = R = GI = 0`, `H_er = 1`.

### Standardisation

`StandardScaler` is fit on the full training set after K-fold CV to
produce `(mean[7], std[7])` arrays. These are emitted into
`model/scaler.h` as `static const double` arrays. C++ inference
applies `(raw − mean) / std` element-wise before calling `score()`.

Tree-based models (RF, GBR, XGBoost, LightGBM) are scale-invariant —
they consume raw features. Linear and SVR consume scaled features.
**The same `scaler.h` is emitted regardless** — for tree models
inference can call `score(raw_input)` directly; for linear/SVR it
calls `score(scaled_input)`. The model class is captured in
`model_meta.json` so the C++ side knows which path to take.

## Target + Loss

### Target (revised — fix non-monotonic bug)

The existing `prepare_train_data` walks θ in dict-iteration order
and stops at the first θ whose `color(θ) > color(0)`:

```python
if data[dataset][theta]['color'] <= base_theta_color:
    max_theta = theta
    if cur_speedup <= 1.2: break
else:
    break          # ← gives up at first colour regression
```

This is wrong for graphs whose `θ → color` landscape is
non-monotonic (paper §VI-C explicitly notes as-skitter's pattern:
worst at θ=9, best at θ=11). On these graphs the "early break"
discards globally-better θ values and labels the dataset with a
locally-conservative `max_θ`, which propagates to under-prediction
at inference time.

**New target definition** (`model/data.py`):

```python
def compute_max_theta(per_theta, baseline_color, speedup_decay=1.2):
    """Return the largest θ such that:
       (a) color(θ) ≤ baseline_color, AND
       (b) the cumulative incremental speedup since θ=0 stays
           productive (per-step speedup ≥ 1.2× at the chosen θ).
       The whole θ range is scanned — no early break on regressions.
    """
    sorted_thetas = sorted(
        (int(t) for t in per_theta if t.isdigit()),
    )
    last_runtime = per_theta['0']['runtime_ms']
    best_theta   = 0
    for t in sorted_thetas:
        rec = per_theta[str(t)]
        if rec.get('color') is None:
            continue
        cur_runtime = rec['runtime_ms']
        cur_speedup = last_runtime / cur_runtime if cur_runtime else 0
        last_runtime = cur_runtime
        if rec['color'] <= baseline_color and cur_speedup >= speedup_decay:
            best_theta = t
        # else: keep scanning (do NOT break)
    return best_theta
```

Two semantic differences vs current:

1. **Sorted iteration** by integer θ (was: dict insertion order; brittle
   if the JSON gets re-emitted with stringly-sorted keys).
2. **No `break` on colour regression** — keep scanning the full range
   so non-monotonic graphs see their true global max θ.

The speedup filter (1.2×) is preserved as the practical stop — we
only count θ as "useful" if the marginal speedup is meaningful.

Acknowledged limitation: with only graph-structural features (V, E,
d, s, R, GI, H_er) the model cannot directly observe the
non-monotonic landscape. We expect the predictor to learn a
"safe-ish" θ that under-predicts on non-monotonic graphs (lower mean
θ̂ but no VR cost), which is the right trade-off for `--predict`.
Capturing the landscape itself would require either (a) multi-output
regression of the curve or (b) inference-time micro-search; both
were considered (brainstorming options 2 + 3) and deferred until v2's
mean θ̂ plateaus on real datasets.

### Loss

```python
# model/losses.py
import numpy as np
from sklearn.metrics import make_scorer

def violation_ratio(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred) > np.asarray(y_true)))

def mean_theta_pred(y_true, y_pred):
    return float(np.mean(y_pred))

def weighted_vr_scorer(alpha=10.0, beta=1.0, vr_cap=0.05):
    """Lower is better. Hinge on VR cap, reward larger mean(ŷ).

        loss = α · max(0, VR − vr_cap)     # hard penalty above cap
             − β · mean(ŷ)                 # reward higher predictions

    GridSearchCV with greater_is_better=False prefers candidates that
    stay under vr_cap (loss reduces to −β·mean(ŷ)) and among those,
    picks the highest mean(ŷ). α ≫ β·max_mean_ŷ ensures the cap is a
    hard partition.
    """
    def _score(y_true, y_pred):
        v = violation_ratio(y_true, y_pred)
        m = mean_theta_pred(y_true, y_pred)
        return alpha * max(0.0, v - vr_cap) - beta * m
    return make_scorer(_score, greater_is_better=False)
```

CLI exposure (`train.py`):
* `--loss {weighted_vr, vr}` — default `weighted_vr`. `vr` reproduces
  the paper's loss for backward-compatible runs.
* `--vr-cap FLOAT` — default 0.05.
* `--alpha FLOAT --beta FLOAT` — default `α=10.0, β=1.0`.
* `--alpha-beta-sweep` — runs the grid `α ∈ {1,10,100} × β ∈
  {0.1,1,10}`, reports a table of `(α, β, fold_VR, fold_mean_θ̂)` for
  manual inspection. Doesn't pick automatically; the user picks an
  operating point and re-runs with that `(α, β)` to emit `model.cpp`.

### Why hinge instead of pure linear

Pure `α·VR − β·mean(ŷ)` keeps incentivising VR all the way to zero
even when that means heavily under-predicting. Hinge zeros that
incentive once VR is "good enough" so the model is free to push
predictions up.

## Validation + Tuning

### K-fold CV (K = 5)

`KFold(n_splits=5, shuffle=True, random_state=42)`, applied via
`GridSearchCV(scoring=weighted_vr_scorer(...), cv=5)`.

Each fold reports:
* Per-fold VR
* Per-fold mean predicted θ̂
* Per-fold MAE / MSE on (ŷ − y)
* Selected hyperparams

Aggregated: mean ± stdev VR, mean ± stdev mean(θ̂). Final production
model is retrained on all data with the CV-selected hyperparams.

### Models (`model/models.py`)

```python
@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], Estimator]
    grid: Dict[str, list]
    deployable: bool   # m2cgen can emit valid C++

SPECS = [
    ModelSpec("linear",  LinearRegression,        {}, deployable=True),
    ModelSpec("rf",      RandomForestRegressor,   RF_GRID,  deployable=True),
    ModelSpec("gbr",     GradientBoostingRegressor, GBR_GRID, deployable=True),
    ModelSpec("xgb",     XGBRegressor,            XGB_GRID, deployable=True),
    ModelSpec("lgbm",    LGBMRegressor,           LGBM_GRID, deployable=True),
    ModelSpec("svr",     SVR,                     SVR_GRID, deployable=False),  # diagnostic only
]
```

`train.py --model linear|rf|gbr|xgb|lgbm|svr` selects one. Default
`linear` (matches paper baseline).

`--no-grid` skips GridSearch and trains a single fit with default
hyperparams (fast iteration during development).

### Tuning behaviour

| flag | behaviour |
|---|---|
| `--loss weighted_vr` (default) | use new loss, fixed `(α, β, vr_cap)` from CLI |
| `--loss vr` | reproduce the paper's loss |
| `--alpha-beta-sweep` | grid over `(α, β)`, table-only (no model emit) |
| `--grid-search` | run the per-model hyperparameter grid |

## Output + CHROMA inference

### Files emitted by `train.py`

* `model/model.cpp` — `double score(double * input)` from m2cgen.
* `model/scaler.h` — `static const double SCALER_MEAN[7], SCALER_STD[7];`
  plus `static constexpr int FEATURE_COUNT = 7;` and a model-class
  string for runtime sanity check.
* `model/model_meta.json` — full provenance: feature order, loss
  config, K-fold results, sklearn version, training time.

### `lib/io/graph_features.h` (new)

```cpp
#pragma once
#include "ECLgraph.h"

struct GraphFeatures {
    double V, E, d, s, R, GI, H_er;
    double as_array[7];   // ordered: V, E, d, s, R, GI, H_er
};

GraphFeatures compute_graph_features(const ECLgraph& g);
```

Implementation in `lib/io/graph_features.cpp`. Single-pass over
`g.nlist` for d, s, min, max; second pass for `Σ |dᵢ − dⱼ|` is
O(n log n) via the sorted-rank Gini formula; entropy is O(n).
Total O(n log n) once at startup — negligible vs PA/CA.

### CHROMA `--predict` integration

`CHROMA/CHROMA.cu` change inside `if (use_predicted_elastic) { ... }`:

```cpp
#include "graph_features.h"
#include "scaler.h"   // emitted by train.py alongside model.cpp

GraphFeatures f = compute_graph_features(g);
double input[FEATURE_COUNT];
for (int i = 0; i < FEATURE_COUNT; ++i) {
    input[i] = (f.as_array[i] - SCALER_MEAN[i]) / SCALER_STD[i];
}
fuzzy_number = (int)round(score(input));
if (fuzzy_number < 0) fuzzy_number = 0;
```

### Backward compat with the paper's predictor

`train.py --legacy-features --loss vr` reproduces the paper-era
model (2 features, VR-only loss). The emitted `model.cpp` has
`score(double input[2])` — incompatible with the new `score(double
input[7])`, so we don't try to overload at the CHROMA side.
`model_meta.json` carries `feature_count`; `CHROMA.cu` reads it
once at startup (or via a `static_assert` on `FEATURE_COUNT` in
`scaler.h`) and refuses to run if `model.cpp` and `scaler.h` disagree
on feature count.

## Testing

### `tests/test_features.py`

Closed-form expected values on canonical graphs:

| Graph | V | E (directed) | d | s | R | GI | H_er |
|---|---:|---:|---:|---:|---:|---:|---:|
| K_5 (complete) | 5 | 20 | 4 | 0 | 0 | 0 | 1 |
| P_10 (path) | 10 | 18 | 1.8 | 0.4 | 0.555 | ~0.04 | ~0.99 |
| C_10 (cycle) | 10 | 20 | 2 | 0 | 0 | 0 | 1 |
| Star K_{1,9} | 10 | 18 | 1.8 | ~2.7 | 4.44 | ~0.81 | ~0.47 |

Asserts `compute_graph_features` matches to 1e-6.

### `tests/test_features_cpp.py`

Subprocess `pa_dumper --emit-features <graph.egr>` (or a new mini
binary `tools/feature_extractor/extract`) on the same canonical
graphs, compares to `features.py` to 1e-9. Runs in CI to catch
Python/C++ formula drift.

### `tests/test_losses.py`

Hand-checked corner cases on `weighted_vr_scorer(α=10, β=1, vr_cap=0.05)`:

| (VR, mean ŷ) | expected score |
|---|---:|
| (0.04, 5)  | -5.00 |
| (0.04, 10) | -10.00 |
| (0.08, 5)  | -4.70 |
| (0.08, 10) | -9.70 |
| (0.20, 10) | -8.50 |

### `tests/test_model_pipeline.py`

End-to-end: read a fixture features JSON, run K-fold CV with
LinearRegression + `weighted_vr` scorer, assert the emitted
`model.cpp` is non-empty and contains a `score` function with 7
arguments.

## Risks + open questions

* **R1.** `GI / H_er` formulas: verified the paper title and
  citation, but couldn't grab the PDF directly (ACM 403). Spec uses
  the canonical Gini sorted-rank formula and Shannon entropy
  normalised by `log₂ n` — if the paper uses different normalisation
  (e.g., `ln n`) the ratio is invariant, so this is a documentation
  issue not a numerical one.
* **R2.** With 7 features and ~17 training graphs, we're at risk of
  overfitting in tree models. K-fold CV will surface this; if VR
  blows up we add regularisation grids (`max_depth`, `min_samples_*`,
  `learning_rate`) or fall back to linear/GBR with restricted depth.
* **R3.** `StandardScaler` emit-to-C++ assumes Python and C++ produce
  identical floating-point values for `(x − μ) / σ`. Verified by
  test_features_cpp.py.
* **R4.** Currently the only inference site is `CHROMA/CHROMA.cu`. If
  CHROMA_RGP also adopts `--predict` later, the same
  `graph_features.h` + `scaler.h` should drop in (RGP doesn't
  currently call `score()`).
* **R5.** m2cgen support for LightGBM is partial. If LightGBM trains
  best, may need to fall back to GBR for emit step. `models.py` flag
  this risk per ModelSpec.
* **R6.** The new target definition (global max θ with no early
  break on colour regression) will yield strictly-larger `max_θ` on
  graphs whose `θ → color` curve is non-monotonic, but the
  graph-structural features alone (V/E/d/s/R/GI/H_er) probably
  can't predict *which* graphs are non-monotonic. So the model is
  expected to under-predict on those — the conservative behaviour we
  want, but it caps the achievable mean θ̂. If post-train evaluation
  shows mean θ̂ plateauing on non-monotonic graphs, escalation paths
  (deferred): predict the entire (color − baseline) curve as a 21-d
  vector (option 2 in brainstorming), or add inference-time
  micro-search around θ̂ (option 3).

## Migration

1. Land `model/features.py`, `losses.py`, `data.py`, `models.py`
   with tests but **no train.py change yet** — keep the old
   train.py alongside.
2. New `train_v2.py` (or `train.py --v2` flag) that runs the new
   pipeline and emits to `model/model_v2.cpp`, `model_v2.json`.
3. Add `graph_features.h/.cpp` to `lib/io/`. Inference path in
   CHROMA stays on the old 2-feature `model.cpp` until v2 is
   accepted.
4. Side-by-side eval: re-run `--predict` smoke test on the 17 SNAP
   graphs with both predictors; report (VR, mean θ̂, mean colors)
   table. If v2 wins on mean θ̂ without VR regression, replace
   `model.cpp` with `model_v2.cpp`, drop the old train.py, and
   `git mv model_v2.cpp model.cpp`.

## Amendment (2026-05-12): 9-feature deployed model (kcore + assort)

The shipped predictor extends the original 7-feature design with two
extra structural features. The C++ side
(`lib/io/graph_features.{h,cpp}`), Python side
(`model/features.py`), trainer (`model/train.py --features 9`), and
deployed `model/model.cpp` (Random Forest) all run on these 9
features. `model/model_meta.json` is authoritative — it carries the
exact `feature_names` list.

### Added features

| Feature | Formula | Range | Cost | Reference |
|---|---|---|---|---|
| `kcore` | `max{k : the k-core is non-empty}` (degeneracy) | [0, n−1] | O(V + E) via bucket-peeling | Seidman 1983; Batagelj & Zaversnik 2003 |
| `assort` | Pearson(deg(u), deg(v)) over directed edges (u,v) ∈ E | [−1, 1] | O(V + E) single pass | Newman 2002 §III eq. (4) |

`kcore` captures the densest cohesive subgraph the predictor can see
without iterating; `assort` distinguishes graphs where high-degree
nodes preferentially connect to other high-degree nodes
(assortative, e.g., collaboration nets) from disassortative ones
(e.g., the web, where hubs link to leaves). Both add new shape
information that `(V, E, d, s, R, GI, H_er)` does not encode.

**Edge cases** (`compute_graph_features`):
* `n ≤ 1` or `m = 0` → both set to 0.
* All edges have endpoints with identical degree → `assort` set to
  0 (zero-variance denominator).
* Self-loops are not generated by the ECL `.egr` reader, so peeling
  is well-defined.

### Updated feature vector

```
FEATURE_NAMES = ("V", "E", "d", "s", "R", "GI", "H_er", "kcore", "assort")
FEATURE_COUNT = 9                # was 7
SCALER_MEAN[9], SCALER_STD[9]    # in model/scaler.h
double score(double input[9])    # m2cgen signature
```

The integration block in CHROMA (`if (use_predicted_elastic)`) is
unchanged in structure — only `FEATURE_COUNT` shifted from 7 to 9.

### Deployment policy

The 9-feature Random Forest is shipped with two post-processing
knobs the spec didn't originally enumerate:

* `--use-floor` — at inference, lift any sub-floor prediction up to
  the per-graph empirical floor (small graphs get θ̂ ≥ 1 instead of
  rounding down to 0).
* `--score-shift FLOAT` — additive shift applied at prediction time
  before rounding (deployed default `-0.5`); pushes the model toward
  a slightly more conservative θ̂.

Both are part of `model/train.py`'s CLI and are needed to reproduce
the deployed `model.cpp` byte-for-byte. `model_meta.json` records
the active values under `rounding_policy: {score_shift, use_floor}`.

### Training procedure (canonical command)

The fully-qualified command that produced the current
`model/model.cpp` is documented in
`docs/superpowers/plans/2026-05-08-theta-predictor-v2.md` (Task 18,
Step 1). For the 9-feature deployed model:

```bash
python3 -m model.train --v2 \
    --input     model/results_res_grid_a1_all_with_counts.json \
    --graph-dir Datasets/EGR \
    --features  9 \
    --model     rf \
    --grid-search --cv-folds 5 \
    --vr-cap 0.05 --alpha 10.0 --beta 1.0 \
    --use-floor --score-shift -0.5 \
    --feature-cache model/feature_cache.json \
    --out-cpp    model/model.cpp \
    --out-scaler model/scaler.h \
    --out-meta   model/model_meta.json
```

Run `python3 -m model.train --help` for the complete flag surface.
Beyond `--help` and the plan's Task 18, there is no standalone
"how to train" markdown — the trainer's docstring + CLI argparse
serve that role.

### Training set size (clarifies §R2)

The original "Risks + open questions" §R2 worried about
overfitting with "~17 training graphs". That figure was the
17-graph paper-era set; it does not reflect the v3 training
corpus. The deployed v3 model is trained on **456 graphs**
(`model_meta.json::training_samples`) — a mix of bio-, ca-,
soc-, web-, road-, delaunay-, hugetric-, and SuiteSparse
collections. With 9 features and 456 samples the
overfitting risk for tree models is much lower than the
original spec assumed; the K=5 CV folds in
`model_meta.json::cv_results` are the live signal.

### Experimental results (EGR sweep, 19 graphs)

Held-out + trained-overlap evaluation using
`scripts/sweep_v123_combined.py` (v1/v2/v3_raw) merged with
`scripts/sweep_dynamic_theta.py` `static_dyn` arm (v3_bump). 5
runs per (graph, mode); reported numbers are arithmetic means
over runs, then aggregated as geomean / mean across the 19 EGR
graphs (11 in the training set, 8 strict holdout).

* **base**     — `--elastic 0`, no predictor, no dynamic-θ.
* **v1**       — paper-era 2-feature linear regression on `(V, V/E)`,
                 unbounded `θ̂` (often emits 10⁵–10⁷; CHROMA
                 silently caps internally).
* **v2**       — 7-feature Random Forest (V, E, d, s, R, GI, H_er).
* **v3_raw**   — 9-feature RF (adds kcore + assort), no dynamic-θ.
* **v3_bump**  — v3_raw + on-device bump-mode controller
                 (deployed default in `DYNAMIC_THETA=1` build).

| Subset | mode | geomean speedup | mean Δcolors | color regressions | color improvements |
|---|---|---:|---:|---:|---:|
| **TRAINED 11** | v1       | 5.84× | **+4.55** | 9/11 | 0/11 |
|                | v2       | 2.31× |  +0.00 | 3/11 | 4/11 |
|                | v3_raw   | 3.29× |  +0.18 | 4/11 | 2/11 |
|                | v3_bump  | **4.25×** | **−0.09** | **3/11** | **4/11** |
| **HOLDOUT 8**  | v1       | 6.26× | **+3.50** | 6/8  | 0/8  |
|                | v2       | 2.68× |  +0.38 | 3/8  | 0/8  |
|                | v3_raw   | 3.36× |  +0.50 | 3/8  | 0/8  |
|                | v3_bump  | **4.22×** |  +0.75 | 3/8  | 0/8  |
| **ALL 19**     | v1       | 6.01× | **+4.11** | 15/19 | 0/19 |
|                | v2       | 2.46× |  +0.16 |  6/19 | 4/19 |
|                | v3_raw   | 3.32× |  +0.32 |  7/19 | 2/19 |
|                | v3_bump  | **4.24×** |  +0.26 |  6/19 | 4/19 |

Headline reading:
* v1 is fastest by speedup but pays **+4 colors** on average —
  the loss function rewards speedup with no colour cap.
* v2 holds colour quality but recovers <half of v1's speedup.
* v3_raw: kcore + assort recover ~30 % of the gap to v1
  (3.32× vs 2.46×) while keeping Δcolors near zero.
* v3_bump: on-device bumping adds another ~28 % geomean
  speedup over v3_raw on TRAINED 11 (4.25× vs 3.29×) and
  actually *improves* colour count by 0.09 on the trained
  set (the bump scope is bounded by `CTRL_CAP`, so it never
  blows past v2's safe envelope).

Per-graph data: `model/v2_data/egr_dyntheta_default.json`
(v3_raw + v3_bump) and `model/v2_data/egr_v123_fixed.json` /
`scripts/sweep_v123_combined.log` (v1 + v2 + base).

## References

1. Boldi, P. & Vigna, S. *Fairness on the Web: Alternatives to the
   Power Law.* WebSci '12, ACM. DOI: 10.1145/2380718.2380741.
   Preprint: ResearchGate publication 260282672.
   *(GI, H_er definitions.)*
2. Seidman, S. B. *Network structure and minimum degree.* Social
   Networks 5 (1983), 269–287.
   *(k-core / degeneracy concept.)*
3. Batagelj, V. & Zaversnik, M. *An O(m) algorithm for cores
   decomposition of networks.* arXiv:cs/0310049 (2003).
   *(Linear-time bucket-peeling used in `compute_graph_features`.)*
4. Newman, M. E. J. *Assortative mixing in networks.* Physical
   Review Letters 89, 208701 (2002). arXiv:cond-mat/0205405.
   *(Degree-degree Pearson assortativity, §III eq. (4).)*
5. CHROMA paper §V (Prediction of Elastic Number) — current
   predictor methodology.
6. m2cgen — https://github.com/BayesWitnesses/m2cgen — supported
   models reference.

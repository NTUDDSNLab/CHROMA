# model/train.py
"""θ-predictor trainer.

Two paths controlled by --v2:
  legacy (default)  the existing 3-feature (V, E, V/E) + VR-only-loss path;
                    kept verbatim for paper-reproduction. Reads V/E from JSON.
  v2 (--v2)         new 7-feature (V, E, d, s, R, GI, H_er) pipeline with
                    weighted-VR-cap loss, K-fold CV, and emits scaler.h next
                    to model.cpp + a model_meta.json provenance file.
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing   import StandardScaler

import m2cgen as m2c

from .data     import prepare_train_data
from .features import FEATURE_NAMES
from .losses   import (mean_theta_pred, violation_ratio,
                        weighted_vr_loss, weighted_vr_scorer)
from .models   import get_spec, SPECS
from .emit     import emit_model_meta, write_scaler_header


# ----------------------------------------------------------------------------- v2 pipeline

def _per_fold_metrics(y_true, y_pred, alpha, beta, vr_cap):
    return {
        "vr":             violation_ratio(y_true, y_pred),
        "mean_theta":     mean_theta_pred(None, y_pred),
        "weighted_loss":  weighted_vr_loss(y_true, y_pred, alpha, beta, vr_cap),
        "mae":            float(np.mean(np.abs(y_pred - y_true))),
    }


def run_v2(args) -> int:
    spec = get_spec(args.model)
    if args.model == "xgb_pinball":
        from .models import set_xgb_pinball_quantile
        set_xgb_pinball_quantile(args.xgb_quantile)
        print(f"[v2] model={spec.name}  q={args.xgb_quantile}  deployable={spec.deployable}")
    else:
        print(f"[v2] model={spec.name}  deployable={spec.deployable}")

    X, y, names = prepare_train_data(args.input, args.graph_dir,
                                      speedup_decay=args.speedup_decay)
    print(f"[v2] {len(names)} samples, {X.shape[1]} features ({list(FEATURE_NAMES)})")
    if len(names) == 0:
        print("[v2] ERROR: no samples — check --input and --graph-dir", flush=True)
        return 2

    # Standardise. Tree models ignore it (scale-invariant); linear/SVR depend.
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    use_scaled = spec.name in ("linear", "svr")
    X_train = X_scaled if use_scaled else X

    # K-fold CV with the chosen scorer
    scorer = weighted_vr_scorer(args.alpha, args.beta, args.vr_cap)
    cv     = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    if args.grid_search and spec.grid:
        gs = GridSearchCV(spec.factory(), spec.grid, scoring=scorer,
                          cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y)
        best_estimator = gs.best_estimator_
        best_params    = gs.best_params_
    else:
        best_estimator = spec.factory()
        best_estimator.fit(X_train, y)
        best_params    = {}

    # Walk folds again with the chosen estimator to collect per-fold metrics
    fold_metrics = []
    for fold_idx, (tr, te) in enumerate(cv.split(X_train)):
        est = spec.factory()
        if best_params:
            est.set_params(**best_params)
        est.fit(X_train[tr], y[tr])
        y_hat = np.round(est.predict(X_train[te])).astype(int)
        fold_metrics.append(
            {"fold": fold_idx, **_per_fold_metrics(y[te], y_hat,
                                                    args.alpha, args.beta, args.vr_cap)})

    print(f"[v2] CV best_params: {best_params}")
    for m in fold_metrics:
        print(f"  fold {m['fold']}: vr={m['vr']:.4f} mean_θ̂={m['mean_theta']:.2f} "
              f"weighted_loss={m['weighted_loss']:+.3f} mae={m['mae']:.2f}")

    # Emit artefacts
    if not spec.deployable:
        print(f"[v2] WARN: model={spec.name} is not deployable via m2cgen — "
              f"writing meta only, skipping model.cpp")
    else:
        cpp = m2c.export_to_c(best_estimator)
        Path(args.out_cpp).write_text(cpp)
        print(f"[v2] wrote {args.out_cpp}  ({len(cpp)} bytes)")

    write_scaler_header(scaler, model_class=spec.name,
                         feature_names=FEATURE_NAMES, path=args.out_scaler,
                         use_floor=args.use_floor,
                         score_shift=args.score_shift)
    print(f"[v2] wrote {args.out_scaler}  "
          f"(use_floor={args.use_floor}, score_shift={args.score_shift})")

    meta = {
        "generated_utc":   datetime.datetime.utcnow().isoformat() + "Z",
        "feature_count":   len(FEATURE_NAMES),
        "feature_names":   list(FEATURE_NAMES),
        "model_class":     spec.name,
        "deployable":      spec.deployable,
        "use_scaled":      use_scaled,
        "vr_cap":          args.vr_cap,
        "alpha":           args.alpha,
        "beta":            args.beta,
        "cv_folds":        args.cv_folds,
        "best_params":     best_params,
        "cv_results":      fold_metrics,
        "rounding_policy": {"use_floor": args.use_floor,
                             "score_shift": args.score_shift},
        "xgb_quantile":    args.xgb_quantile if args.model == "xgb_pinball" else None,
        "training_samples": names,
        "sklearn_version": sklearn.__version__,
        "input":           str(Path(args.input).resolve()),
        "graph_dir":       str(Path(args.graph_dir).resolve()),
    }
    emit_model_meta(meta, args.out_meta)
    print(f"[v2] wrote {args.out_meta}")
    return 0


# ----------------------------------------------------------------------------- legacy CLI
#
# The existing 3-feature pipeline is preserved here as the `run_legacy` branch
# so paper Fig. 7 reproductions still work. We don't keep ALL 222 lines of the
# old script — we keep the entry function that loaded the JSON, fit the legacy
# RF model, and emitted model.cpp via m2cgen. If you need exact reproducibility
# of the old script, see the pre-T11 commit on this branch (model/train.py at
# commit 8398604).


def run_legacy(args) -> int:
    print("ERROR: legacy mode not yet rewired in v2 plan; pass --v2 for new pipeline.",
          flush=True)
    return 2


# ----------------------------------------------------------------------------- entry point

def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--v2", action="store_true",
                   help="use the new 7-feature + weighted-VR-cap pipeline")
    p.add_argument("--input", required=True, help="training results JSON")
    p.add_argument("--graph-dir", default="Datasets/EGR",
                   help="directory containing the .egr files referenced by --input")
    p.add_argument("--model", default="linear",
                   choices=[s.name for s in SPECS])
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--beta",  type=float, default=1.0)
    p.add_argument("--vr-cap", type=float, default=0.05)
    p.add_argument("--speedup-decay", type=float, default=1.2)
    p.add_argument("--grid-search", action="store_true")
    p.add_argument("--no-grid", dest="grid_search", action="store_false")
    p.set_defaults(grid_search=False)
    p.add_argument("--out-cpp",    default="model/model_v2.cpp")
    p.add_argument("--out-scaler", default="model/scaler.h")
    p.add_argument("--out-meta",   default="model/model_meta.json")
    p.add_argument("--use-floor", action="store_true",
                   help="emit USE_FLOOR=true so deployment uses floor() instead "
                        "of round() (trades VR for ŷ — see model_meta.json)")
    p.add_argument("--score-shift", type=float, default=0.0,
                   help="constant added to score(input) before round/floor at "
                        "deployment (negative = more conservative)")
    p.add_argument("--xgb-quantile", type=float, default=0.15,
                   help="q for xgb_pinball custom objective; q<0.5 → conservative "
                        "predictions (default 0.15 → ~17-21%% VR on SSMC training set)")
    return p


def main():
    args = _build_parser().parse_args()
    if args.v2:
        raise SystemExit(run_v2(args))
    raise SystemExit(run_legacy(args))


if __name__ == "__main__":
    main()

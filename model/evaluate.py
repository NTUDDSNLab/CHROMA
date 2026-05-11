# model/evaluate.py
"""Side-by-side eval of legacy (model.cpp, V/E features) vs v2 (model_v2.cpp,
7 features) θ predictors on the SNAP results JSON.

Refits both predictors on ALL data (mirrors the original training practice
of "fit on full set then emit"). Computes per-graph predictions + aggregate
VR / mean θ̂. Use to decide whether to promote v2 to default in T19.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .data     import compute_max_theta, prepare_train_data
from .features import FEATURE_NAMES


def evaluate(args):
    from sklearn.linear_model  import LinearRegression
    from sklearn.preprocessing import StandardScaler

    # v2 features + targets
    X, y, names = prepare_train_data(args.input, args.graph_dir)
    print(f"[eval] {len(names)} graphs, {X.shape[1]} features", file=sys.stderr)
    if len(names) == 0:
        print("[eval] ERROR: no graphs found in graph dir.", file=sys.stderr)
        return 2

    # v1 feature subset: V, E, V/E (in the order the legacy predictor used)
    V_E_only = X[:, :2]
    V_over_E = V_E_only[:, 0] / V_E_only[:, 1]
    X_v1 = np.column_stack([V_E_only, V_over_E])

    # Train both estimators on all data (matches the legacy "emit from full fit" practice)
    v1_est = LinearRegression().fit(X_v1, y)
    v2_scaler = StandardScaler().fit(X)
    v2_est    = LinearRegression().fit(v2_scaler.transform(X), y)

    v1_pred = np.round(v1_est.predict(X_v1)).astype(int)
    v2_pred = np.round(v2_est.predict(v2_scaler.transform(X))).astype(int)

    # Clamp negatives to 0 (matches CHROMA.cu's runtime clamp)
    v1_pred = np.clip(v1_pred, 0, None)
    v2_pred = np.clip(v2_pred, 0, None)

    # Per-graph table
    print(f"\n{'graph':40s}  {'V':>9s}  {'E':>9s}  {'true':>4s}  "
          f"{'v1':>3s}  {'v2':>3s}  v1over v2over")
    for i, name in enumerate(names):
        v1_over = "!" if v1_pred[i] > y[i] else " "
        v2_over = "!" if v2_pred[i] > y[i] else " "
        print(f"{name[:40]:40s}  {int(X[i,0]):9d}  {int(X[i,1]):9d}  "
              f"{y[i]:4d}  {v1_pred[i]:3d}  {v2_pred[i]:3d}  "
              f"  {v1_over}      {v2_over}")

    # Aggregate
    print()
    print(f"[v1] VR = {(v1_pred > y).mean()*100:5.2f}%   mean θ̂ = {v1_pred.mean():4.2f}   mean true = {y.mean():4.2f}")
    print(f"[v2] VR = {(v2_pred > y).mean()*100:5.2f}%   mean θ̂ = {v2_pred.mean():4.2f}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--graph-dir", default="Datasets/EGR")
    args = p.parse_args()
    raise SystemExit(evaluate(args))


if __name__ == "__main__":
    main()

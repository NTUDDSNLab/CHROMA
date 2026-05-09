"""Training-data prep for the θ predictor.

Two responsibilities:
  * compute_max_theta — derive y from a per-dataset {θ → record} dict.
    Fixes the non-monotonic bug in the legacy prepare_train_data: scans the
    full sorted θ range, no early break on a colour regression, but still
    respects the 1.2× per-step speedup productivity guard.
  * prepare_train_data — assemble (X, y) from a training-results JSON +
    a directory of .egr files (Task 8 wires this in).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping, Tuple, List

import numpy as np

from .features import FEATURE_NAMES, compute_features, load_ecl_graph


def compute_max_theta(per_theta: Mapping[str, dict],
                      baseline_color: int,
                      speedup_decay: float = 1.2) -> int:
    """Return the largest θ in `per_theta` such that
       (a) color(θ) ≤ baseline_color, AND
       (b) speedup(θ-1 → θ) ≥ speedup_decay.

    Scans the full sorted θ range — does NOT break at the first colour
    regression. Non-numeric / non-record keys (e.g. 'vertices', 'edges')
    are ignored.
    """
    sorted_thetas = sorted(int(k) for k in per_theta if k.isdigit())
    if not sorted_thetas:
        return 0

    last_runtime = per_theta["0"]["runtime_ms"]
    best_theta   = 0
    for t in sorted_thetas:
        rec = per_theta[str(t)]
        color = rec.get("color")
        if color is None:
            continue
        cur_runtime = rec["runtime_ms"]
        if t == 0:
            last_runtime = cur_runtime
            continue
        cur_speedup  = (last_runtime / cur_runtime) if cur_runtime > 0 else 0.0
        last_runtime = cur_runtime
        if color <= baseline_color and cur_speedup >= speedup_decay:
            best_theta = t
        # else: keep scanning (do NOT break) — fixes non-monotonic landscape
    return best_theta


def prepare_train_data(
    json_path,
    graph_dir,
    speedup_decay: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build (X, y, names) for training.

      X[i, :]  → 7-feature vector for dataset names[i] (computed from .egr)
      y[i]     → compute_max_theta of dataset names[i] (from JSON records)
      names[i] → graph filename, in JSON-iteration order, skipping any
                 dataset whose .egr is missing from graph_dir.
    """
    json_path = Path(json_path)
    graph_dir = Path(graph_dir)
    with open(json_path) as f:
        data_dict = json.load(f)

    rows_X: list[np.ndarray] = []
    rows_y: list[int]        = []
    names:  list[str]        = []

    for name, per_theta in data_dict.items():
        egr_path = graph_dir / name
        if not egr_path.is_file():
            print(f"[prepare_train_data] skip {name}: .egr not in {graph_dir}",
                  file=sys.stderr)
            continue
        if "0" not in per_theta or per_theta["0"].get("color") is None:
            print(f"[prepare_train_data] skip {name}: missing baseline θ=0",
                  file=sys.stderr)
            continue

        feats = compute_features(load_ecl_graph(egr_path))
        baseline = per_theta["0"]["color"]
        max_theta = compute_max_theta(per_theta, baseline, speedup_decay)

        rows_X.append(np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64))
        rows_y.append(int(max_theta))
        names.append(name)

    if not rows_X:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0, dtype=int), []
    return np.vstack(rows_X), np.asarray(rows_y, dtype=int), names

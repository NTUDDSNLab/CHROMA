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

from typing import Mapping


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

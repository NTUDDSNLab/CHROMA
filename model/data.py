"""Training-data prep for the θ predictor.

Two responsibilities:
  * compute_max_theta — derive y from a per-dataset {θ → record} dict.
    Fixes the non-monotonic bug in the legacy prepare_train_data: scans the
    full sorted θ range, no early break on a colour regression, but still
    respects the 1.2× per-step speedup productivity guard.
  * prepare_train_data — assemble (X, y) from a training-results JSON +
    a directory of .egr files (Task 8 wires this in).

Feature caching:
  prepare_train_data accepts an optional `feature_cache` path. When given:
    * On first run: computes all 9 features, writes JSON cache, proceeds.
    * On subsequent runs: reads from cache (skips .egr load + compute_features).
  The cache is keyed by graph filename and stores all 9 feature values.
  This makes v2/v3 training nearly instant after v1 has warmed the cache.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Tuple, List

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
    feature_cache: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build (X, y, names) for training.

      X[i, :]  → 9-feature vector for dataset names[i] (computed from .egr
                 or loaded from feature_cache)
      y[i]     → compute_max_theta of dataset names[i] (from JSON records)
      names[i] → graph filename, in JSON-iteration order, skipping any
                 dataset whose .egr is missing from graph_dir.

    feature_cache: optional path to a JSON file used to persist computed
      features across training runs. If the file exists, features are loaded
      from it (skipping .egr I/O). Any graph not in the cache is computed and
      appended. The cache is saved after processing all graphs.
    """
    json_path = Path(json_path)
    graph_dir = Path(graph_dir)
    with open(json_path) as f:
        data_dict = json.load(f)

    # Load feature cache if provided
    cache: dict[str, list] = {}
    cache_path: Optional[Path] = None
    if feature_cache is not None:
        cache_path = Path(feature_cache)
        if cache_path.is_file():
            with open(cache_path) as cf:
                cache = json.load(cf)
            print(f"[prepare_train_data] loaded feature cache: {len(cache)} entries "
                  f"from {cache_path}", file=sys.stderr)
        else:
            print(f"[prepare_train_data] feature cache will be created at {cache_path}",
                  file=sys.stderr)

    rows_X: list[np.ndarray] = []
    rows_y: list[int]        = []
    names:  list[str]        = []
    cache_dirty = False
    t0 = time.time()

    for i, (name, per_theta) in enumerate(data_dict.items()):
        egr_path = graph_dir / name
        if not egr_path.is_file():
            print(f"[prepare_train_data] skip {name}: .egr not in {graph_dir}",
                  file=sys.stderr)
            continue
        if "0" not in per_theta or per_theta["0"].get("color") is None:
            print(f"[prepare_train_data] skip {name}: missing baseline θ=0",
                  file=sys.stderr)
            continue

        if name in cache:
            feats = {k: v for k, v in zip(FEATURE_NAMES, cache[name])}
        else:
            t1 = time.time()
            feats = compute_features(load_ecl_graph(egr_path))
            elapsed = time.time() - t1
            if elapsed > 5.0:
                print(f"[prepare_train_data] {name}: features in {elapsed:.1f}s",
                      file=sys.stderr)
            if cache_path is not None:
                cache[name] = [feats[k] for k in FEATURE_NAMES]
                cache_dirty = True

        baseline = per_theta["0"]["color"]
        max_theta = compute_max_theta(per_theta, baseline, speedup_decay)

        rows_X.append(np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64))
        rows_y.append(int(max_theta))
        names.append(name)

        # Checkpoint cache every 20 graphs
        if cache_dirty and cache_path is not None and (i + 1) % 20 == 0:
            cache_path.write_text(json.dumps(cache, indent=2))
            cache_dirty = False
            print(f"[prepare_train_data] cache checkpoint: {len(cache)} entries "
                  f"({time.time()-t0:.0f}s elapsed)", file=sys.stderr)

    # Final cache flush
    if cache_dirty and cache_path is not None:
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"[prepare_train_data] cache saved: {len(cache)} entries", file=sys.stderr)

    if not rows_X:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0, dtype=int), []
    return np.vstack(rows_X), np.asarray(rows_y, dtype=int), names

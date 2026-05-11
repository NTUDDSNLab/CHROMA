import pytest
from model import data


def _mk(theta_records, vertices=100, edges=200):
    """Helper to build a per-dataset dict with the same schema as the real JSON."""
    out = {"vertices": vertices, "edges": edges}
    out.update(theta_records)
    return out


def test_monotonic_simple():
    """Color stays at baseline as θ grows; speedup stays > 1.2× → max_θ = 5."""
    per = {str(t): {"color": 10, "runtime_ms": 1000.0 / (1.5 ** t)}
           for t in range(6)}
    assert data.compute_max_theta(per, baseline_color=10) == 5


def test_break_on_speedup_decay():
    """Color stays good but speedup falls below 1.2× at θ=2 → cap at 1."""
    runtimes = [1000, 800, 700, 690, 685]   # speedups: 1.25, 1.14, ...
    per = {str(t): {"color": 10, "runtime_ms": rt} for t, rt in enumerate(runtimes)}
    # θ=1: speedup 1.25 ≥ 1.2 ✓ best=1; θ=2: 1.143 < 1.2 ✗; θ=3: 1.014 ✗; ...
    assert data.compute_max_theta(per, baseline_color=10) == 1


def test_non_monotonic_color_keeps_scanning():
    """θ=2 regresses (color 12 > baseline 10); θ=4 recovers (color 9).
    With sorted iteration + no early break, max_θ should reach 4."""
    per = {
        "0": {"color": 10, "runtime_ms": 1000.0},
        "1": {"color": 10, "runtime_ms": 800.0},   # speedup 1.25 ≥ 1.2 ✓
        "2": {"color": 12, "runtime_ms": 600.0},   # color regression — old code would break here
        "3": {"color": 11, "runtime_ms": 480.0},   # still > baseline
        "4": {"color":  9, "runtime_ms": 380.0},   # color ≤ baseline AND speedup 1.263 ≥ 1.2
    }
    assert data.compute_max_theta(per, baseline_color=10) == 4


def test_skips_none_color():
    per = {
        "0": {"color": 10, "runtime_ms": 1000.0},
        "1": {"color": None, "runtime_ms": 800.0},   # measurement missing
        "2": {"color":  10, "runtime_ms": 600.0},
    }
    assert data.compute_max_theta(per, baseline_color=10) == 2


def test_keys_not_numeric_ignored():
    per = {
        "0": {"color": 10, "runtime_ms": 1000.0},
        "vertices": 99,
        "edges":    200,
        "1": {"color": 10, "runtime_ms": 800.0},
    }
    assert data.compute_max_theta(per, baseline_color=10) == 1


import numpy as np
from pathlib import Path

FIXDIR = Path(__file__).parent / "fixtures"


def test_prepare_train_data(canonical_egr_fixtures):
    X, y, names = data.prepare_train_data(
        json_path=FIXDIR / "tiny_training.json",
        graph_dir=FIXDIR,
    )
    # 3 graphs, 9 features each (V, E, d, s, R, GI, H_er, kcore, assort — see model/features.py FEATURE_NAMES)
    assert X.shape == (3, 9)
    assert y.shape == (3,)
    assert names == ["K_5.egr", "P_10.egr", "C_10.egr"]

    # Spot-check K_5 row: V=5, E=20, d=4, s=0, R=0
    k5_idx = names.index("K_5.egr")
    assert X[k5_idx, 0] == pytest.approx(5)        # V
    assert X[k5_idx, 1] == pytest.approx(20)       # E (from .egr, not JSON)
    assert X[k5_idx, 2] == pytest.approx(4.0)      # d

    # Targets: K_5 has 3 thetas all colour ≤ 5 with speedups 1.43, 1.4 — both ≥ 1.2 → max=2
    assert y[k5_idx] == 2


def test_prepare_train_data_skips_missing_egr(tmp_path):
    """If a dataset's .egr isn't in graph_dir, the row is skipped with a warning."""
    json_path = tmp_path / "j.json"
    json_path.write_text('{"missing.egr": {"vertices": 1, "edges": 0, "0": {"color": 1, "runtime_ms": 1.0}}}')
    X, y, names = data.prepare_train_data(json_path=json_path, graph_dir=tmp_path)
    assert X.shape == (0, 9)
    assert y.shape == (0,)
    assert names == []

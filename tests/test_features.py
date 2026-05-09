# tests/test_features.py
import math

import pytest
from model import features


def test_load_K5(canonical_egr_fixtures):
    g = features.load_ecl_graph(canonical_egr_fixtures["K_5"])
    assert g.nodes == 5
    assert g.edges == 20
    assert list(g.nindex) == [0, 4, 8, 12, 16, 20]
    # Each vertex's nbrs are the other 4
    for v in range(5):
        nbrs = list(g.nlist[g.nindex[v]:g.nindex[v + 1]])
        assert sorted(nbrs) == [u for u in range(5) if u != v]


def test_load_P10(canonical_egr_fixtures):
    g = features.load_ecl_graph(canonical_egr_fixtures["P_10"])
    assert g.nodes == 10
    assert g.edges == 18


def test_basic_stats_K5(canonical_egr_fixtures):
    f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures["K_5"]))
    # K_5: regular degree-4 graph
    assert f["V"] == 5
    assert f["E"] == 20
    assert f["d"] == pytest.approx(4.0)
    assert f["s"] == pytest.approx(0.0)
    assert f["R"] == pytest.approx(0.0)


def test_basic_stats_P10(canonical_egr_fixtures):
    f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures["P_10"]))
    # Path: 2 endpoints (deg 1) + 8 inner (deg 2). mean = 1.8, var = 0.16
    assert f["V"] == 10
    assert f["E"] == 18
    assert f["d"] == pytest.approx(1.8)
    assert f["s"] == pytest.approx(0.4)
    assert f["R"] == pytest.approx((2 - 1) / 1.8)


def test_basic_stats_star(canonical_egr_fixtures):
    f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures["star_K_1_9"]))
    # 1 hub deg 9 + 9 leaves deg 1. mean = 1.8, var = 5.76, std = 2.4
    assert f["V"] == 10
    assert f["E"] == 18
    assert f["d"] == pytest.approx(1.8)
    assert f["s"] == pytest.approx(2.4)
    assert f["R"] == pytest.approx((9 - 1) / 1.8)


def test_basic_stats_C10(canonical_egr_fixtures):
    f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures["C_10"]))
    assert f["V"] == 10
    assert f["E"] == 20
    assert f["d"] == pytest.approx(2.0)
    assert f["s"] == pytest.approx(0.0)
    assert f["R"] == pytest.approx(0.0)

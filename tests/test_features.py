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


def test_gini_regular_zero(canonical_egr_fixtures):
    """K_5 (all deg 4) and C_10 (all deg 2) → GI = 0."""
    for name in ("K_5", "C_10"):
        f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures[name]))
        assert f["GI"] == pytest.approx(0.0, abs=1e-12), name


def test_gini_P10(canonical_egr_fixtures):
    """P_10 deg=[1,1,2,2,2,2,2,2,2,2]. By MAD formula:
       Σᵢ Σⱼ |dᵢ−dⱼ| = 2·(2 deg-1 × 8 deg-2) = 32; GI = 32 / (2·100·1.8) = 16/180 ≈ 0.0889."""
    f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures["P_10"]))
    assert f["GI"] == pytest.approx(16/180, abs=1e-9)


def test_gini_star(canonical_egr_fixtures):
    """K_{1,9} deg=[9,1,...,1]. MAD = 2·1·9·8 = 144; mean=1.8; GI=144/(2·100·1.8)=0.4."""
    f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures["star_K_1_9"]))
    assert f["GI"] == pytest.approx(0.4, abs=1e-9)


def test_entropy_regular_one(canonical_egr_fixtures):
    """Regular graphs: every pᵢ = 1/n → H = log₂ n → H_er = 1."""
    for name in ("K_5", "C_10"):
        f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures[name]))
        assert f["H_er"] == pytest.approx(1.0, abs=1e-9), name


def test_entropy_P10(canonical_egr_fixtures):
    """P_10: 2 vertices p=1/18, 8 vertices p=2/18.
       H = -2·(1/18)·log₂(1/18) - 8·(2/18)·log₂(2/18)
         = log₂(18) - 16/18
       H_er = H / log₂(10)"""
    H = math.log2(18) - 16/18
    expected = H / math.log2(10)
    f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures["P_10"]))
    assert f["H_er"] == pytest.approx(expected, abs=1e-9)


def test_entropy_star(canonical_egr_fixtures):
    """K_{1,9}: 1 hub p=9/18=0.5, 9 leaves p=1/18.
       H = -1·0.5·log₂(0.5) - 9·(1/18)·log₂(1/18)
         = 0.5 + 0.5·log₂(18)
       H_er = H / log₂(10)"""
    H = 0.5 + 0.5 * math.log2(18)
    expected = H / math.log2(10)
    f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures["star_K_1_9"]))
    assert f["H_er"] == pytest.approx(expected, abs=1e-9)


# --- v3: kcore + assortativity ---

def test_kcore_canonical(canonical_egr_fixtures):
    """K_5: complete graph → degeneracy=4 (everyone deg 4 even after peeling)
    P_10: path → degeneracy=1 (always remove a deg-1 endpoint)
    C_10: cycle → degeneracy=2 (regular, peel order destroys it as 2-core)
    star K_{1,9}: 1 hub deg 9 + 9 leaves deg 1 → degeneracy=1 (peel leaves first)
    """
    cases = {"K_5": 4, "P_10": 1, "C_10": 2, "star_K_1_9": 1}
    for name, expected in cases.items():
        f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures[name]))
        assert f["kcore"] == expected, f"{name}: got kcore={f['kcore']}, expected {expected}"


def test_assortativity_constant_degree_zero(canonical_egr_fixtures):
    """K_5 (all deg 4) and C_10 (all deg 2): degree variance = 0 → return 0 by convention."""
    for name in ("K_5", "C_10"):
        f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures[name]))
        assert f["assort"] == pytest.approx(0.0, abs=1e-12), name


def test_assortativity_P10(canonical_egr_fixtures):
    """P_10: 4 directed edges incident to a deg-1 vertex (paired with deg-2),
    14 inner directed edges (deg-2 to deg-2). E[xy]=64/18=32/9, E[x]=E[y]=17/9.
    cov = 32/9 − 289/81 = −1/81. Var = 11/3 − 289/81 = 8/81. r = −1/8 = −0.125."""
    f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures["P_10"]))
    assert f["assort"] == pytest.approx(-0.125, abs=1e-9)


def test_assortativity_star_perfectly_disassortative(canonical_egr_fixtures):
    """K_{1,9}: every directed edge is between deg=9 hub and deg=1 leaf → r = −1."""
    f = features.compute_features(features.load_ecl_graph(canonical_egr_fixtures["star_K_1_9"]))
    assert f["assort"] == pytest.approx(-1.0, abs=1e-9)

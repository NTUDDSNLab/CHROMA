# tests/test_features.py
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

# tests/conftest.py
import struct
from pathlib import Path
import pytest

FIXDIR = Path(__file__).parent / "fixtures"
FIXDIR.mkdir(exist_ok=True)


def write_egr(path, nodes, nindex, nlist):
    """Write an ECLgraph .egr file: int nodes, int edges, int nindex[nodes+1],
    int nlist[edges]. No edge weights (matches ECLgraph.h reader's optional path)."""
    edges = len(nlist)
    assert len(nindex) == nodes + 1
    assert nindex[-1] == edges
    with open(path, "wb") as f:
        f.write(struct.pack("ii", nodes, edges))
        f.write(struct.pack(f"{nodes+1}i", *nindex))
        f.write(struct.pack(f"{edges}i", *nlist))


def _build_K5():
    # Complete graph on 5 nodes — all degrees 4, 20 directed edges
    nlist = []
    nindex = [0]
    for v in range(5):
        nbrs = [u for u in range(5) if u != v]
        nlist.extend(nbrs)
        nindex.append(len(nlist))
    return 5, nindex, nlist


def _build_P10():
    # Path 0-1-2-...-9 — degree 1 for endpoints, 2 elsewhere, 18 directed
    nlist = []
    nindex = [0]
    for v in range(10):
        nbrs = []
        if v > 0:   nbrs.append(v - 1)
        if v < 9:   nbrs.append(v + 1)
        nlist.extend(nbrs)
        nindex.append(len(nlist))
    return 10, nindex, nlist


def _build_C10():
    # Cycle on 10 — all degree 2, 20 directed
    nlist = []
    nindex = [0]
    for v in range(10):
        nbrs = [(v - 1) % 10, (v + 1) % 10]
        nlist.extend(sorted(nbrs))
        nindex.append(len(nlist))
    return 10, nindex, nlist


def _build_star():
    # K_{1,9} — node 0 connected to 1..9; 18 directed
    nlist = []
    nindex = [0]
    nlist.extend(range(1, 10)); nindex.append(len(nlist))   # node 0
    for v in range(1, 10):
        nlist.append(0); nindex.append(len(nlist))
    return 10, nindex, nlist


@pytest.fixture(scope="session", autouse=True)
def canonical_egr_fixtures():
    """Generate the 4 canonical .egr fixtures once per test session."""
    for name, builder in [
        ("K_5", _build_K5),
        ("P_10", _build_P10),
        ("C_10", _build_C10),
        ("star_K_1_9", _build_star),
    ]:
        path = FIXDIR / f"{name}.egr"
        if not path.exists():
            n, nidx, nlist = builder()
            write_egr(path, n, nidx, nlist)
    return {n: FIXDIR / f"{n}.egr"
            for n in ("K_5", "P_10", "C_10", "star_K_1_9")}

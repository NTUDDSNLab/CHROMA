# model/features.py
"""Graph feature extraction for the θ predictor.

Loads CHROMA's ECL `.egr` binary CSR format and computes nine features
matched to the C++ implementation in lib/io/graph_features.cpp:
  V       number of vertices
  E       number of directed edges (each undirected edge counted twice)
  d       average degree (= E / V)
  s       population standard deviation of degrees
  R       relative range of degree, (max − min) / d
  GI      Gini index of degree distribution (sorted-rank form)
  H_er    relative edge-distribution entropy, normalised by log₂(V)
  kcore   degeneracy = max k such that the k-core is non-empty
  assort  degree-degree Pearson assortativity over directed edges

References:
  GI / H_er : Boldi & Vigna 2012, "Fairness on the Web".
  kcore     : Seidman 1983; Batagelj & Zaversnik 2003 (linear-time peeling).
  assort    : Newman 2002, "Assortative mixing in networks".
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ECLGraph:
    nodes: int
    edges: int
    nindex: np.ndarray   # int32, length nodes+1
    nlist:  np.ndarray   # int32, length edges


def load_ecl_graph(path) -> ECLGraph:
    """Read CHROMA's .egr binary CSR. Mirrors lib/io/ECLgraph.h."""
    path = Path(path)
    with open(path, "rb") as f:
        nodes, edges = struct.unpack("ii", f.read(8))
        nindex = np.frombuffer(f.read(4 * (nodes + 1)), dtype=np.int32)
        nlist  = np.frombuffer(f.read(4 * edges), dtype=np.int32)
    if nindex[-1] != edges:
        raise ValueError(f"{path}: nindex[-1]={nindex[-1]} != edges={edges}")
    return ECLGraph(nodes=int(nodes), edges=int(edges),
                    nindex=nindex, nlist=nlist)


FEATURE_NAMES: tuple[str, ...] = ("V", "E", "d", "s", "R", "GI", "H_er", "kcore", "assort")


def _degree_array(g: ECLGraph) -> np.ndarray:
    """deg[v] = nindex[v+1] − nindex[v]. int64 to dodge overflow on big graphs."""
    return np.diff(g.nindex.astype(np.int64))


def _gini(deg: np.ndarray) -> float:
    """Sorted-rank Gini: G = (Σᵢ (2i−n−1) dᵢ) / (n · Σdᵢ), i = 1..n on sorted dᵢ.
    Equivalent to the MAD form (Σᵢⱼ |dᵢ−dⱼ| / (2 n² μ)) — see plan task notes."""
    n = deg.size
    s = deg.sum()
    if n == 0 or s == 0:
        return 0.0
    sorted_d = np.sort(deg.astype(np.float64))
    coeffs   = (2.0 * np.arange(1, n + 1) - n - 1)
    return float((coeffs * sorted_d).sum() / (n * s))


def _relative_entropy(deg: np.ndarray, m: int) -> float:
    """H_er = (−Σ pᵢ log₂ pᵢ) / log₂ n, pᵢ = dᵢ / m. Returns 1 for regular
    graphs, 0 in the degenerate empty case."""
    n = deg.size
    if n <= 1 or m == 0:
        return 0.0
    p = deg.astype(np.float64) / float(m)
    nz = p > 0                   # 0 log 0 ≡ 0
    H = -np.sum(p[nz] * np.log2(p[nz]))
    return float(H / np.log2(n))


def _kcore_max(g: ECLGraph) -> int:
    """Degeneracy = max k such that k-core is non-empty.

    Bucket-based peeling (Batagelj & Zaversnik 2003): repeatedly remove the
    minimum-degree vertex; track the maximum core number reached. O(V + E).
    """
    n = g.nodes
    if n == 0:
        return 0
    deg     = _degree_array(g).astype(np.int64).tolist()
    nindex  = g.nindex.astype(np.int64)
    nlist   = g.nlist.astype(np.int64)
    max_d   = max(deg) if deg else 0

    bucket  = [[] for _ in range(max_d + 2)]
    pos     = [0] * n
    for v in range(n):
        bucket[deg[v]].append(v)
        pos[v] = len(bucket[deg[v]]) - 1

    degeneracy = 0
    cur_d      = 0
    removed    = [False] * n
    for _ in range(n):
        # Find the smallest d with a non-empty bucket (monotone scan).
        while cur_d <= max_d and not bucket[cur_d]:
            cur_d += 1
        if cur_d > max_d:
            break
        v = bucket[cur_d].pop()
        removed[v] = True
        if cur_d > degeneracy:
            degeneracy = cur_d
        for u in nlist[nindex[v]:nindex[v + 1]]:
            if removed[u]:
                continue
            old = deg[u]
            if old <= cur_d:
                continue                 # already at or below current peel level
            # remove u from bucket[old]
            ob = bucket[old]
            p_u = pos[u]
            last = ob[-1]
            ob[p_u] = last
            pos[last] = p_u
            ob.pop()
            deg[u] = old - 1
            bucket[old - 1].append(u)
            pos[u] = len(bucket[old - 1]) - 1
            if old - 1 < cur_d:          # demoted below current scan: rewind
                cur_d = old - 1
    return int(degeneracy)


def _assortativity(g: ECLGraph) -> float:
    """Pearson correlation of (deg(u), deg(v)) over directed edges (u, v).
    Returns 0 when degrees are constant (correlation undefined). The sign
    distinguishes assortative (+) from disassortative (−) mixing.

    Newman 2002 "Assortative mixing in networks" §III, eq. (4)."""
    n = g.nodes
    if n < 2 or g.edges == 0:
        return 0.0
    deg = _degree_array(g).astype(np.float64)
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(g.nindex.astype(np.int64)))
    dst = g.nlist.astype(np.int64)
    x   = deg[src]
    y   = deg[dst]
    var_x = float(x.var())
    var_y = float(y.var())
    if var_x == 0.0 or var_y == 0.0:
        return 0.0                       # constant degrees → r is undefined
    return float(np.corrcoef(x, y)[0, 1])


def compute_features(g: ECLGraph) -> dict:
    """Return dict with the 9 features defined in FEATURE_NAMES."""
    n = g.nodes
    m = g.edges
    if n == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    deg = _degree_array(g)
    d_mean = float(m) / float(n)            # since Σ deg(v) = m for ECL .egr
    d_max = int(deg.max())
    d_min = int(deg.min())
    s = float(np.sqrt(((deg - d_mean) ** 2).mean()))
    r = (d_max - d_min) / d_mean if d_mean > 0 else 0.0

    return {
        "V":      float(n),
        "E":      float(m),
        "d":      d_mean,
        "s":      s,
        "R":      float(r),
        "GI":     _gini(deg),
        "H_er":   _relative_entropy(deg, m),
        "kcore":  float(_kcore_max(g)),
        "assort": _assortativity(g),
    }

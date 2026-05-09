# model/features.py
"""Graph feature extraction for the θ predictor.

Loads CHROMA's ECL `.egr` binary CSR format and computes seven
features matched to the C++ implementation in lib/io/graph_features.cpp:
  V       number of vertices
  E       number of directed edges (each undirected edge counted twice)
  d       average degree (= E / V)
  s       population standard deviation of degrees
  R       relative range of degree, (max − min) / d
  GI      Gini index of degree distribution (sorted-rank form)
  H_er    relative edge-distribution entropy, normalised by log₂(V)

Reference for GI / H_er: Boldi & Vigna 2012, "Fairness on the Web".
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


FEATURE_NAMES: tuple[str, ...] = ("V", "E", "d", "s", "R", "GI", "H_er")


def _degree_array(g: ECLGraph) -> np.ndarray:
    """deg[v] = nindex[v+1] − nindex[v]. int64 to dodge overflow on big graphs."""
    return np.diff(g.nindex.astype(np.int64))


def compute_features(g: ECLGraph) -> dict:
    """Return dict with the 7 features defined in FEATURE_NAMES."""
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

    # Placeholders for GI / H_er — wired in Tasks 4 + 5
    return {
        "V":    float(n),
        "E":    float(m),
        "d":    d_mean,
        "s":    s,
        "R":    float(r),
        "GI":   0.0,
        "H_er": 0.0,
    }

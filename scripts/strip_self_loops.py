#!/usr/bin/env python3
"""Strip self-loops from .egr graphs so CHROMA's coloring pipeline can use them.

For each input file, removes any nlist entry where nlist[i] == owner(i), then
rewrites a new .egr (default suffix `.nl.egr` next to the original — never
overwrites the source unless --in-place).
"""
from __future__ import annotations
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.features import load_ecl_graph


def write_egr(path: Path, nodes: int, nindex: np.ndarray, nlist: np.ndarray) -> None:
    edges = nlist.size
    assert nindex.size == nodes + 1
    assert int(nindex[-1]) == edges
    with open(path, "wb") as f:
        f.write(struct.pack("ii", nodes, edges))
        f.write(nindex.astype(np.int32).tobytes())
        f.write(nlist.astype(np.int32).tobytes())


def strip(g_path: Path, out_path: Path) -> tuple[int, int]:
    """Strip self-loops from g_path, write to out_path. Returns (n_stripped, edges_after)."""
    g = load_ecl_graph(g_path)
    if g.nodes == 0 or g.edges == 0:
        write_egr(out_path, g.nodes,
                  np.zeros(g.nodes + 1, dtype=np.int32),
                  np.zeros(0, dtype=np.int32))
        return 0, 0

    owner = np.repeat(np.arange(g.nodes, dtype=np.int64),
                       np.diff(g.nindex.astype(np.int64)))
    keep = (owner != g.nlist.astype(np.int64))
    n_stripped = int((~keep).sum())
    new_nlist = g.nlist[keep].astype(np.int32, copy=False)

    # Recompute nindex by counting per-vertex kept entries
    keep_per_v = np.bincount(owner[keep].astype(np.int64), minlength=g.nodes)
    new_nindex = np.zeros(g.nodes + 1, dtype=np.int32)
    new_nindex[1:] = np.cumsum(keep_per_v).astype(np.int32)

    write_egr(out_path, g.nodes, new_nindex, new_nlist)
    return n_stripped, int(new_nindex[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help=".egr files to strip")
    ap.add_argument("--suffix", default=".nl.egr",
                    help="output filename suffix (default .nl.egr)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite input file (DESTRUCTIVE; default is to write a sibling)")
    ap.add_argument("--out-dir", default=None,
                    help="if set, write all outputs into this directory (created if missing)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_stripped = 0
    for in_str in args.inputs:
        p = Path(in_str)
        if args.in_place:
            out = p
        elif out_dir:
            out = out_dir / (p.stem + args.suffix)
        else:
            out = p.with_suffix("")  # drop .egr
            out = out.parent / (out.name + args.suffix)
        n, e_after = strip(p, out)
        total_stripped += n
        print(f"  {p.name:38s}  stripped {n:>6}  edges_after={e_after:>10}  -> {out.name}")

    print(f"\nTotal self-loops removed: {total_stripped}")


if __name__ == "__main__":
    main()

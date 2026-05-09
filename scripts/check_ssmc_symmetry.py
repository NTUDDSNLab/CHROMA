#!/usr/bin/env python3
"""Audit Datasets/SSMC/*.egr for CHROMA compatibility.

For each graph check the three properties CHROMA's PA/CA pipeline assumes:
  symmetric : for every (u, v) entry in nlist[u], there is a (v, u) entry in nlist[v]
  no_loops  : no v ∈ nlist[u] equals u
  csr_ok    : nindex strictly non-decreasing, nindex[-1] == edges, all neighbours in [0, nodes)

Prints per-graph status and a final summary. Exits 0 if all graphs are CHROMA-compatible.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the worktree's loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.features import load_ecl_graph, ECLGraph


def csr_well_formed(g: ECLGraph) -> tuple[bool, str]:
    if len(g.nindex) != g.nodes + 1:
        return False, f"nindex length {len(g.nindex)} != nodes+1 {g.nodes+1}"
    if g.nindex[0] != 0:
        return False, f"nindex[0] = {g.nindex[0]} != 0"
    if int(g.nindex[-1]) != g.edges:
        return False, f"nindex[-1] = {int(g.nindex[-1])} != edges {g.edges}"
    deltas = np.diff(g.nindex)
    if (deltas < 0).any():
        return False, "nindex not non-decreasing"
    if g.nlist.size:
        lo = int(g.nlist.min())
        hi = int(g.nlist.max())
        if lo < 0 or hi >= g.nodes:
            return False, f"nlist out of range [{lo}, {hi}] vs nodes {g.nodes}"
    return True, ""


def has_self_loops(g: ECLGraph) -> int:
    """Count self-loops (entries where nlist[i] == owner vertex)."""
    if g.nodes == 0:
        return 0
    owner = np.repeat(np.arange(g.nodes, dtype=np.int64), np.diff(g.nindex))
    return int(np.sum(owner == g.nlist.astype(np.int64)))


def is_symmetric(g: ECLGraph) -> tuple[bool, int]:
    """Symmetric iff edge-set equals its reverse. Returns (ok, n_asymmetric_directed_edges).

    Encodes each directed edge as (u * nodes + v) and compares the multiset to its
    reverse. With int64 we tolerate up to nodes^2 ≈ 4.6e18 (well above any real graph).
    """
    if g.edges == 0:
        return True, 0
    n = np.int64(g.nodes)
    src = np.repeat(np.arange(g.nodes, dtype=np.int64), np.diff(g.nindex))
    dst = g.nlist.astype(np.int64)
    fwd = src * n + dst
    rev = dst * n + src
    fwd_sorted = np.sort(fwd)
    rev_sorted = np.sort(rev)
    if np.array_equal(fwd_sorted, rev_sorted):
        return True, 0
    # multiset diff: count entries in fwd that aren't matched by an entry in rev
    missing = np.setdiff1d(fwd_sorted, rev_sorted, assume_unique=False).size
    return False, int(missing)


def audit_one(path: Path) -> dict:
    t0 = time.time()
    try:
        g = load_ecl_graph(path)
    except Exception as e:
        return {"path": path, "ok": False, "load_err": str(e)}

    ok_csr, csr_msg = csr_well_formed(g)
    if not ok_csr:
        return {"path": path, "ok": False, "csr_err": csr_msg,
                "nodes": g.nodes, "edges": g.edges,
                "elapsed_s": time.time() - t0}

    loops = has_self_loops(g)
    sym_ok, n_asym = is_symmetric(g)
    return {
        "path": path,
        "ok": (loops == 0) and sym_ok,
        "nodes": g.nodes,
        "edges": g.edges,
        "self_loops": loops,
        "symmetric": sym_ok,
        "asym_edges": n_asym,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssmc-dir", default="/home/chsieh45/PunchShadow/CHROMA/Datasets/SSMC")
    ap.add_argument("--max-size-mb", type=int, default=200,
                    help="skip graphs larger than this (default 200; symmetric "
                         "check uses ~16·m bytes of RAM)")
    args = ap.parse_args()

    ssmc = Path(args.ssmc_dir)
    paths = sorted(p for p in ssmc.glob("*.egr") if p.is_file())
    print(f"Found {len(paths)} .egr files in {ssmc}", file=sys.stderr)

    results = []
    for p in paths:
        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > args.max_size_mb:
            print(f"  SKIP {p.name}  ({size_mb:.0f} MB > {args.max_size_mb} MB)",
                  file=sys.stderr)
            results.append({"path": p, "ok": None, "skipped_size_mb": size_mb})
            continue
        r = audit_one(p)
        results.append(r)
        if r.get("ok"):
            tag = "OK"
        elif r.get("load_err"):
            tag = f"LOAD-ERR ({r['load_err']})"
        elif r.get("csr_err"):
            tag = f"CSR-ERR ({r['csr_err']})"
        else:
            issues = []
            if not r.get("symmetric", True):
                issues.append(f"asym_edges={r['asym_edges']}")
            if r.get("self_loops", 0) > 0:
                issues.append(f"self_loops={r['self_loops']}")
            tag = "FAIL " + " ".join(issues)
        print(f"  {tag:55s} {p.name:40s} V={r.get('nodes', '?'):>10}  "
              f"E={r.get('edges', '?'):>10}  ({r.get('elapsed_s', 0):.1f}s)",
              file=sys.stderr)

    # Summary
    audited   = [r for r in results if r.get("ok") is not None]
    ok        = [r for r in audited if r["ok"]]
    asym      = [r for r in audited if not r["ok"] and r.get("symmetric") is False]
    loops     = [r for r in audited if not r["ok"] and r.get("self_loops", 0) > 0]
    csr_err   = [r for r in audited if r.get("csr_err")]
    load_err  = [r for r in audited if r.get("load_err")]
    skipped   = [r for r in results if r.get("skipped_size_mb")]

    print()
    print(f"Total .egr files     : {len(results)}")
    print(f"Audited              : {len(audited)}")
    print(f"  OK (symmetric+clean): {len(ok)}")
    print(f"  asymmetric          : {len(asym)}")
    print(f"  has self-loops      : {len(loops)}")
    print(f"  CSR malformed       : {len(csr_err)}")
    print(f"  load error          : {len(load_err)}")
    print(f"Skipped (size cap)   : {len(skipped)}")

    if ok:
        print(f"\nCHROMA-compatible graphs ({len(ok)}):")
        for r in ok:
            print(f"  {r['path'].name}  V={r['nodes']}  E={r['edges']}")

    return 0 if not (asym or loops or csr_err or load_err) else 1


if __name__ == "__main__":
    sys.exit(main())

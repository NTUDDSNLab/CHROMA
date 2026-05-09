#!/usr/bin/env python3
"""Sweep CHROMA across SSMC graphs (≤50 MB) at theta 0..10, single run each.
Output JSON in the format consumed by model/train.py --v2."""
from __future__ import annotations
import argparse, json, re, subprocess, sys, time
from pathlib import Path

RUNTIME_RE = re.compile(r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
COLORS_RE  = re.compile(r"colors\s+used:\s*(\d+)", re.IGNORECASE)
NODES_RE   = re.compile(r"^Nodes:\s*(\d+)$", re.IGNORECASE | re.MULTILINE)
EDGES_RE   = re.compile(r"^Edges:\s*(\d+)$", re.IGNORECASE | re.MULTILINE)
VERIF_OK   = re.compile(r"verification\s+passed", re.IGNORECASE)


def run_one(binary: str, graph: Path, theta: int, algo: str, timeout: int):
    """Run CHROMA once. Returns dict with color/runtime_ms/nodes/edges or None on failure."""
    cmd = [binary, "-f", str(graph), "-a", algo, "-e", str(theta)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = proc.stdout
    if proc.returncode != 0 or not VERIF_OK.search(out):
        return None
    rt = RUNTIME_RE.search(out)
    cc = COLORS_RE.search(out)
    nn = NODES_RE.search(out)
    ee = EDGES_RE.search(out)
    if not (rt and cc):
        return None
    rec = {
        "color":      int(cc.group(1)),
        "runtime_ms": float(rt.group(1)),
    }
    if nn and ee:
        rec["_nodes"] = int(nn.group(1))   # used only for top-level vertices/edges
        rec["_edges"] = int(ee.group(1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="/home/chsieh45/PunchShadow/CHROMA/CHROMA/CHROMA")
    ap.add_argument("--dataset-dir", default="/home/chsieh45/PunchShadow/CHROMA/Datasets/SSMC")
    ap.add_argument("--max-size-mb", type=int, default=50)
    ap.add_argument("--theta-start", type=int, default=0)
    ap.add_argument("--theta-end",   type=int, default=10)
    ap.add_argument("--algorithm", default="cuSL_ELS_SDC_CTA_S")
    ap.add_argument("--timeout", type=int, default=300)  # 5 min per run
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="for testing: process only first N graphs")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    max_bytes = args.max_size_mb * 1024 * 1024
    graphs = sorted(p for p in dataset_dir.glob("*.egr") if p.stat().st_size <= max_bytes)
    if args.limit:
        graphs = graphs[: args.limit]
    print(f"[sweep] {len(graphs)} graphs (≤{args.max_size_mb} MB)", file=sys.stderr)

    out: dict = {}
    t_start = time.time()
    for gi, g in enumerate(graphs):
        per_theta = {}
        first_rec = None
        for theta in range(args.theta_start, args.theta_end + 1):
            rec = run_one(args.binary, g, theta, args.algorithm, args.timeout)
            if rec is None:
                print(f"  [{gi+1}/{len(graphs)}] {g.name}: theta={theta} FAILED",
                      file=sys.stderr)
                # If theta=0 (baseline) fails, skip this graph entirely
                if theta == 0:
                    per_theta = None
                    break
                continue
            if first_rec is None:
                first_rec = rec
            per_theta[str(theta)] = {
                "color":      rec["color"],
                "runtime_ms": rec["runtime_ms"],
            }
        if not per_theta:
            print(f"  [{gi+1}/{len(graphs)}] {g.name}: SKIP (baseline failed)", file=sys.stderr)
            continue
        entry: dict = dict(per_theta)
        if first_rec and "_nodes" in first_rec:
            entry["vertices"] = first_rec["_nodes"]
            entry["edges"]    = first_rec["_edges"]
        out[g.name] = entry
        elapsed = time.time() - t_start
        print(f"  [{gi+1}/{len(graphs)}] {g.name}: {len(per_theta)} thetas, "
              f"colors={[v['color'] for v in per_theta.values()]} "
              f"({elapsed:.0f}s elapsed)", file=sys.stderr)
        # Periodic checkpoint write
        if (gi + 1) % 10 == 0:
            Path(args.out).write_text(json.dumps(out, indent=2))
            print(f"  [checkpoint] wrote {args.out} ({len(out)} graphs)", file=sys.stderr)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[sweep] done. {len(out)} graphs in {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

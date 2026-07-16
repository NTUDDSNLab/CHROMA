#!/usr/bin/env python3
"""Sweep η (cta_s_threshold, CLI --eta) impact on cuSL_ELS_SDC_CTA_S.

η is the phase-2 dispatch threshold: remove_size < η takes the SDC
warp-per-vertex path, remove_size >= η the CTA-balanced path. Only the
CTA_S kernels read it, so the coloring result is η-invariant — runtime
is the metric of interest.

For each dataset (default: every .egr under --dataset-dir) and each η:
run `CHROMA -a <algo> -e <θ> --eta <η>` RUNS times, keep the best run
(min colors, tie -> min runtime); record {color, runtime_ms,
iter_count}. Writes eta_impact_results.json consumed by
plot_eta_impact.py.

Examples:
    python3 scripts/plots/eta_impact/eta_impact.py
    python3 scripts/plots/eta_impact/eta_impact.py \\
        --datasets cit-Patents --etas 512 2048 8192 --runs 2 --out /tmp/ei.json
"""
from __future__ import annotations
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

RUNTIME_RE = re.compile(r"Total\s+runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
COLORS_RE = re.compile(r"colors\s+used:\s*(\d+)", re.IGNORECASE)
ITER_RE = re.compile(r"Iter\s+count:\s*(\d+)", re.IGNORECASE)
DISPATCH_RE = re.compile(r"CTA_S dispatch:\s*warp_iters=(\d+)\s+cta_iters=(\d+)"
                         r"\s+warp_nodes=(\d+)\s+cta_nodes=(\d+)")

# 0 = pure-CTA baseline (the dispatch never picks the warp path), then ×2
# up to 64K around the default (2048), then ×4 out to 64M — past the
# largest graph's node count, i.e. the pure-SDC limit — to expose the
# turnover where the warp-per-vertex path starts losing to CTA.
DEFAULT_ETAS = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                262144, 1048576, 4194304, 16777216, 67108864]


def read_egr_header(path: Path) -> tuple:
    with open(path, "rb") as f:
        nodes, edges = struct.unpack("<ii", f.read(8))
    return nodes, edges


def resolve_egr(ds_dir: Path, stem: str) -> Optional[Path]:
    for p in sorted(ds_dir.glob("*.egr")):
        if p.stem.split('.')[0] == stem or p.stem == stem:
            return p
    return None


def run(cmd: list, timeout: int):
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0, f"TIMEOUT after {timeout}s"
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").splitlines()[-3:]
        return None, dt, f"rc={r.returncode}: {' | '.join(tail)}"
    return r, dt, None


def parse_sweep(stdout: str) -> Optional[dict]:
    mr, mc = RUNTIME_RE.search(stdout), COLORS_RE.search(stdout)
    if not (mr and mc):
        return None
    mi = ITER_RE.search(stdout)
    rec = {
        "runtime_ms": float(mr.group(1)),
        "color": int(mc.group(1)),
        "iter_count": int(mi.group(1)) if mi else None,
    }
    md = DISPATCH_RE.search(stdout)
    if md:
        wi, ci, wn, cn = map(int, md.groups())
        rec["warp_iter_pct"] = 100.0 * wi / max(wi + ci, 1)
        rec["warp_node_pct"] = 100.0 * wn / max(wn + cn, 1)
    return rec


def best_of(runs: list) -> Optional[dict]:
    if not runs:
        return None
    return sorted(runs, key=lambda x: (x["color"], x["runtime_ms"]))[0]


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--binary", default=str(repo / "CHROMA" / "CHROMA"))
    ap.add_argument("--dataset-dir", default=str(repo / "Datasets" / "EGR"))
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Dataset stems (default: every .egr in --dataset-dir)")
    ap.add_argument("--algo", default="cuSL_ELS_SDC_CTA_S",
                    help="Only the CTA_S kernels read --eta.")
    ap.add_argument("--elastic", type=int, default=10,
                    help="Fixed θ (-e) used for every η run (default 10)")
    ap.add_argument("--etas", nargs="+", type=int, default=DEFAULT_ETAS)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1200,
                    help="Per-CHROMA-invocation timeout (seconds)")
    ap.add_argument("--out", default=str(here / "eta_impact_results.json"))
    args = ap.parse_args()

    if not os.access(args.binary, os.X_OK):
        sys.exit(f"ERROR: CHROMA binary not found or not executable at "
                 f"{args.binary}")
    ds_dir = Path(args.dataset_dir)
    datasets = args.datasets or sorted(
        {p.stem.split('.')[0] for p in ds_dir.glob("*.egr")})
    if not datasets:
        sys.exit(f"ERROR: no .egr files in {ds_dir}")

    data = {}
    for stem in datasets:
        egr = resolve_egr(ds_dir, stem)
        if egr is None:
            print(f"# {stem:16s} SKIP: no .egr in {ds_dir}", file=sys.stderr)
            data[stem] = {"error": f"no .egr for {stem} in {ds_dir}"}
            continue
        nodes, edges = read_egr_header(egr)
        entry = {"nodes": nodes, "edges": edges, "sweep": {}}
        for eta in args.etas:
            recs = []
            last_err = None
            for _ in range(args.runs):
                r, _dt, err = run(
                    [args.binary, "-f", str(egr), "-a", args.algo,
                     "-e", str(args.elastic), "--eta", str(eta)],
                    args.timeout)
                if err:
                    last_err = err
                    continue
                rec = parse_sweep(r.stdout)
                if rec is None:
                    last_err = "parse-failed (no runtime/colors line)"
                    continue
                recs.append(rec)
            best = best_of(recs)
            if best is None:
                entry["sweep"][str(eta)] = {
                    "error": last_err or "all runs failed"}
                print(f"# {stem:16s} η={eta:6d} FAIL ({last_err})",
                      file=sys.stderr)
            else:
                entry["sweep"][str(eta)] = best
                print(f"# {stem:16s} η={eta:6d} colors={best['color']:3d} "
                      f"rt={best['runtime_ms']:9.3f}ms "
                      f"iter={best['iter_count']}", file=sys.stderr)
        data[stem] = entry

    out = {
        "datasets": datasets,
        "algo": args.algo,
        "elastic": args.elastic,
        "etas": args.etas,
        "runs": args.runs,
        "data": data,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

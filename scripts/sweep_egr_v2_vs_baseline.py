#!/usr/bin/env python3
"""Sweep cuSL_ELS_SDC_CTA_S over Datasets/EGR/*.egr comparing:
  baseline : --elastic 0  (no EGC)
  v2 RF    : --predict --v2-model  (current deployed RF + floor + shift=-0.5)

Reports predicted θ, color count, total runtime per graph; per-mode summaries.
Uses CHROMA's built-in --runs to average noise.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

EGC_RE       = re.compile(r"EGC θ:\s*(-?\d+)", re.IGNORECASE)
# --runs N format prints "Total time : avg=X.XX min=Y max=Z"
AVG_TOTAL_RE = re.compile(r"^\s*Total\s+time\s*:\s*avg=\s*([0-9.]+)", re.IGNORECASE | re.MULTILINE)
AVG_COLOR_RE = re.compile(r"^\s*colors\s+used\s*:\s*avg=\s*([0-9.]+)",  re.IGNORECASE | re.MULTILINE)
AVG_ITER_RE  = re.compile(r"^\s*iter\s+count\s*:\s*avg=\s*([0-9.]+)",   re.IGNORECASE | re.MULTILINE)
MIN_TOTAL_RE = re.compile(r"^\s*Total\s+time\s*:\s*avg=\s*[0-9.]+\s+min=\s*([0-9.]+)", re.IGNORECASE | re.MULTILINE)
# Each run logs "[Run k/N] ... colors: C  iters: I" — collect at least one to confirm verification
RUN_RE       = re.compile(r"\[Run \d+/\d+\][^\n]*colors:\s*(\d+)\s+iters:\s*(\d+)")
VERIF_OK     = re.compile(r"verification\s+passed|colors used\s*:\s*avg", re.IGNORECASE)


def parse(out: str) -> dict | None:
    egc      = EGC_RE.search(out)
    avg_t    = AVG_TOTAL_RE.search(out)
    avg_c    = AVG_COLOR_RE.search(out)
    avg_iter = AVG_ITER_RE.search(out)
    min_t    = MIN_TOTAL_RE.search(out)
    runs     = RUN_RE.findall(out)
    if not (egc and avg_t and avg_c and runs):
        return None
    return {
        "theta":        int(egc.group(1)),
        "avg_color":    float(avg_c.group(1)),
        "avg_total_ms": float(avg_t.group(1)),
        "min_total_ms": float(min_t.group(1)) if min_t else None,
        "avg_iter":     float(avg_iter.group(1)) if avg_iter else None,
        "n_runs":       len(runs),
    }


def run_one(binary: str, graph: Path, mode: str, runs: int, timeout: int):
    cmd = [binary, "-f", str(graph), "-a", "cuSL_ELS_SDC_CTA_S", "--runs", str(runs)]
    if mode == "baseline":
        cmd.extend(["-e", "0"])
    elif mode == "v2":
        cmd.extend(["--predict", "--v2-model"])
    else:
        raise ValueError(mode)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return parse(proc.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="/home/chsieh45/PunchShadow/CHROMA/.claude/worktrees/theta-predictor-v2/CHROMA/CHROMA")
    ap.add_argument("--egr-dir", default="/home/chsieh45/PunchShadow/CHROMA/Datasets/EGR")
    ap.add_argument("--runs", type=int, default=5,
                    help="how many CHROMA --runs to average per (graph, mode) measurement")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    egr_dir = Path(args.egr_dir)
    graphs = sorted(egr_dir.glob("*.egr"))
    print(f"# binary={args.binary}", file=sys.stderr)
    print(f"# graphs={len(graphs)}, runs={args.runs}, algo=cuSL_ELS_SDC_CTA_S",
          file=sys.stderr)

    rows = []
    for g in graphs:
        b = run_one(args.binary, g, "baseline", args.runs, args.timeout)
        v = run_one(args.binary, g, "v2",       args.runs, args.timeout)
        rows.append({"graph": g.name, "baseline": b, "v2": v})
        bt = "?" if b is None else f"{b['avg_total_ms']:8.2f}ms"
        bc = "?" if b is None else f"{b['avg_color']:5.1f}c"
        vt = "?" if v is None else f"{v['avg_total_ms']:8.2f}ms"
        vc = "?" if v is None else f"{v['avg_color']:5.1f}c"
        vth = "?" if v is None else v["theta"]
        speedup = "?" if not (b and v and v['avg_total_ms']>0) else f"{b['avg_total_ms']/v['avg_total_ms']:.2f}×"
        print(f"  {g.name:38s}  base θ=0 {bt} {bc}  |  v2 θ={vth:>2} {vt} {vc}  | spd {speedup}",
              file=sys.stderr)

    # Per-mode summaries
    def collect(mode):
        ok = [r[mode] for r in rows if r[mode] is not None]
        return ok

    base = collect("baseline")
    v2   = collect("v2")
    print("", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    if base and v2:
        # Pair-wise: only graphs where BOTH modes ran
        pairs = [(r["baseline"], r["v2"]) for r in rows if r["baseline"] and r["v2"]]
        if pairs:
            base_t = [b["avg_total_ms"] for b, _ in pairs]
            v2_t   = [v["avg_total_ms"] for _, v in pairs]
            base_c = [b["avg_color"]    for b, _ in pairs]
            v2_c   = [v["avg_color"]    for _, v in pairs]
            speedups = [bt / vt if vt > 0 else 0 for bt, vt in zip(base_t, v2_t)]
            color_diffs = [vc - bc for vc, bc in zip(v2_c, base_c)]
            wins  = sum(1 for s in speedups if s > 1.0)
            ties  = sum(1 for s in speedups if 0.95 <= s <= 1.05)
            losses = sum(1 for s in speedups if s < 0.95)

            print(f"Pairs: {len(pairs)}  v2 wins (≥1.0× speedup): {wins}  "
                  f"ties (0.95-1.05×): {ties}  losses: {losses}", file=sys.stderr)
            print(f"Geomean v2 speedup: {(prod(speedups) ** (1/len(speedups))):.3f}×",
                  file=sys.stderr) if speedups else None
            print(f"Mean v2 speedup:    {sum(speedups)/len(speedups):.3f}×",
                  file=sys.stderr)
            print(f"Mean Δ colors (v2 − baseline): {sum(color_diffs)/len(color_diffs):+.2f}",
                  file=sys.stderr)
            print(f"Worst v2 color regression:    {max(color_diffs):+.0f}",
                  file=sys.stderr)
            print(f"Best  v2 color improvement:   {min(color_diffs):+.0f}",
                  file=sys.stderr)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"# wrote {args.out}", file=sys.stderr)


def prod(xs):
    r = 1.0
    for x in xs: r *= x
    return r


if __name__ == "__main__":
    main()

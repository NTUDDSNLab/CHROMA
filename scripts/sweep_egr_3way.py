#!/usr/bin/env python3
"""3-way sweep on Datasets/EGR/*.egr comparing:
  baseline : --elastic 0                                   (no EGC)
  v1       : --predict   on main-repo CHROMA (legacy 2-feat model.cpp + round)
  v2       : --predict --v2-model on worktree CHROMA       (RF + floor + shift=-0.5)

Reports per-graph predicted θ, color count, total runtime; aggregate stats.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

EGC_RE       = re.compile(r"EGC θ:\s*(-?\d+)", re.IGNORECASE)
AVG_TOTAL_RE = re.compile(r"^\s*Total\s+time\s*:\s*avg=\s*([0-9.]+)", re.IGNORECASE | re.MULTILINE)
AVG_COLOR_RE = re.compile(r"^\s*colors\s+used\s*:\s*avg=\s*([0-9.]+)",  re.IGNORECASE | re.MULTILINE)
RUN_RE       = re.compile(r"\[Run \d+/\d+\][^\n]*colors:\s*(\d+)\s+iters:\s*(\d+)")


def parse(out):
    egc   = EGC_RE.search(out)
    avg_t = AVG_TOTAL_RE.search(out)
    avg_c = AVG_COLOR_RE.search(out)
    runs  = RUN_RE.findall(out)
    if not (egc and avg_t and avg_c and runs):
        return None
    return {"theta": int(egc.group(1)),
            "avg_color": float(avg_c.group(1)),
            "avg_total_ms": float(avg_t.group(1)),
            "n_runs": len(runs)}


def run_one(binary, graph, mode, runs, timeout):
    cmd = [binary, "-f", str(graph), "-a", "cuSL_ELS_SDC_CTA_S", "--runs", str(runs)]
    if mode == "baseline":
        cmd.extend(["-e", "0"])
    elif mode == "v1":
        cmd.append("--predict")
    elif mode == "v2":
        cmd.extend(["--predict", "--v2-model"])
    else:
        raise ValueError(mode)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return parse(proc.stdout)


def gmean(xs):
    if not xs: return 0.0
    p = 1.0
    for x in xs: p *= x
    return p ** (1.0 / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree-binary",
                    default="/home/chsieh45/PunchShadow/CHROMA/.claude/worktrees/theta-predictor-v2/CHROMA/CHROMA",
                    help="binary built with V2_MODEL=1 (provides baseline + v2)")
    ap.add_argument("--main-binary",
                    default="/home/chsieh45/PunchShadow/CHROMA/CHROMA/CHROMA",
                    help="binary built with V2_MODEL=0 (provides legacy v1 --predict)")
    ap.add_argument("--egr-dir", default="/home/chsieh45/PunchShadow/CHROMA/Datasets/EGR")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    graphs = sorted(Path(args.egr_dir).glob("*.egr"))
    print(f"# {len(graphs)} graphs, runs={args.runs}, algo=cuSL_ELS_SDC_CTA_S",
          file=sys.stderr)
    print(f"# v1 binary: {args.main_binary}", file=sys.stderr)
    print(f"# v2 binary: {args.worktree_binary}", file=sys.stderr)

    rows = []
    for g in graphs:
        b  = run_one(args.worktree_binary, g, "baseline", args.runs, args.timeout)
        v1 = run_one(args.main_binary,     g, "v1",       args.runs, args.timeout)
        v2 = run_one(args.worktree_binary, g, "v2",       args.runs, args.timeout)
        rows.append({"graph": g.name, "baseline": b, "v1": v1, "v2": v2})

        def fmt(rec, label):
            if rec is None: return f"{label} ?"
            return f"{label} θ={rec['theta']:>2} {rec['avg_total_ms']:8.2f}ms {rec['avg_color']:5.1f}c"

        line = f"  {g.name:32s}  " + " | ".join([fmt(b, "base"), fmt(v1, "v1"), fmt(v2, "v2")])
        if b and v1 and v2:
            line += f"  | spd v1={b['avg_total_ms']/v1['avg_total_ms']:5.2f}× v2={b['avg_total_ms']/v2['avg_total_ms']:5.2f}×"
        print(line, file=sys.stderr)

    # Pairwise stats (graphs where all 3 modes ran)
    full = [r for r in rows if r["baseline"] and r["v1"] and r["v2"]]
    print("", file=sys.stderr)
    print("=" * 100, file=sys.stderr)
    print(f"Full triples: {len(full)} / {len(rows)}", file=sys.stderr)
    if full:
        bt = [r["baseline"]["avg_total_ms"] for r in full]
        v1t = [r["v1"]["avg_total_ms"]      for r in full]
        v2t = [r["v2"]["avg_total_ms"]      for r in full]
        bc  = [r["baseline"]["avg_color"]   for r in full]
        v1c = [r["v1"]["avg_color"]         for r in full]
        v2c = [r["v2"]["avg_color"]         for r in full]
        v1_spd = [a/b if b > 0 else 0 for a, b in zip(bt, v1t)]
        v2_spd = [a/b if b > 0 else 0 for a, b in zip(bt, v2t)]
        v2_vs_v1 = [a/b if b > 0 else 0 for a, b in zip(v1t, v2t)]
        print()
        print(f"{'metric':30s}  {'v1 (legacy)':>13s}  {'v2 (RF+f-0.5)':>15s}  {'Δ (v2 − v1)':>13s}", file=sys.stderr)
        print(f"{'mean speedup vs baseline':30s}  {sum(v1_spd)/len(v1_spd):>12.2f}×  "
              f"{sum(v2_spd)/len(v2_spd):>14.2f}×  {(sum(v2_spd)-sum(v1_spd))/len(v1_spd):>+12.2f}×",
              file=sys.stderr)
        print(f"{'geomean speedup vs baseline':30s}  {gmean(v1_spd):>12.2f}×  "
              f"{gmean(v2_spd):>14.2f}×  {gmean(v2_spd)-gmean(v1_spd):>+12.2f}×",
              file=sys.stderr)
        print(f"{'mean Δ colors vs baseline':30s}  {sum(v1c)/len(v1c)-sum(bc)/len(bc):>+12.2f}   "
              f"{sum(v2c)/len(v2c)-sum(bc)/len(bc):>+13.2f}   "
              f"{(sum(v2c)-sum(v1c))/len(v1c):>+12.2f}",
              file=sys.stderr)
        print(f"{'mean predicted θ':30s}  {sum(r['v1']['theta'] for r in full)/len(full):>13.2f}  "
              f"{sum(r['v2']['theta'] for r in full)/len(full):>15.2f}  "
              f"{sum(r['v2']['theta']-r['v1']['theta'] for r in full)/len(full):>+13.2f}",
              file=sys.stderr)
        v1_wins = sum(1 for s in v1_spd if s > 1.05)
        v2_wins = sum(1 for s in v2_spd if s > 1.05)
        v2_beats_v1 = sum(1 for s in v2_vs_v1 if s > 1.05)
        v1_beats_v2 = sum(1 for s in v2_vs_v1 if s < 0.95)
        print()
        print(f"v1 wins vs baseline (≥1.05×): {v1_wins}/{len(full)}", file=sys.stderr)
        print(f"v2 wins vs baseline (≥1.05×): {v2_wins}/{len(full)}", file=sys.stderr)
        print(f"v2 beats v1 (≥1.05× faster):  {v2_beats_v1}/{len(full)}", file=sys.stderr)
        print(f"v1 beats v2 (≥1.05× faster):  {v1_beats_v2}/{len(full)}", file=sys.stderr)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"# wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

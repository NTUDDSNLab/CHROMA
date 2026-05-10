#!/usr/bin/env python3
"""4-way sweep on Datasets/EGR/*.egr comparing:
  baseline    : --elastic 0
  v1          : --predict on main-repo CHROMA (legacy 3-feat, no policy)
  v2_raw      : --predict --v2-model on CHROMA built with USE_FLOOR=false, SCORE_SHIFT=0
  v2_deployed : --predict --v2-model on CHROMA built with USE_FLOOR=true,  SCORE_SHIFT=-0.5
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

EGC_RE       = re.compile(r"EGC θ:\s*(-?\d+)", re.IGNORECASE)
AVG_TOTAL_RE = re.compile(r"^\s*Total\s+time\s*:\s*avg=\s*([0-9.]+)", re.IGNORECASE | re.MULTILINE)
AVG_COLOR_RE = re.compile(r"^\s*colors\s+used\s*:\s*avg=\s*([0-9.]+)",  re.IGNORECASE | re.MULTILINE)
RUN_RE       = re.compile(r"\[Run \d+/\d+\][^\n]*colors:\s*(\d+)\s+iters:\s*(\d+)")


def parse(out):
    egc, avg_t, avg_c, runs = (EGC_RE.search(out), AVG_TOTAL_RE.search(out),
                                AVG_COLOR_RE.search(out), RUN_RE.findall(out))
    if not (egc and avg_t and avg_c and runs):
        return None
    return {"theta": int(egc.group(1)),
            "avg_color": float(avg_c.group(1)),
            "avg_total_ms": float(avg_t.group(1))}


def run_one(binary, graph, mode, runs, timeout):
    cmd = [binary, "-f", str(graph), "-a", "cuSL_ELS_SDC_CTA_S", "--runs", str(runs)]
    if mode == "baseline":   cmd.extend(["-e", "0"])
    elif mode == "v1":       cmd.append("--predict")
    elif mode in ("v2_raw", "v2_deployed"):
                              cmd.extend(["--predict", "--v2-model"])
    try:
        return parse(subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout)
    except subprocess.TimeoutExpired:
        return None


def gmean(xs):
    if not xs: return 0.0
    p = 1.0
    for x in xs: p *= x
    return p ** (1.0 / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree-binary",
                    default="/home/chsieh45/PunchShadow/CHROMA/.claude/worktrees/theta-predictor-v2/CHROMA/CHROMA")
    ap.add_argument("--main-binary",      default="/home/chsieh45/PunchShadow/CHROMA/CHROMA/CHROMA")
    ap.add_argument("--v2-raw-binary",    default="/tmp/CHROMA_v2_raw")
    ap.add_argument("--v2-dep-binary",    default="/tmp/CHROMA_v2_deployed")
    ap.add_argument("--egr-dir", default="/home/chsieh45/PunchShadow/CHROMA/Datasets/EGR")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    graphs = sorted(Path(args.egr_dir).glob("*.egr"))
    print(f"# {len(graphs)} graphs, runs={args.runs}, algo=cuSL_ELS_SDC_CTA_S", file=sys.stderr)

    rows = []
    for g in graphs:
        b   = run_one(args.worktree_binary, g, "baseline",    args.runs, args.timeout)
        v1  = run_one(args.main_binary,     g, "v1",          args.runs, args.timeout)
        vR  = run_one(args.v2_raw_binary,   g, "v2_raw",      args.runs, args.timeout)
        vD  = run_one(args.v2_dep_binary,   g, "v2_deployed", args.runs, args.timeout)
        rows.append({"graph": g.name, "baseline": b, "v1": v1, "v2_raw": vR, "v2_deployed": vD})

        def fmt(rec, label):
            if rec is None: return f"{label} ?"
            return f"{label} θ={rec['theta']:>2} {rec['avg_total_ms']:8.2f}ms {rec['avg_color']:5.1f}c"

        line = f"  {g.name:32s}  " + " | ".join([fmt(b, "base"), fmt(v1, "v1"),
                                                  fmt(vR, "raw"), fmt(vD, "dep")])
        if all([b, v1, vR, vD]):
            line += (f"  | spd v1={b['avg_total_ms']/v1['avg_total_ms']:5.2f}× "
                     f"raw={b['avg_total_ms']/vR['avg_total_ms']:5.2f}× "
                     f"dep={b['avg_total_ms']/vD['avg_total_ms']:5.2f}×")
        print(line, file=sys.stderr)

    full = [r for r in rows if all(r[k] for k in ("baseline","v1","v2_raw","v2_deployed"))]
    print("", file=sys.stderr)
    print("=" * 110, file=sys.stderr)
    print(f"Full quadruples: {len(full)} / {len(rows)}\n", file=sys.stderr)

    if full:
        bt  = [r["baseline"]["avg_total_ms"] for r in full]
        bc  = [r["baseline"]["avg_color"]    for r in full]
        cols = []
        for k in ("v1", "v2_raw", "v2_deployed"):
            t = [r[k]["avg_total_ms"] for r in full]
            c = [r[k]["avg_color"]    for r in full]
            th = [r[k]["theta"]       for r in full]
            spd = [bb/tt if tt > 0 else 0 for bb, tt in zip(bt, t)]
            cols.append({
                "name":           k,
                "mean_speedup":   sum(spd)/len(spd),
                "geomean_speed":  gmean(spd),
                "mean_dcolors":   sum(c)/len(c) - sum(bc)/len(bc),
                "mean_theta":     sum(th)/len(th),
                "wins_vs_base":   sum(1 for s in spd if s > 1.05),
                "ties_vs_base":   sum(1 for s in spd if 0.95 <= s <= 1.05),
                "losses_vs_base": sum(1 for s in spd if s < 0.95),
            })

        hdr_w = 14
        print(f"{'metric':30s}  " + "  ".join(f"{c['name']:>{hdr_w}s}" for c in cols), file=sys.stderr)
        def row(label, key, formatter):
            cells = "  ".join(f"{formatter(c[key]):>{hdr_w}s}" for c in cols)
            print(f"{label:30s}  {cells}", file=sys.stderr)
        row("mean speedup",      "mean_speedup",   lambda v: f"{v:.2f}×")
        row("geomean speedup",   "geomean_speed",  lambda v: f"{v:.2f}×")
        row("mean Δ colors",     "mean_dcolors",   lambda v: f"{v:+.2f}")
        row("mean predicted θ",  "mean_theta",     lambda v: f"{v:.2f}")
        row("wins vs baseline",  "wins_vs_base",   lambda v: f"{v}")
        row("ties vs baseline",  "ties_vs_base",   lambda v: f"{v}")
        row("losses vs baseline","losses_vs_base", lambda v: f"{v}")

        # Pairwise: v2_raw vs v1, v2_dep vs v1
        print("", file=sys.stderr)
        v1_t = [r["v1"]["avg_total_ms"]         for r in full]
        vR_t = [r["v2_raw"]["avg_total_ms"]     for r in full]
        vD_t = [r["v2_deployed"]["avg_total_ms"] for r in full]
        spd_R_v1 = [a/b for a, b in zip(v1_t, vR_t)]
        spd_D_v1 = [a/b for a, b in zip(v1_t, vD_t)]
        print(f"v2_raw      beats v1 (≥1.05× faster):  {sum(1 for s in spd_R_v1 if s > 1.05):>2}/{len(full)}", file=sys.stderr)
        print(f"v1 beats v2_raw      (≥1.05× faster):  {sum(1 for s in spd_R_v1 if s < 0.95):>2}/{len(full)}", file=sys.stderr)
        print(f"v2_deployed beats v1 (≥1.05× faster):  {sum(1 for s in spd_D_v1 if s > 1.05):>2}/{len(full)}", file=sys.stderr)
        print(f"v1 beats v2_deployed (≥1.05× faster):  {sum(1 for s in spd_D_v1 if s < 0.95):>2}/{len(full)}", file=sys.stderr)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\n# wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

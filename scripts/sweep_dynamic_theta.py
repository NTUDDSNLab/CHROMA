#!/usr/bin/env python3
"""Three-way EGR sweep comparing dynamic θ controller vs static v3.

  static_v3      : --predict
  dyn_only       : --dynamic-theta            (θ_initial = 0)
  static + dyn   : --predict --dynamic-theta  (θ_initial from v3 RF)

Per-graph: total time, color count, θ_initial, θ_final, n_bumps.
Aggregate (mean / geomean speedup, mean Δ colors, wins) on:
  - EGR overlap-with-train (11)
  - EGR holdout            ( 8)
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path

EGC_RE       = re.compile(r"EGC θ:\s*(-?\d+)", re.IGNORECASE)
AVG_TOTAL_RE = re.compile(r"^\s*Total\s+time\s*:\s*avg=\s*([0-9.]+)", re.IGNORECASE | re.MULTILINE)
AVG_COLOR_RE = re.compile(r"^\s*colors\s+used\s*:\s*avg=\s*([0-9.]+)",  re.IGNORECASE | re.MULTILINE)
RUN_RE       = re.compile(r"\[Run \d+/\d+\][^\n]*colors:\s*(\d+)\s+iters:\s*(\d+)")
TRAJ_RE      = re.compile(r"θ trajectory: start=(-?\d+)\s+bumps=\[(.*?)\]\s+total=(\d+)")

def parse(out):
    egc, t, c, runs = (EGC_RE.search(out), AVG_TOTAL_RE.search(out),
                        AVG_COLOR_RE.search(out), RUN_RE.findall(out))
    if not (egc and t and c and runs):
        return None
    rec = {"theta_initial": int(egc.group(1)),
           "avg_color":     float(c.group(1)),
           "avg_total_ms":  float(t.group(1)),
           "n_runs":        len(runs)}
    traj = TRAJ_RE.search(out)
    if traj:
        rec["theta_initial_logged"] = int(traj.group(1))
        rec["n_bumps"]              = int(traj.group(3))
        bumps = traj.group(2).strip()
        if bumps:
            last = bumps.split(",")[-1]
            m = re.search(r"θ=(\d+)", last)
            if m: rec["theta_final"] = int(m.group(1))
        else:
            rec["theta_final"] = rec["theta_initial_logged"]
    return rec


def run_one(binary, graph, mode, runs, timeout, dyn_K, dyn_rate, dyn_step, dyn_cap):
    cmd = [binary, "-f", str(graph), "-a", "cuSL_ELS_SDC_CTA_S", "--runs", str(runs)]
    if mode == "static_v3":
        cmd.append("--predict")
    elif mode == "dyn_only":
        cmd.extend(["--dynamic-theta",
                    "--dynamic-K", str(dyn_K),
                    "--dynamic-rate", str(dyn_rate),
                    "--dynamic-step", str(dyn_step)])
        if dyn_cap > 0:
            cmd.extend(["--dynamic-cap", str(dyn_cap)])
    elif mode == "static_dyn":
        cmd.extend(["--predict", "--dynamic-theta",
                    "--dynamic-K", str(dyn_K),
                    "--dynamic-rate", str(dyn_rate),
                    "--dynamic-step", str(dyn_step)])
        if dyn_cap > 0:
            cmd.extend(["--dynamic-cap", str(dyn_cap)])
    else:
        raise ValueError(mode)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return parse(proc.stdout)
    except subprocess.TimeoutExpired:
        return None


def gmean(xs):
    if not xs: return 0.0
    p = 1.0
    for x in xs: p *= x
    return p ** (1.0 / len(xs))


OVERLAP = {"Email-Enron.col.egr","Slashdot0811.egr","Slashdot0902.egr","Stanford.egr",
           "as-skitter.egr","cit-Patents.egr","delaunay_n24.egr","soc-Epinions1.col.egr",
           "wiki-Talk.col.egr","wiki-Vote.col.egr","youtube.egr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="CHROMA/CHROMA")
    ap.add_argument("--egr-dir", default="/home/chsieh45/PunchShadow/CHROMA/Datasets/EGR")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dyn-K", type=int, default=10)
    ap.add_argument("--dyn-rate", type=float, default=0.005)
    ap.add_argument("--dyn-step", type=int, default=1)
    ap.add_argument("--dyn-cap", type=int, default=0)
    ap.add_argument("--out", default="model/v2_data/egr_dynamic_theta.json")
    args = ap.parse_args()

    graphs = sorted(Path(args.egr_dir).glob("*.egr"))
    print(f"# {len(graphs)} graphs, runs={args.runs}, K={args.dyn_K}, "
          f"rate={args.dyn_rate}, step={args.dyn_step}, cap={args.dyn_cap or 'auto'}",
          file=sys.stderr)

    rows = []
    for g in graphs:
        rec = {"graph": g.name, "in_train_overlap": g.name in OVERLAP}
        baseline_cmd = [args.binary, "-f", str(g), "-a", "cuSL_ELS_SDC_CTA_S",
                         "--runs", str(args.runs), "-e", "0"]
        try:
            rec["baseline"] = parse(subprocess.run(baseline_cmd, capture_output=True,
                                                    text=True, timeout=args.timeout).stdout)
        except subprocess.TimeoutExpired:
            rec["baseline"] = None
        for m in ("static_v3", "dyn_only", "static_dyn"):
            rec[m] = run_one(args.binary, g, m, args.runs, args.timeout,
                              args.dyn_K, args.dyn_rate, args.dyn_step, args.dyn_cap)

        def fmt(r):
            if r is None: return "?"
            extras = ""
            if "n_bumps" in r:
                extras = f"  bumps={r['n_bumps']:>2}  θf={r.get('theta_final', '?')}"
            return f"θ={r['theta_initial']:>2} {r['avg_total_ms']:7.1f}ms {r['avg_color']:5.1f}c{extras}"

        flag = "⚠trained" if rec["in_train_overlap"] else "·holdout"
        print(f"{g.name:32s} {flag} | base {fmt(rec['baseline']):26s} | "
              f"sv3 {fmt(rec['static_v3']):26s} | "
              f"dyn {fmt(rec['dyn_only']):40s} | "
              f"sv3+dyn {fmt(rec['static_dyn']):40s}",
              file=sys.stderr)
        rows.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {args.out}", file=sys.stderr)

    # Aggregate
    modes = ["static_v3", "dyn_only", "static_dyn"]
    for label, subset in [("ALL 19", rows),
                           ("OVERLAP 11", [r for r in rows if r["in_train_overlap"]]),
                           ("HOLDOUT 8", [r for r in rows if not r["in_train_overlap"]])]:
        full = [r for r in subset if r["baseline"] and all(r[m] for m in modes)]
        if not full:
            continue
        bt = [r["baseline"]["avg_total_ms"] for r in full]
        bc = [r["baseline"]["avg_color"]    for r in full]
        print(f"\n=== {label} ({len(full)}) ===", file=sys.stderr)
        print(f'{"metric":24s}  ' + "  ".join(f"{m:>14s}" for m in modes), file=sys.stderr)
        for label2, key in [("mean speedup", "spd"), ("geomean speedup", "gspd"),
                             ("mean Δ colors", "dc"), ("mean predicted θ", "th"),
                             ("wins vs base", "wins"), ("graphs ramped", "rampn")]:
            cells = []
            for m in modes:
                ts = [r[m]["avg_total_ms"] for r in full]
                cs = [r[m]["avg_color"]    for r in full]
                ths = [r[m]["theta_initial"] for r in full]
                spd = [b/x if x > 0 else 0 for b, x in zip(bt, ts)]
                rampn = sum(1 for r in full if r[m].get("n_bumps", 0) > 0)
                v = {"spd": sum(spd)/len(spd),
                     "gspd": gmean(spd),
                     "dc": sum(cs)/len(cs) - sum(bc)/len(bc),
                     "th": sum(ths)/len(ths),
                     "wins": sum(1 for s in spd if s > 1.05),
                     "rampn": rampn}[key]
                if key in ("spd","gspd"): cells.append(f"{v:.2f}×")
                elif key == "dc":         cells.append(f"{v:+.2f}")
                elif key == "th":         cells.append(f"{v:.2f}")
                elif key == "wins":       cells.append(f"{v}/{len(full)}")
                elif key == "rampn":      cells.append(f"{v}/{len(full)}")
            print(f"{label2:24s}  " + "  ".join(f"{c:>14s}" for c in cells),
                  file=sys.stderr)


if __name__ == "__main__":
    main()

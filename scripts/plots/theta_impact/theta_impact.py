#!/usr/bin/env python3
"""Sweep θ-impact for the paper Fig. 6 redraw.

For each dataset (default: as-skitter, cit-Patents, europe_osm):
  - θ = 0..THETA_MAX: run `CHROMA -a cuSL_ELS_SDC -e <θ>` RUNS times,
    keep the best run (min colors, tie -> min runtime); record
    {color, runtime_ms, iter_count}.
  - CEP θ: one run  CHROMA -a cuSL_ELS_SDC --predict --predict-model v0_paper
  - AEP θ: one run  CHROMA -a cuSL_ELS_SDC --predict --predict-model <M>
                           --no-dynamic-theta   (<M> = --predict-model, default v3)
Writes theta_impact_results.json consumed by plot_theta_impact.py.

The `EGC θ: N (Predicted)` line reports the predictor's *initial* θ
(before any online bumping), so the parsed value is independent of
whether the dynamic-θ controller is on; CEP follows the paper's
v0_paper path, AEP adds --no-dynamic-theta per the v3_raw definition.

Examples:
    python3 scripts/plots/theta_impact/theta_impact.py
    python3 scripts/plots/theta_impact/theta_impact.py \\
        --datasets cit-Patents --theta-max 3 --runs 2 --out /tmp/ti.json
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

# Patterns for CHROMA verbose single-run output. CHROMA prints phase
# timings ("PA runtime:", "CA runtime:", ...) BEFORE the "Total
# runtime:" line, so RUNTIME_RE is anchored to "Total runtime:" — a
# bare "runtime:" (as grid_elastic.py uses) would match PA time first.
# COLORS_RE / ITER_RE mirror grid_elastic.py; PRED_RE matches
# "EGC θ: <d> (Predicted)".
RUNTIME_RE = re.compile(r"Total\s+runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
COLORS_RE = re.compile(r"colors\s+used:\s*(\d+)", re.IGNORECASE)
ITER_RE = re.compile(r"Iter\s+count:\s*(\d+)", re.IGNORECASE)
PRED_RE = re.compile(r"EGC[^:]*:\s*(\d+)\s*\(Predicted\)")


def read_egr_header(path: Path) -> tuple:
    """Return (nodes, edges) from the .egr CSR binary header."""
    with open(path, "rb") as f:
        nodes, edges = struct.unpack("<ii", f.read(8))
    return nodes, edges


def resolve_egr(ds_dir: Path, stem: str) -> Optional[Path]:
    """Stem -> .egr, .col double-suffix aware (mirrors run_pa_sweep.py)."""
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
    return {
        "runtime_ms": float(mr.group(1)),
        "color": int(mc.group(1)),
        "iter_count": int(mi.group(1)) if mi else None,
    }


def best_of(runs: list) -> Optional[dict]:
    """grid_elastic keep-best: min colors, tie -> min runtime."""
    if not runs:
        return None
    return sorted(runs, key=lambda x: (x["color"], x["runtime_ms"]))[0]


def predicted_theta(binary, egr, algo, model, no_bump, timeout) -> tuple:
    cmd = [binary, "-f", str(egr), "-a", algo, "--predict",
           "--predict-model", model]
    if no_bump:
        cmd.append("--no-dynamic-theta")
    r, _dt, err = run(cmd, timeout)
    if err:
        return None, err
    m = PRED_RE.search(r.stdout)
    if not m:
        return None, "no '(Predicted)' line in output"
    return int(m.group(1)), None


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--binary", default=str(repo / "CHROMA" / "CHROMA"))
    ap.add_argument("--dataset-dir", default=str(repo / "Datasets" / "EGR"))
    ap.add_argument("--datasets", nargs="+",
                    default=["as-skitter", "cit-Patents", "europe_osm"])
    ap.add_argument("--algo", default="cuSL_ELS_SDC")
    ap.add_argument("--predict-model", default="v3",
                    help="Model for the AEP θ run (e.g. v3, skew); "
                         "CEP stays pinned to v0_paper.")
    ap.add_argument("--theta-max", type=int, default=20)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1200,
                    help="Per-CHROMA-invocation timeout (seconds)")
    ap.add_argument("--out", default=str(here / "theta_impact_results.json"))
    args = ap.parse_args()

    if not os.access(args.binary, os.X_OK):
        sys.exit(f"ERROR: CHROMA binary not found or not executable at "
                 f"{args.binary} (build CHROMA/ with PRE_MODEL=1 for --predict)")
    ds_dir = Path(args.dataset_dir)

    data = {}
    for stem in args.datasets:
        egr = resolve_egr(ds_dir, stem)
        if egr is None:
            print(f"# {stem:16s} SKIP: no .egr in {ds_dir}", file=sys.stderr)
            data[stem] = {"error": f"no .egr for {stem} in {ds_dir}"}
            continue
        nodes, edges = read_egr_header(egr)
        entry = {"nodes": nodes, "edges": edges, "sweep": {}}
        for theta in range(0, args.theta_max + 1):
            recs = []
            last_err = None
            for _ in range(args.runs):
                r, _dt, err = run(
                    [args.binary, "-f", str(egr), "-a", args.algo,
                     "-e", str(theta)], args.timeout)
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
                entry["sweep"][str(theta)] = {
                    "error": last_err or "all runs failed"}
                print(f"# {stem:16s} θ={theta:2d} FAIL ({last_err})",
                      file=sys.stderr)
            else:
                entry["sweep"][str(theta)] = best
                print(f"# {stem:16s} θ={theta:2d} colors={best['color']:3d} "
                      f"rt={best['runtime_ms']:9.3f}ms "
                      f"iter={best['iter_count']}", file=sys.stderr)
        cep, cep_err = predicted_theta(args.binary, egr, args.algo,
                                       "v0_paper", False, args.timeout)
        aep, aep_err = predicted_theta(args.binary, egr, args.algo,
                                       args.predict_model, True, args.timeout)
        entry["cep_theta"] = cep
        entry["aep_theta"] = aep
        print(f"# {stem:16s} CEP θ={cep} ({cep_err or 'ok'}) | "
              f"AEP θ={aep} ({aep_err or 'ok'})", file=sys.stderr)
        data[stem] = entry

    out = {
        "datasets": args.datasets,
        "algo": args.algo,
        "theta_max": args.theta_max,
        "runs": args.runs,
        "predict_model": args.predict_model,
        "data": data,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Batch runner for CHROMA.

Runs the CHROMA binary across multiple datasets N times (default 5),
selects the run with the smallest color usage (ties broken by fastest runtime),
and writes a JSON summary including configuration and per-dataset results.

Examples:
  python3 batch_test.py \
    --dataset-dir Datasets \
    --pattern "*.egr" \
    --binary CHROMA/CHROMA \
    --algorithm P_SL_ELS \
    --elastic 10 \
    --runs 5 \
    --out results_chroma.json

  # Use predict model (requires building CHROMA with make PRE_MODEL=1)
  python3 batch_test.py \
    --dataset-dir Datasets \
    --pattern "*.egr" \
    --binary CHROMA/CHROMA \
    --algorithm P_SL_ELS \
    --predict \
    --runs 5 \
    --out results_chroma_pred.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import csv


RUNTIME_RE = re.compile(r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
COLORS_RE = re.compile(r"colors\s+used:\s*(\d+)", re.IGNORECASE)
NODES_RE = re.compile(r"^Nodes:\s*(\d+)$", re.IGNORECASE | re.MULTILINE)
EDGES_RE = re.compile(r"^Edges:\s*(\d+)$", re.IGNORECASE | re.MULTILINE)
THETA_RE = re.compile(r"^EGC\s*θ:\s*(\d+)(?:.*)$", re.IGNORECASE | re.MULTILINE)


ALGO_MAPPING = {
    "0": "cuSL_ELS",
    "1": "cuSL_ELS_SDC",
    "cuSL_ELS": "cuSL_ELS",
    "cuSL_ELS_SDC": "cuSL_ELS_SDC"
}


@dataclass
class RunResult:
    runtime_ms: float
    colors_used: int
    nodes: Optional[int] = None
    edges: Optional[int] = None
    theta: Optional[int] = None
    theta_predicted: Optional[bool] = None
    ok: bool = True
    error: Optional[str] = None
    stdout: Optional[str] = None


def parse_output(stdout: str) -> RunResult:
    runtime_ms: Optional[float] = None
    colors_used: Optional[int] = None
    theta: Optional[int] = None
    theta_predicted: Optional[bool] = None

    m = RUNTIME_RE.search(stdout)
    if m:
        runtime_ms = float(m.group(1))

    m = COLORS_RE.search(stdout)
    if m:
        colors_used = int(m.group(1))

    nodes = None
    edges = None
    m = NODES_RE.search(stdout)
    if m:
        nodes = int(m.group(1))
    m = EDGES_RE.search(stdout)
    if m:
        edges = int(m.group(1))

    # Parse theta (EGC θ) and whether it was predicted
    m = THETA_RE.search(stdout)
    if m:
        try:
            theta = int(m.group(1))
        except ValueError:
            theta = None
        # If line contains '(Predicted)' mark as predicted
        # Case-insensitive check on the matched line span
        span_start, span_end = m.span(0)
        line_text = stdout[span_start:span_end]
        theta_predicted = ("predicted" in line_text.lower())

    if runtime_ms is None or colors_used is None:
        return RunResult(
            runtime_ms=0.0,
            colors_used=-1,
            nodes=nodes,
            edges=edges,
            theta=theta,
            theta_predicted=theta_predicted,
            ok=False,
            error="Failed to parse runtime and/or colors",
            stdout=stdout,
        )

    return RunResult(
        runtime_ms=runtime_ms,
        colors_used=colors_used,
        nodes=nodes,
        edges=edges,
        theta=theta,
        theta_predicted=theta_predicted,
        ok=True,
        stdout=stdout,
    )


def run_chroma(binary: str, dataset: str, algorithm: str, elastic: int, predict: bool, extra_args: List[str],
               timeout_sec: Optional[int]) -> RunResult:
    # Build command; if predict is enabled, omit -r and add --predict
    if predict:
        cmd = [binary, "-f", dataset, "-a", algorithm, "--predict"]
    else:
        cmd = [binary, "-f", dataset, "-a", algorithm, "-e", str(elastic)]
    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            runtime_ms=0.0,
            colors_used=-1,
            ok=False,
            error=f"Timeout after {timeout_sec}s",
        )
    except FileNotFoundError:
        return RunResult(
            runtime_ms=0.0,
            colors_used=-1,
            ok=False,
            error=f"Binary not found: {binary}",
        )

    res = parse_output(proc.stdout)
    if proc.returncode != 0:
        res.ok = False
        res.error = f"Exit code {proc.returncode}"
    return res


def pick_best(results: List[RunResult]) -> Optional[RunResult]:
    valid = [r for r in results if r.ok]
    if not valid:
        return None
    # Smallest colors first, then fastest runtime
    return sorted(valid, key=lambda r: (r.colors_used, r.runtime_ms))[0]


def discover_datasets(dataset_dir: str, pattern: str, recursive: bool) -> List[str]:
    base = os.path.abspath(dataset_dir)
    if recursive:
        patt = os.path.join(base, "**", pattern)
        files = glob.glob(patt, recursive=True)
    else:
        patt = os.path.join(base, pattern)
        files = glob.glob(patt)
    files = sorted(f for f in files if os.path.isfile(f))
    return files


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Batch test runner for CHROMA")
    ap.add_argument("--dataset-dir", "-d", default="Datasets",
                    help="Directory containing datasets (default: Datasets)")
    ap.add_argument("--pattern", default="*.egr",
                    help="Glob pattern to match dataset files (default: *.egr)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories while matching datasets")
    ap.add_argument("--binary", default=os.path.join("CHROMA", "CHROMA"),
                    help="Path to CHROMA binary (default: CHROMA/CHROMA)")
    ap.add_argument("--algorithm", "-a", default="P_SL_ELS",
                    help="Algorithm spec passed to CHROMA -a (e.g., P_SL_ELS, 0)")
    ap.add_argument("--elastic", "-e", type=int, default=10,
                    help="Elastic θ value passed to CHROMA -e (default: 10)")
    ap.add_argument("--predict", action="store_true", default=False,
                    help="Use prediction model for elastic parameter (adds --predict to CHROMA and omits -e)")
    ap.add_argument("--runs", type=int, default=5,
                    help="Number of runs per dataset (default: 5)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Per-run timeout in seconds (optional)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER,
                    help="Extra args appended to CHROMA invocation (use after --)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: results_{algo}_YYYYmmdd_HHMMSS.json)")
    ap.add_argument("--csv", default=None,
                    help="Output CSV path (optional)")

    args = ap.parse_args(argv)

    datasets = discover_datasets(args.dataset_dir, args.pattern, args.recursive)
    if not datasets:
        print(f"No datasets matched in {args.dataset_dir!r} with pattern {args.pattern!r}", file=sys.stderr)
        return 2

    binary = os.path.abspath(args.binary)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"results_{os.path.basename(binary)}_{args.algorithm}_{ts}.json"
    out_path = out_path.replace(os.sep, "_") if os.path.dirname(out_path) == "" else out_path

    summary: Dict[str, object] = {
        "chroma_config": {
            "binary": binary,
            "algorithm": args.algorithm,
            "elastic": args.elastic,
            "predict": bool(args.predict),
            "runs_per_dataset": args.runs,
            "dataset_dir": os.path.abspath(args.dataset_dir),
            "pattern": args.pattern,
            "recursive": bool(args.recursive),
            "extra_args": args.extra or [],
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        },
        "datasets": {},
    }

    for path in datasets:
        print(f"Running {os.path.basename(path)} ...", flush=True)
        run_results: List[RunResult] = []
        skip_remaining = False
        
        for i in range(args.runs):
            if skip_remaining:
                print(f"  ⚠️  Skipping remaining runs due to previous timeout")
                break
                
            res = run_chroma(
                binary=binary,
                dataset=os.path.abspath(path),
                algorithm=str(args.algorithm),
                elastic=int(args.elastic),
                predict=bool(args.predict),
                extra_args=args.extra or [],
                timeout_sec=args.timeout,
            )
            run_results.append(res)
            status = "ok" if res.ok else f"fail: {res.error}"
            print(f"  run {i+1}/{args.runs}: colors={res.colors_used} runtime_ms={res.runtime_ms:.3f} [{status}]")
            
            # Automatically skip remaining runs if timeout detected
            if "timeout" in res.error.lower() if res.error else False:
                skip_remaining = True
                print(f"  ⚠️  Timeout detected! Automatically skipping remaining runs for {os.path.basename(path)}")

        best = pick_best(run_results)
        # Per-dataset aggregate entry; record theta if consistently parsed from best run
        entry: Dict[str, object] = {
            "path": os.path.abspath(path),
            "best_color": best.colors_used if best else None,
            "best_runtime_ms": round(best.runtime_ms, 6) if best else None,
            "nodes": best.nodes if best and best.nodes is not None else None,
            "edges": best.edges if best and best.edges is not None else None,
            "theta": best.theta if best and best.theta is not None else None,
            "theta_predicted": bool(best.theta_predicted) if best and best.theta_predicted is not None else None,
            "all_runs": [
                {
                    "ok": r.ok,
                    "color": r.colors_used,
                    "runtime_ms": round(r.runtime_ms, 6),
                    "theta": r.theta,
                    "theta_predicted": r.theta_predicted,
                    "error": r.error,
                }
                for r in run_results
            ],
        }
        summary["datasets"][os.path.basename(path)] = entry

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)

    print(f"\nWrote summary to: {out_path}")

    # Write CSV if requested
    if args.csv:
        csv_path = args.csv
        print(f"Writing CSV to: {csv_path}")
        
        # Prepare rows
        csv_rows = []
        # Columns: dataset name, algorithm, elastic number, runtime, color count
        fieldnames = ["dataset name", "algorithm", "elastic number", "runtime", "color count"]
        
        sorted_datasets = sorted(summary["datasets"].keys())
        for ds_name in sorted_datasets:
            ds_data = summary["datasets"][ds_name]
            # 'resilient' is the elastic number. If predict was used, we might want to note that,
            # but user specifically asked for "elastic number (resilisent number改)".
            # We'll use the global arg value for 'elastic number' as it's the input parameter.
            
            row = {
                "dataset name": ds_name,
                "algorithm": ALGO_MAPPING.get(str(args.algorithm), str(args.algorithm)),
                "elastic number": ds_data.get("theta") if ds_data.get("theta") is not None else args.elastic,
                "runtime": ds_data.get("best_runtime_ms", ""),
                "color count": ds_data.get("best_color", "")
            }
            csv_rows.append(row)
            
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"Wrote CSV to: {csv_path}")
        except IOError as e:
            print(f"Error writing CSV: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


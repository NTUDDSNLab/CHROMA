#!/usr/bin/env python3
"""
Grid search for CHROMA resilient number.

For each dataset, iterate resilient values from --res-start to --res-end (inclusive)
with stride 1. For each (dataset, resilient) pair, run the binary 5 times, pick the
best run (smallest colors, ties by fastest runtime), and write a compact JSON:

  {
    "datasetA.egr": {
      "vertices" : 1000000,
      "edges" : 10000000,
      "8": {"color": 23, "runtime_ms": 12.345, "iter_count": 5},
      "9": {"color": 22, "runtime_ms": 12.111, "iter_count": 6}
    },
    "datasetB.egr": { ... }
  }

Usage example:
  python3 grid_resilient.py \
    --dataset-dir Datasets --pattern "*.egr or *.txt" \
    --binary CHROMA/CHROMA --algorithm P_SL_WBR \
    --res-start 5 --res-end 12 \
    --out results_resilient_grid.json

Timeout handling:
  --skip-on-timeout: Skip remaining runs for a resilient value if any run times out
  --skip-dataset-on-timeout: Skip entire dataset if any run times out
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
from typing import Dict, List, Optional


# Reuse parsing patterns consistent with batch_test.py
RUNTIME_RE = re.compile(r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
COLORS_RE = re.compile(r"colors\s+used:\s*(\d+)", re.IGNORECASE)
NODES_RE = re.compile(r"^Nodes:\s*(\d+)$", re.IGNORECASE | re.MULTILINE)
EDGES_RE = re.compile(r"^Edges:\s*(\d+)$", re.IGNORECASE | re.MULTILINE)
ITER_COUNT_RE = re.compile(r"Iter\s+count:\s*(\d+)", re.IGNORECASE)


@dataclass
class RunResult:
    runtime_ms: float
    colors_used: int
    iter_count: Optional[int] = None
    nodes: Optional[int] = None
    edges: Optional[int] = None
    ok: bool = True
    error: Optional[str] = None


def parse_output(stdout: str) -> RunResult:
    runtime_ms: Optional[float] = None
    colors_used: Optional[int] = None
    iter_count: Optional[int] = None
    nodes: Optional[int] = None
    edges: Optional[int] = None

    m = RUNTIME_RE.search(stdout)
    if m:
        runtime_ms = float(m.group(1))

    m = COLORS_RE.search(stdout)
    if m:
        colors_used = int(m.group(1))

    m = ITER_COUNT_RE.search(stdout)
    if m:
        iter_count = int(m.group(1))

    m = NODES_RE.search(stdout)
    if m:
        nodes = int(m.group(1))

    m = EDGES_RE.search(stdout)
    if m:
        edges = int(m.group(1))

    if runtime_ms is None or colors_used is None:
        return RunResult(runtime_ms=0.0, colors_used=-1, iter_count=iter_count, nodes=nodes, edges=edges, ok=False, error="parse-failed")

    return RunResult(runtime_ms=runtime_ms, colors_used=colors_used, iter_count=iter_count, nodes=nodes, edges=edges, ok=True)


def run_chroma(binary: str, dataset: str, algorithm: str, resilient: int,
               extra_args: List[str], timeout_sec: Optional[int]) -> RunResult:
    cmd = [binary, "-f", dataset, "-a", algorithm, "-r", str(resilient)]
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
        return RunResult(runtime_ms=0.0, colors_used=-1, iter_count=None, nodes=None, edges=None, ok=False, error="timeout")
    except FileNotFoundError:
        return RunResult(runtime_ms=0.0, colors_used=-1, iter_count=None, nodes=None, edges=None, ok=False, error="binary-not-found")

    res = parse_output(proc.stdout)
    if proc.returncode != 0:
        res.ok = False
        res.error = f"exit-{proc.returncode}"
    return res


def pick_best(results: List[RunResult]) -> Optional[RunResult]:
    valid = [r for r in results if r.ok]
    if not valid:
        return None
    # Smallest colors first, then fastest runtime
    return sorted(valid, key=lambda r: (r.colors_used, r.runtime_ms))[0]


def discover_datasets(dataset_dir: str, pattern: str, recursive: bool) -> List[str]:
    base = os.path.abspath(dataset_dir)
    patt = os.path.join(base, "**", pattern) if recursive else os.path.join(base, pattern)
    files = glob.glob(patt, recursive=recursive)
    return sorted(f for f in files if os.path.isfile(f))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Grid search resilient number for CHROMA")
    ap.add_argument("--dataset-dir", "-d", default="Datasets",
                    help="Directory containing datasets (default: Datasets)")
    ap.add_argument("--pattern", default="*.egr",
                    help="Glob pattern to match dataset files (default: *.egr)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories while matching datasets")
    ap.add_argument("--binary", default=os.path.join("CHROMA", "CHROMA"),
                    help="Path to CHROMA binary (default: CHROMA/CHROMA)")
    ap.add_argument("--algorithm", "-a", default="P_SL_WBR",
                    help="Algorithm passed to CHROMA -a. Available algorithms: "
                         "0 or P_SL_WBR (default), 1 or P_SL_WBR_SDC")
    ap.add_argument("--res-start", type=int, required=True,
                    help="Start resilient number (inclusive)")
    ap.add_argument("--res-end", type=int, required=True,
                    help="End resilient number (inclusive)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Per-run timeout in seconds (optional)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER,
                    help="Extra args appended to CHROMA invocation (use after --)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: results_resilient_grid_YYYYmmdd_HHMMSS.json)")
    ap.add_argument("--skip-on-timeout", action="store_true", default=False,
                    help="Skip remaining runs for a resilient value if any run times out")
    ap.add_argument("--skip-dataset-on-timeout", action="store_true", default=False,
                    help="Skip entire dataset if any run times out")

    args = ap.parse_args(argv)

    if args.res_end < args.res_start:
        print("--res-end must be >= --res-start", file=sys.stderr)
        return 2

    datasets = discover_datasets(args.dataset_dir, args.pattern, args.recursive)
    if not datasets:
        print(f"No datasets matched in {args.dataset_dir!r} with pattern {args.pattern!r}", file=sys.stderr)
        return 2

    binary = os.path.abspath(args.binary)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"results_resilient_grid_{ts}.json"

    results: Dict[str, Dict[str, any]] = {}

    # Outer loop per dataset: finish all resilient numbers before next dataset
    for ds_path in datasets:
        ds_name = os.path.basename(ds_path)
        print(f"Dataset: {ds_name}")
        ds_entry: Dict[str, any] = {}
        
        # Store vertices and edges info (will be set from first successful run)
        vertices: Optional[int] = None
        edges: Optional[int] = None
        dataset_has_timeout = False

        for r in range(int(args.res_start), int(args.res_end) + 1):
            if dataset_has_timeout and args.skip_dataset_on_timeout:
                print(f"  θ={r}: Skipping (previous timeout with --skip-dataset-on-timeout)")
                ds_entry[str(r)] = {"color": None, "runtime_ms": None}
                continue
                
            print(f"  θ={r} ...", flush=True)
            run_results: List[RunResult] = []
            resilient_has_timeout = False
            
            for i in range(5):  # exactly 5 runs per setting
                if resilient_has_timeout and args.skip_on_timeout:
                    print(f"    run {i+1}/5: Skipped (previous timeout with --skip-on-timeout)")
                    continue
                    
                res = run_chroma(
                    binary=binary,
                    dataset=os.path.abspath(ds_path),
                    algorithm=str(args.algorithm),
                    resilient=int(r),
                    extra_args=args.extra or [],
                    timeout_sec=args.timeout,
                )
                run_results.append(res)
                status = "ok" if res.ok else f"fail:{res.error}"
                iter_info = f" iter={res.iter_count}" if res.iter_count is not None else ""
                print(f"    run {i+1}/5: colors={res.colors_used} runtime_ms={res.runtime_ms:.3f}{iter_info} [{status}]")
                
                # Check for timeout
                if res.error == "timeout":
                    resilient_has_timeout = True
                    dataset_has_timeout = True
                    if args.skip_on_timeout:
                        print(f"    ⚠️  Timeout detected! Skipping remaining runs for θ={r} (--skip-on-timeout)")
                        break
                    elif args.skip_dataset_on_timeout:
                        print(f"    ⚠️  Timeout detected! Will skip remaining datasets (--skip-dataset-on-timeout)")
                        break

            best = pick_best(run_results)
            if best is None:
                ds_entry[str(r)] = {"color": None, "runtime_ms": None, "iter_count": None}
            else:
                ds_entry[str(r)] = {
                    "color": best.colors_used,
                    "runtime_ms": round(best.runtime_ms, 6),
                    "iter_count": best.iter_count
                }
                # Store vertices and edges from first successful run
                if vertices is None and best.nodes is not None:
                    vertices = best.nodes
                if edges is None and best.edges is not None:
                    edges = best.edges
                    
            # Skip remaining resilient values if dataset timeout occurred
            if dataset_has_timeout and args.skip_dataset_on_timeout:
                print(f"  ⚠️  Skipping remaining θ values for {ds_name} due to timeout")
                break

        # Add vertices and edges info to dataset entry
        if vertices is not None:
            ds_entry["vertices"] = vertices
        if edges is not None:
            ds_entry["edges"] = edges
            
        results[ds_name] = ds_entry

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=False)

    print(f"\nWrote grid results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


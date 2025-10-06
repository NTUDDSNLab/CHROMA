#!/usr/bin/env python3
"""
Batch runner for CHROMA_RGP (multi-GPU resilient graph partitioning).

Runs CHROMA_RGP across multiple datasets and partition counts, repeats each
configuration N times (default 5), and records the best run per
(dataset, partition) by the smallest color count, breaking ties by the
fastest runtime. Results are written as JSON with the structure:

  {
    "<partitioner>": {
      "<dataset_basename>": {
        "<partition_count>": {"color": <int>, "runtime": <float_ms>},
        ...
      },
      ...
    }
  }

Examples:
  python3 CHROMA_RGP/batch_rgp.py \
    --dataset-dir Datasets \
    --pattern "*.egr" \
    --partitioners metis ldg \
    --parts 2 4 8 \
    --runs 5 \
    --out rgp_results.json

Notes:
  - Runtime is recorded in milliseconds (ms), matching binary output.
  - Use --partitioners to choose one or more partitioning methods
    (default: metis). You can still use --partitioner for a single
    method; it is treated the same as a single-element --partitioners.
  - Use --binary to override the CHROMA_RGP binary path if needed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional


# Regex patterns match CHROMA_RGP output
RUNTIME_RE = re.compile(r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
COLORS_RE = re.compile(r"colors\s+used:\s*(\d+)", re.IGNORECASE)


@dataclass
class RunResult:
    runtime_ms: float
    colors_used: int
    ok: bool = True
    error: Optional[str] = None
    stdout: Optional[str] = None


def parse_output(stdout: str) -> RunResult:
    runtime_ms: Optional[float] = None
    colors_used: Optional[int] = None

    m = RUNTIME_RE.search(stdout)
    if m:
        runtime_ms = float(m.group(1))

    m = COLORS_RE.search(stdout)
    if m:
        colors_used = int(m.group(1))

    if runtime_ms is None or colors_used is None:
        return RunResult(
            runtime_ms=0.0,
            colors_used=-1,
            ok=False,
            error="Failed to parse runtime and/or colors",
            stdout=stdout,
        )

    return RunResult(
        runtime_ms=runtime_ms,
        colors_used=colors_used,
        ok=True,
        stdout=stdout,
    )


def run_rgp(
    binary: str,
    dataset_path: str,
    parts: int,
    resilient: int,
    partitioner: str,
    extra_args: Optional[List[str]] = None,
    timeout_sec: Optional[int] = None,
) -> RunResult:
    cmd = [binary, "-f", dataset_path, "-p", str(parts), "-r", str(resilient)]
    if partitioner:
        cmd.extend(["--partitioner", partitioner])
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
        return RunResult(runtime_ms=0.0, colors_used=-1, ok=False, error=f"Timeout after {timeout_sec}s")
    except FileNotFoundError:
        return RunResult(runtime_ms=0.0, colors_used=-1, ok=False, error=f"Binary not found: {binary}")

    res = parse_output(proc.stdout)
    if proc.returncode != 0:
        res.ok = False
        rc = proc.returncode
        res.error = f"Exit code {rc}" if not res.error else f"{res.error} (exit {rc})"
    return res


def discover_datasets(dataset_dir: str, pattern: str, recursive: bool) -> List[str]:
    base = os.path.abspath(dataset_dir)
    patt = os.path.join(base, "**", pattern) if recursive else os.path.join(base, pattern)
    files = sorted(f for f in glob.glob(patt, recursive=recursive) if os.path.isfile(f))
    return files


def main(argv: Optional[List[str]] = None) -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="Batch runner for CHROMA_RGP")
    ap.add_argument("--dataset-dir", "-d", default="Datasets",
                    help="Directory containing datasets (default: Datasets)")
    ap.add_argument("--pattern", default="*.egr",
                    help="Glob pattern for dataset files (default: *.egr)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories when matching datasets")
    ap.add_argument("--binary", default=os.path.join(script_dir, "CHROMA_RGP"),
                    help="Path to CHROMA_RGP binary (default: CHROMA_RGP/CHROMA_RGP)")
    ap.add_argument("--parts", "-p", type=int, nargs="+", required=False, default=[2],
                    help="Partition counts to test, e.g. -p 2 4 8 (default: 2)")
    ap.add_argument("--resilient", "-r", type=int, default=10,
                    help="Resilient θ value passed as -r (default: 10)")
    ap.add_argument("--partitioners", "-P", nargs="+", default=None,
                    help="Partitioner methods to test: metis, round_robin, random, ldg, kahip")
    ap.add_argument("--partitioner", default="metis",
                    help="Single partitioner (equivalent to passing one item to --partitioners)")
    ap.add_argument("--runs", type=int, default=5,
                    help="Runs per (dataset, partition) (default: 5)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Per-run timeout in seconds (optional)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER,
                    help="Extra args appended to the CHROMA_RGP invocation (use after --)")
    ap.add_argument("--out", default="rgp_results.json",
                    help="Output JSON path (default: rgp_results.json)")

    args = ap.parse_args(argv)

    datasets = discover_datasets(args.dataset_dir, args.pattern, args.recursive)
    if not datasets:
        print(f"No datasets found in {args.dataset_dir!r} with pattern {args.pattern!r}", file=sys.stderr)
        return 2

    binary = os.path.abspath(args.binary)
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        print(f"Binary not executable: {binary}", file=sys.stderr)
        return 2

    # Partitioners to evaluate
    partitioners: List[str] = args.partitioners if args.partitioners else [args.partitioner]

    # Output structure: partitioner -> dataset -> parts -> {color, runtime}
    out_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for part_method in partitioners:
        print(f"Partitioner: {part_method}")
        out_data.setdefault(part_method, {})

        for dpath in datasets:
            dpath_abs = os.path.abspath(dpath)
            dname = os.path.splitext(os.path.basename(dpath_abs))[0]
            out_data[part_method].setdefault(dname, {})
            print(f"  Dataset: {os.path.basename(dpath_abs)}", flush=True)

            for parts in args.parts:
                print(f"    Parts={parts}:", flush=True)
                run_results: List[RunResult] = []
                for i in range(args.runs):
                    res = run_rgp(
                        binary=binary,
                        dataset_path=dpath_abs,
                        parts=int(parts),
                        resilient=int(args.resilient),
                        partitioner=str(part_method),
                        extra_args=args.extra or [],
                        timeout_sec=args.timeout,
                    )
                    run_results.append(res)
                    status = "ok" if res.ok else f"fail: {res.error}"
                    print(
                        f"      run {i+1}/{args.runs}: colors={res.colors_used} "
                        f"runtime_ms={res.runtime_ms:.3f} [{status}]"
                    )

                # Pick best: min colors, then min runtime
                valids = [r for r in run_results if r.ok]
                if not valids:
                    print("      all runs failed", file=sys.stderr)
                    out_data[part_method][dname][str(parts)] = {
                        "color": None,
                        "runtime": None,
                    }
                    continue

                best = sorted(valids, key=lambda r: (r.colors_used, r.runtime_ms))[0]
                out_data[part_method][dname][str(parts)] = {
                    "color": int(best.colors_used),
                    "runtime": float(round(best.runtime_ms, 6)),  # ms
                }
                print(
                    f"      best => color={best.colors_used}, "
                    f"runtime_ms={best.runtime_ms:.3f}"
                )

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote results to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

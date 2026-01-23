#!/usr/bin/env python3
"""
Batch runner for JP-Series Graph Coloring binary.

Runs the JP-Series binary with different algorithms against all .egr datasets
in the specified directory.

Usage:
  python3 batch_jp_test.py -f <dataset_dir> -o <output.csv> -t <threads> [-r <runs>]
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# Regex to parse output
RUNTIME_RE = re.compile(r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
COLORS_RE = re.compile(r"colors\s+used:\s*(\d+)", re.IGNORECASE)

ALGORITHMS = ["cuSL", "JP-ADG", "JP-SLL"]
BINARY_NAME = "JP-Series"

def get_binary_path(base_dir: str) -> Optional[str]:
    """Finds the JP-Series binary in base_dir."""
    path = os.path.join(base_dir, BINARY_NAME)
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None

def get_datasets(dataset_dir: str) -> List[str]:
    """Finds all .egr files in dataset_dir."""
    pattern = os.path.join(os.path.abspath(dataset_dir), "*.egr")
    files = glob.glob(pattern)
    return sorted([f for f in files if os.path.isfile(f)])

def run_command(binary: str, dataset: str, algorithm: str) -> Tuple[Optional[float], Optional[int], bool, str]:
    """
    Runs the command: ./JP-Series -f <dataset> -a <algorithm>
    Returns: (runtime_ms, colors_used, success, error_msg)
    """
    cmd = [binary, "-f", dataset, "-a", algorithm]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )
    except Exception as e:
        return None, None, False, str(e)

    if proc.returncode != 0:
        return None, None, False, f"Exit code {proc.returncode}"

    # Parse stdout
    runtime = None
    colors = None
    
    m_run = RUNTIME_RE.search(proc.stdout)
    if m_run:
        runtime = float(m_run.group(1))
        
    m_col = COLORS_RE.search(proc.stdout)
    if m_col:
        colors = int(m_col.group(1))

    if runtime is not None and colors is not None:
        return runtime, colors, True, ""
    
    return None, None, False, "Failed to parse output"

def main():
    parser = argparse.ArgumentParser(description="Batch JP-Series Graph Coloring Runner")
    parser.add_argument("-f", "--dataset-dir", required=True, help="Directory containing .egr datasets")
    parser.add_argument("-t", "--threads", type=int, required=True, help="Number of threads (Ignored by binary)")
    parser.add_argument("-o", "--output-csv-path", required=True, help="Path to output CSV file")
    parser.add_argument("-r", "--runs-per-benchmark", type=int, default=1, help="Runs per algorithm per dataset")

    args = parser.parse_args()

    # Inform user about ignored threads
    print(f"Note: Thread count {args.threads} is accepted but ignored by {BINARY_NAME} binary.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    binary = get_binary_path(script_dir)
    datasets = get_datasets(args.dataset_dir)

    if not binary:
        print(f"Binary {BINARY_NAME} not found in script directory!", file=sys.stderr)
        return 1
    
    if not datasets:
        print(f"No .egr datasets found in {args.dataset_dir}", file=sys.stderr)
        return 1

    print(f"Binary: {binary}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Found {len(datasets)} datasets")

    results = []

    for algo in ALGORITHMS:
        for dataset in datasets:
            ds_name = os.path.basename(dataset)
            print(f"Processing {algo} on {ds_name}...")
            
            best_run = None # (runtime, colors)

            for i in range(args.runs_per_benchmark):
                rt, col, success, err = run_command(binary, dataset, algo)
                if success:
                    print(f"  Run {i+1}: runtime={rt}ms, colors={col}")
                    if best_run is None:
                        best_run = (rt, col)
                    else:
                        # Logic: best means smallest color count, then fastest runtime
                        curr_rt, curr_col = best_run
                        if col < curr_col:
                            best_run = (rt, col)
                        elif col == curr_col and rt < curr_rt:
                            best_run = (rt, col)
                else:
                    print(f"  Run {i+1}: Failed ({err})")

            # Record result
            if best_run:
                final_rt, final_col = best_run
                results.append({
                    "dataset name": ds_name,
                    "algorithm": algo,
                    "elastic number": 0, # Default for binary is 0
                    "runtime": final_rt,
                    "color count": final_col
                })
            else:
                print(f"  All runs failed for {algo} on {ds_name}")

    # Write CSV
    if results:
        fieldnames = ["dataset name", "algorithm", "elastic number", "runtime", "color count"]
        try:
            with open(args.output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"Successfully wrote results to {args.output_csv_path}")
        except IOError as e:
            print(f"Error writing CSV: {e}", file=sys.stderr)
            return 1
    else:
        print("No results to write.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

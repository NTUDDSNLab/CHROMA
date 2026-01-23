#!/usr/bin/env python3
"""
Batch runner for CPU Graph Coloring binaries.

Runs all cpu_* binaries in the current directory against all .egr datasets
in the specified directory. Logic:
1. Iterate over all cpu_* executables.
2. Iterate over all .egr files in --dataset-dir.
3. Run each binary on each dataset --runs times.
4. Pick the best run (min colors, then min runtime).
5. Write results to CSV.
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
# Example: "runtime:    12.345678 ms"
RUNTIME_RE = re.compile(r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
# Example: "colors used: 123"
COLORS_RE = re.compile(r"colors\s+used:\s*(\d+)", re.IGNORECASE)

def get_cpu_binaries(base_dir: str) -> List[str]:
    """Finds all executable files starting with 'cpu_' in base_dir."""
    binaries = []
    for f in os.listdir(base_dir):
        if f.startswith("cpu_"):
            full_path = os.path.join(base_dir, f)
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                binaries.append(full_path)
    return sorted(binaries)

def get_datasets(dataset_dir: str) -> List[str]:
    """Finds all .egr files in dataset_dir."""
    pattern = os.path.join(os.path.abspath(dataset_dir), "*.egr")
    files = glob.glob(pattern)
    return sorted([f for f in files if os.path.isfile(f)])

def run_command(binary: str, dataset: str, threads: int) -> Tuple[Optional[float], Optional[int], bool, str]:
    """
    Runs the command: binary dataset threads
    Returns: (runtime_ms, colors_used, success, error_msg)
    """
    cmd = [binary, dataset, str(threads)]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            # timeout=300 # Optional safety timeout
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
    parser = argparse.ArgumentParser(description="Batch CPU Graph Coloring Runner")
    parser.add_argument("-f", "--dataset-dir", required=True, help="Directory containing .egr datasets")
    parser.add_argument("-t", "--threads", type=int, required=True, help="Number of threads")
    parser.add_argument("-o", "--output-csv-path", required=True, help="Path to output CSV file")
    parser.add_argument("-r", "--runs-per-benchmark", type=int, default=1, help="Runs per dataset per algorithm")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    binaries = get_cpu_binaries(script_dir)
    datasets = get_datasets(args.dataset_dir)

    if not binaries:
        print("No cpu_* binaries found in script directory!", file=sys.stderr)
        return 1
    
    if not datasets:
        print(f"No .egr datasets found in {args.dataset_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(binaries)} algorithms: {[os.path.basename(b) for b in binaries]}")
    print(f"Found {len(datasets)} datasets")

    results = []

    for binary in binaries:
        algo_name = os.path.basename(binary)
        for dataset in datasets:
            ds_name = os.path.basename(dataset)
            print(f"Processing {algo_name} on {ds_name}...")
            
            best_run = None # (runtime, colors)

            for i in range(args.runs_per_benchmark):
                rt, col, success, err = run_command(binary, dataset, args.threads)
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
                    "algorithm": algo_name,
                    "elastic number": "-", # Not applicable for CPU algos
                    "runtime": final_rt,
                    "color count": final_col
                })
            else:
                # Log failure row? Or skip? 
                # User asked to "record outcome". I'll skip completely failed cases or record empty
                # Requirements imply "best runtime", if no runs succeeded, we can't report.
                print(f"  All runs failed for {algo_name} on {ds_name}")

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

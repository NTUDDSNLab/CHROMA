#!/usr/bin/env python3
import argparse
import subprocess
import re
import csv
import sys
import os

# Regex patterns
PHASE1_RE = re.compile(r"Phase 1 \(Boundary\+Preload\)\s*:\s*([\d\.]+)\s*ms")
PHASE2_RE = re.compile(r"Phase 2 \(Parallel Exec\)\s*:\s*([\d\.]+)\s*ms")
PHASE3_RE = re.compile(r"Phase 3 \(Result Merge\)\s*:\s*([\d\.]+)\s*ms")
COLORS_RE = re.compile(r"colors\s+used:\s*(\d+)", re.IGNORECASE)
GPU_STATS_RE = re.compile(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|")

def run_chroma(binary_path, graph_file, partitions, partitioner, elastic=10):
    cmd = [binary_path, "-f", graph_file, "-p", str(partitions), "-e", str(elastic)]
    if partitioner:
        cmd.extend(["--partitioner", partitioner])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running CHROMA_RGP: {e}", file=sys.stderr)
        print(e.stdout)
        print(e.stderr)
        return None

def parse_output(output):
    data = {}
    
    # Parse Phases
    p1 = PHASE1_RE.search(output)
    if p1:
        data['Phase 1'] = float(p1.group(1))
    
    p2 = PHASE2_RE.search(output)
    if p2:
        data['Phase 2'] = float(p2.group(1))
        
    p3 = PHASE3_RE.search(output)
    if p3:
        data['Phase 3'] = float(p3.group(1))
        
    # Parse Colors
    c = COLORS_RE.search(output)
    if c:
        data['Colors'] = int(c.group(1))
        
    # Parse GPU Max Mem
    # The table lines look like: | 0      | 1               |          41.8083 |              1106.00 |             24111.06 |
    # Group 4 is Max Mem (MB).
    gpu_mems = []
    for line in output.splitlines():
        m = GPU_STATS_RE.search(line)
        if m:
            # gpu_id = int(m.group(1))
            max_mem = float(m.group(4))
            gpu_mems.append(max_mem)
            
    data['Max Mem (MB)'] = gpu_mems
    return data

def main():
    parser = argparse.ArgumentParser(description="Batch execute CHROMA_RGP and record results.")
    parser.add_argument("--range", required=True, help="Partition range e.g. '2,3,4'")
    parser.add_argument("--graph", "-f", required=True, help="Path to graph file")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    parser.add_argument("--binary", default="../CHROMA_RGP/CHROMA_RGP", help="Path to CHROMA_RGP executable")
    parser.add_argument("--elastic", "-e", default=10, type=int, help="Elastic parameter (default 10)")
    parser.add_argument("--partitioner", help="Partitioner method (optional)")

    args = parser.parse_args()

    # Parse partition range
    try:
        partitions_list = [int(p.strip()) for p in args.range.split(',')]
    except ValueError:
        print("Error: Invalid format for --range. Use comma separated integers like '2,3,4'.", file=sys.stderr)
        sys.exit(1)

    binary_path = os.path.abspath(args.binary)
    if not os.path.exists(binary_path):
         # Try relative to script if default
         script_dir = os.path.dirname(os.path.abspath(__file__))
         candidate = os.path.join(script_dir, args.binary)
         if os.path.exists(candidate):
             binary_path = candidate
         else:
             print(f"Error: Binary not found at {binary_path}", file=sys.stderr)
             sys.exit(1)

    # Prepare CSV
    fieldnames = ['Partitioner', 'Partitions', 'Phase 1 (Boundary+Preload)', 'Phase 2 (Parallel Exec)', 'Phase 3 (Result Merge)', 'Colors', 'Max Mem (MB)']
    
    # Check if we should write header (if file doesn't exist or is empty)
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0

    with open(args.output, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for p in partitions_list:
            print(f"Processing partitions: {p}, partitioner: {args.partitioner}")
            output = run_chroma(binary_path, args.graph, p, args.partitioner, args.elastic)
            
            if output:
                parsed = parse_output(output)
                
                # Format for CSV
                row = {
                    'Partitioner': args.partitioner if args.partitioner else "default",
                    'Partitions': p,
                    'Phase 1 (Boundary+Preload)': parsed.get('Phase 1'),
                    'Phase 2 (Parallel Exec)': parsed.get('Phase 2'),
                    'Phase 3 (Result Merge)': parsed.get('Phase 3'),
                    'Colors': parsed.get('Colors'),
                    'Max Mem (MB)': str(parsed.get('Max Mem (MB)', [])) # Convert list to string
                }
                
                writer.writerow(row)
                csvfile.flush() # Ensure it's written immediately

    print(f"Completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()

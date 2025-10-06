"""
1. Run all the CPU algorithms, including parallel and sequential.
2. The dataset is stored in ../Datasets/*.egr
3. Each dataset is run 5 times. (Please set the timeout of 60 seconds.)
4. Capture the compuation result in each run
    color used: <color_used>
    runtime: <runtime>
5. Use the best result of each algorithm on each dataset (smallest color used) to be the final result.
6. Save the result to CPU/CPU_result.json
    Format:
        {
            "algorithm_name": {
                "dataset_name": {
                    "color_used": <color_used>,
                    "runtime": <runtime>
                }
            }
        }
"""

import os
import subprocess
import json
import glob
import re
import multiprocessing
from typing import Dict, List, Tuple, Optional

# Path configuration - all paths relative to the script location
SEQUENTIAL_DIR = "Sequential"
PARALLEL_DIR = "Parallel" 
DATASETS_DIR = "../Datasets/test"
OUTPUT_FILE = "CPU_result.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_available_algorithms() -> Dict[str, List[str]]:
    """Get all available CPU algorithms (sequential and parallel)."""
    algorithms = {
        "sequential": [],
        "parallel": []
    }
    
    # Sequential algorithms
    seq_dir = os.path.join(SCRIPT_DIR, SEQUENTIAL_DIR)
    for file in os.listdir(seq_dir):
        if file.startswith("cpu_") and os.access(os.path.join(seq_dir, file), os.X_OK):
            algorithms["sequential"].append(file)
    
    # Parallel algorithms
    par_dir = os.path.join(SCRIPT_DIR, PARALLEL_DIR)
    for file in os.listdir(par_dir):
        if file.startswith("cpu_") and os.access(os.path.join(par_dir, file), os.X_OK):
            algorithms["parallel"].append(file)
    
    return algorithms


def get_dataset_files() -> List[str]:
    """Get all .egr dataset files."""
    datasets_dir = os.path.join(SCRIPT_DIR, DATASETS_DIR)
    egr_files = glob.glob(os.path.join(datasets_dir, "*.egr"))
    return [os.path.basename(f) for f in egr_files]


def run_algorithm_with_timeout(algorithm_path: str, dataset_path: str, thread_count: Optional[int] = None, timeout: int = 60) -> Optional[str]:
    """Run algorithm with specified timeout and return output."""
    try:
        cmd = [algorithm_path, dataset_path]
        if thread_count is not None:
            cmd.append(str(thread_count))
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(algorithm_path)
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"Algorithm failed with return code {result.returncode}: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"Algorithm timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error running algorithm: {e}")
        return None


def parse_algorithm_output(output: str) -> Optional[Tuple[int, float]]:
    """Parse algorithm output to extract colors used and runtime."""
    if not output:
        return None
    
    colors_used = None
    runtime = None
    
    # Extract colors used
    colors_match = re.search(r'colors used:\s*(\d+)', output)
    if colors_match:
        colors_used = int(colors_match.group(1))
    
    # Extract runtime (in milliseconds)
    runtime_match = re.search(r'runtime:\s*([\d.]+)\s*ms', output)
    if runtime_match:
        runtime = float(runtime_match.group(1))
    
    if colors_used is not None and runtime is not None:
        return (colors_used, runtime)
    else:
        return None


def get_max_thread_count() -> int:
    """Get the maximum number of threads to use for parallel algorithms."""
    # Start with logical CPU count (includes hyperthreading)
    max_threads = multiprocessing.cpu_count()
    
    # Check if OMP_NUM_THREADS environment variable suggests a different limit
    omp_threads = os.environ.get('OMP_NUM_THREADS')
    if omp_threads:
        try:
            omp_count = int(omp_threads)
            # Use the higher of the two for maximum performance
            max_threads = max(max_threads, omp_count)
        except ValueError:
            pass
    
    # Also check other common thread environment variables
    for env_var in ['NTHREADS', 'PARALLEL_THREADS', 'NUM_THREADS']:
        env_value = os.environ.get(env_var)
        if env_value:
            try:
                env_count = int(env_value)
                max_threads = max(max_threads, env_count)
            except ValueError:
                pass
    
    return max_threads


def run_dataset_with_algorithm(algorithm_type: str, algorithm_name: str, dataset_name: str, num_runs: int = 5) -> Optional[Tuple[int, float]]:
    """Run a specific dataset with a specific algorithm multiple times and return best result."""
    if algorithm_type == "sequential":
        algorithm_path = os.path.join(SCRIPT_DIR, SEQUENTIAL_DIR, algorithm_name)
        thread_count = None
    else:  # parallel
        algorithm_path = os.path.join(SCRIPT_DIR, PARALLEL_DIR, algorithm_name)
        thread_count = get_max_thread_count()
    
    dataset_path = os.path.join(SCRIPT_DIR, DATASETS_DIR, dataset_name)
    
    best_result = None
    best_colors = float('inf')
    skip_remaining = False
    
    print(f"Running {algorithm_name} on {dataset_name} ({num_runs} times) with {thread_count if thread_count else 'single'} thread(s)...")
    
    for run_idx in range(num_runs):
        if skip_remaining:
            print(f"  ⚠️  Skipping remaining runs due to previous timeout")
            break
            
        output = run_algorithm_with_timeout(algorithm_path, dataset_path, thread_count)
        result = parse_algorithm_output(output)
        
        if result:
            colors_used, runtime = result
            print(f"  Run {run_idx + 1}: {colors_used} colors, {runtime:.3f} ms")
            
            # Keep the result with the smallest number of colors
            if colors_used < best_colors:
                best_colors = colors_used
                best_result = (colors_used, runtime)
        else:
            print(f"  Run {run_idx + 1}: Failed")
            
            # Automatically skip remaining runs if timeout occurred
            if output is None:  # timeout occurred
                skip_remaining = True
                print(f"  ⚠️  Timeout detected! Automatically skipping remaining runs for {algorithm_name} on {dataset_name}")
    
    return best_result


def main():
    """Main execution function."""
    print("Starting CPU algorithm benchmark...")
    
    # Get available algorithms and datasets
    algorithms = get_available_algorithms()
    datasets = get_dataset_files()
    
    # Show the maximum thread count that will be used for parallel algorithms
    max_threads = get_max_thread_count()
    print(f"Maximum threads for parallel algorithms: {max_threads}")
    
    print(f"Found {len(algorithms['sequential'])} sequential algorithms: {algorithms['sequential']}")
    print(f"Found {len(algorithms['parallel'])} parallel algorithms: {algorithms['parallel']}")
    print(f"Found {len(datasets)} datasets: {datasets}")
    
    results = {}
    
    # Run all algorithms on all datasets
    all_algorithms = []
    for alg in algorithms['sequential']:
        all_algorithms.append(('sequential', alg))
    for alg in algorithms['parallel']:
        all_algorithms.append(('parallel', alg))
    
    # Process each algorithm
    for alg_type, alg_name in all_algorithms:
        print(f"\n=== Processing algorithm: {alg_name} ({alg_type}) ===")
        results[alg_name] = {}
        
        # Run this algorithm on all datasets
        for dataset in datasets:
            dataset_name = dataset.replace('.egr', '')
            print(f"\nRunning {alg_name} on {dataset_name}...")
            
            result = run_dataset_with_algorithm(alg_type, alg_name, dataset)
            
            if result:
                colors_used, runtime = result
                results[alg_name][dataset_name] = {
                    "color_used": colors_used,
                    "runtime": runtime
                }
                print(f"Best result for {alg_name} on {dataset_name}: {colors_used} colors, {runtime:.3f} ms")
            else:
                print(f"No successful results for {alg_name} on {dataset_name}")
                # Still record the dataset but with null values to maintain structure
                results[alg_name][dataset_name] = {
                    "color_used": None,
                    "runtime": None
                }
    
    # Save results to JSON file
    output_file = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("Benchmark completed!")


if __name__ == "__main__":
    main()
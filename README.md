# CHROMA: Coloring with High-Quality Resilient Optimized Multi-GPU Allocation

## Overview
CHROMA delivers resilient graph coloring across single-GPU, multi-GPU, and CPU backends. The project couples CUDA kernels with resilient graph partitioning (RGP) and CPU heuristics to explore state-of-the-art MIS and SGR strategies on large graphs stored under `Datasets/`.

## Repository Layout
- `CHROMA/` – CUDA single-GPU implementation (`CHROMA.cu`, `Makefile`, optional `model/model.cpp`).
- `CHROMA_RGP/` – Multi-GPU extension with resilient graph partitioning; depends on METIS (and optionally KaHIP).
- `JP-Series/` – Additional CUDA heuristics (JP-SL, JP-ADG, JP-SLL) with dedicated `Makefile`.
- `CPU/Sequential`, `CPU/Parallel` – CPU algorithms built via local Makefiles; orchestrated by `CPU/run.py`.
- `Datasets/` – Input graphs in `.egr`, `.txt`, or `.bin` formats; test sets live under `Datasets/test/`.
- `Scripts/` – Batch and grid evaluators (`batch_test.py`, `grid_resilient.py`).
- `lib/io/` – Shared I/O helpers (`graph.cpp`, `mmio.cpp`).
- `model/` – Optional prediction artifacts (`model.cpp`, generators) used when enabling `PRE_MODEL=1`.
- `External/` – Vendored third-party libraries (METIS, KaHIP, GKlib).

## Requirements
- Linux host with CUDA 12.x toolchain (`nvcc`) and an Ada-class GPU (`ARCH=sm_89` or `sm_90`).
- OpenMP for CHROMA_RGP build; METIS installed under `$HOME/local` or update `INCLUDES_METIS`/`LIBS_METIS`.
- Optional: KaHIP (multi-cut partitioning) and Kokkos if building `CPU/Parallel` Kokkos targets.

## Build & Run
### CHROMA (single GPU)
- **Build**
  ```bash
  cd CHROMA
  make ARCH=sm_89 [PRE_MODEL=1]
  ```
  Use `PRE_MODEL=1` when `model/model.cpp` (generated via `m2cgen`) is present for θ prediction.
- **Run**
  ```bash
  CHROMA/CHROMA -f Datasets/facebook.egr -a P_SL_WBR -r 10
  ```
  Running from inside `CHROMA/` works with `./CHROMA -f ../Datasets/...`. Expect `result verification passed`, `colors used:`, and `runtime:` in the output.
  - **Flags**
    - `-f, --file <path>`: Input graph (required); supports `.egr`, `.txt`, `.bin`.
    - `-a, --algorithm <id>`: Choose `0`/`P_SL_WBR` (default) or `1`/`P_SL_WBR_SDC`.
    - `-r, --resilient <θ>`: Manually set resilient parameter (default `0`).
    - `-p, --predict`: Enable prediction model to auto-select θ (`PRE_MODEL=1`).
    - `-h, --help`: Print usage summary and exit.

### CHROMA_RGP (multi GPU + RGP)
- **Pre-requisite**: Install METIS; adjust `CHROMA_RGP/Makefile` if headers/libraries reside outside `$HOME/local`.
- **Build**
  ```bash
  cd CHROMA_RGP
  make ARCH=sm_89
  ```
- **Run**
  ```bash
  CHROMA_RGP/CHROMA_RGP -f Datasets/facebook.egr -p 2 -r 10
  ```
  `-p` selects partition count; θ defaults to 10 when omitted.
  - **Flags**
    - `-f, --file <path>`: Input graph (required).
    - `-r, --resilient <θ>`: Target resilient number (default `10`).
    - `-p, --parts <count>`: Number of partitions for RGP (default `2`).
    - `--partitioner <name>`: Partitioning strategy `metis` (default), `round_robinm`, `random`, `ldg`, or `kahip`.
    - `-h, --help`: Print usage summary and exit.

### JP-Series CUDA Heuristics
- **Build**
  ```bash
  cd JP-Series
  make ARCH=sm_89
  ```
- **Run**
  ```bash
  JP-Series/JP-Series -f Datasets/facebook.egr -a JP-SL
  ```
  Switch algorithms with `-a JP-ADG` or `-a JP-SLL` as needed.
  - **Flags**
    - `-f, --file <path>`: Input graph (required).
    - `-a, --algorithm <id>`: `0`/`JP-SL` (default), `1`/`JP-ADG`, `2`/`JP-SLL`.
    - `-r, --resilient <θ>`: Optional resilient number (default `0`).
    - `-h, --help`: Print usage summary and exit.

### CPU Implementations
- **Sequential**
  ```bash
  cd CPU/Sequential
  make            # builds cpu_Dstura, cpu_greedy, cpu_SDL
  ./cpu_greedy ../Datasets/test/facebook.egr
  ```
  - **Arguments**
    - `<graph.egr>`: Path to dataset (required). Executables read the graph and print colors/runtime.
- **Parallel (OpenMP/Kokkos)**
  ```bash
  cd CPU/Parallel
  make            # builds OpenMP targets (cpu_ADG, cpu_SL, cpu_SLL, cpu_ecl)
  # make kokkos   # builds cpu_greedy_kokkos when Kokkos is configured
  ./cpu_SL ../Datasets/test/facebook.egr 32
  ```
  - **Arguments**
    - `<graph.egr>`: Path to dataset (required).
    - `<threads>`: Positive integer specifying OpenMP thread count (required for OpenMP/Kokkos binaries).
- **Batch runner**
  ```bash
  cd CPU
  python3 run.py
  ```
  The runner executes compiled `cpu_*` binaries five times per dataset and stores results in `CPU/CPU_result.json`.

## Utility Scripts
- `python3 Scripts/batch_test.py --dataset-dir Datasets --binary CHROMA/CHROMA --runs 5 --out results.json`
- `python3 Scripts/grid_resilient.py --dataset-dir Datasets --res-start 5 --res-end 12`
Capture sample output (colors, runtime) alongside the dataset for reproducibility.
  - **batch_test.py Flags**
    - `--dataset-dir/-d`: Dataset directory (default `Datasets`).
    - `--pattern`: Glob for dataset filenames (default `*.egr`).
    - `--recursive`: Recurse into subdirectories.
    - `--binary`: Path to CHROMA binary (default `CHROMA/CHROMA`).
    - `--algorithm/-a`: Algorithm for `-a` flag (default `P_SL_WBR`).
    - `--resilient`: θ passed via `-r` (default `10`).
    - `--predict`: Enable prediction model (in /model/model.cpp)to auto-select θ (`PRE_MODEL=1`).
    - `--runs`: Repetitions per dataset (default `5`).
    - `--timeout`: Seconds before aborting a single run (optional).
    - `--extra -- <args>`: Additional CHROMA arguments appended after `--` sentinel.
    - `--out`: Output JSON path (default timestamped file).
  - **grid_resilient.py Flags**
    - `--dataset-dir/-d`: Dataset directory (default `Datasets`).
    - `--pattern`: Glob for dataset filenames (default `*.egr`).
    - `--recursive`: Recurse into subdirectories.
    - `--binary`: Path to CHROMA binary (default `CHROMA/CHROMA`).
    - `--algorithm/-a`: Algorithm string passed to CHROMA (default `P_SL_WBR`).
    - `--res-start`: Starting θ (inclusive, required).
    - `--res-end`: Ending θ (inclusive, required).
    - `--timeout`: Seconds before aborting a single run (optional).
    - `--extra -- <args>`: Additional CHROMA arguments appended after `--` sentinel.
    - `--out`: Output JSON path (default `results_resilient_grid_<timestamp>.json`).

## Datasets & References
Download additional graphs from the [ECL collection](https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/index.html). Algorithmic baselines include JP-SL/JP-SLL/JP-ADG (MIS) and CSRColor/Kokkos VB-BIT (SGR) for comparative studies.

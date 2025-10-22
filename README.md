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

### CPU Implementations
- **Sequential**
  ```bash
  cd CPU/Sequential
  make            # builds cpu_Dstura, cpu_greedy, cpu_SDL
  ./cpu_greedy ../Datasets/test/facebook.egr
  ```
- **Parallel (OpenMP/Kokkos)**
  ```bash
  cd CPU/Parallel
  make            # builds OpenMP targets (cpu_ADG, cpu_SL, cpu_SLL, cpu_ecl)
  # make kokkos   # builds cpu_greedy_kokkos when Kokkos is configured
  ./cpu_SL ../Datasets/test/facebook.egr
  ```
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

## Datasets & References
Download additional graphs from the [ECL collection](https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/index.html). Algorithmic baselines include JP-SL/JP-SLL/JP-ADG (MIS) and CSRColor/Kokkos VB-BIT (SGR) for comparative studies.

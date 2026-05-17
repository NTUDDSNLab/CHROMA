# run_sotas — GPU SOTA + JP-Series coloring sweep

`sweep_gpu_sotas.py` builds (fresh, by default) and sweeps 12 GPU
graph-coloring tools over a directory of `.egr` graphs, writing one
aggregated JSON of **total execution time (excluding graph loading)** and
**color count** per `(tool, dataset)`.

## Tools (12)

SOTA (`External/`): `csrcolor`, `data_wlc`, `data_pq`, `kokkos_VB`,
`kokkos_VBBIT`, `pgc_parallel`, `Picasso`, `ECL-GC`, `ECL-GC-R`.
JP-Series: `cuSL`, `JP-ADG`, `JP-SLL`.

## Usage

    python3 scripts/run_sotas/sweep_gpu_sotas.py --dataset-dir Datasets/test
    python3 scripts/run_sotas/sweep_gpu_sotas.py --dataset-dir Datasets/test --arch 89
    python3 scripts/run_sotas/sweep_gpu_sotas.py --dataset-dir Datasets/EGR \
        --runs 3 --only cuSL,ECL-GC,ECL-GC-R --skip-build
    python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest

`--dataset-dir` is required. `--arch` is numeric (e.g. `89` → `sm_89`;
Picasso → `-DCMAKE_CUDA_ARCHITECTURES=89`); default = `nvidia-smi`
auto-detect. `kokkos` arch is fixed by the prebuilt Kokkos (sm_86) and is
not overridden. Build-all-fresh by default; `--skip-build` reuses existing
binaries. Any build/run failure is recorded in the JSON and the sweep
continues.

## Output JSON

`config` (run metadata) + `builds` (per build-unit result) + `tools` (per
tool metadata) + `datasets` (names) + `rows` (flat list, one per
`tool × dataset`: `best_total_exec_ms`, `best_colors`, `runs[]`, `error`).
No CA/PA columns. Default out: `scripts/run_sotas/gpu_sotas_results.json`.

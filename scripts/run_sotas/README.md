# run_sotas — GPU SOTA + JP-Series coloring sweep

`sweep_gpu_sotas.py` builds (fresh, by default) and sweeps 18 GPU
graph-coloring tools over a directory of `.egr` graphs, writing one
aggregated JSON of **total execution time (excluding graph loading)** and
**color count** per `(tool, dataset)`.

## Tools (18)

SOTA (`External/`): `csrcolor`, `data_wlc`, `data_pq`, `kokkos_VB`,
`kokkos_VBBIT`, `pgc_parallel`, `Picasso`, `ECL-GC`, `ECL-GC-R`.
JP-Series: `cuSL`, `JP-ADG`, `JP-SLL`.
CHROMA (`CHROMA/`, one `make PRE_MODEL=1` binary): `CHROMA` (-a 0),
`CHROMA+` (-a 1), `CHROMA*` (-a 1 -p --predict-model v0_paper),
`CHROMA_v2-b-adw` (-a 1 -p --predict-model v3 --no-dynamic-theta),
`CHROMA_v2-b` (-a 10 -p --predict-model v3 --no-dynamic-theta),
`CHROMA_v2` (-a 10 -p --predict-model v3).

Note: the `*` / `+` in CHROMA names are valid in JSON and in the
`--only`/`--exclude` CSV, but quote them in a shell to avoid globbing,
e.g. `--only 'CHROMA*'`.

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

# Design Spec — GPU SOTA + JP-Series Coloring Sweep

- **Date:** 2026-05-17
- **Topic:** `scripts/run_sotas/sweep_gpu_sotas.py`
- **Status:** Approved (brainstorming → spec)

## 1. Goal

Add `scripts/run_sotas/` containing a single Python script `sweep_gpu_sotas.py`
that sweeps the GPU state-of-the-art (SOTA) graph-coloring tools under
`External/` plus all JP-Series algorithms over a directory of `.egr` graphs,
and writes **one aggregated JSON file** recording, per `(tool, dataset)`:

- **total execution time** (milliseconds, **excluding graph-loading time**)
- **color count**

CA/PA time is **explicitly out of scope** (no tool exposes a native CA vs PA
split; user decision: drop those columns entirely).

## 2. Non-goals

- No CA/PA / per-phase timing columns.
- No modification of any tool's source to add instrumentation.
- No new partitioner sweeps (METIS/GKlib/KaHIP/mt-KaHIP are partitioners, not
  coloring tools — excluded).
- No CPU baselines (csrcolor `greedy`, kokkos serial, Picasso `palcolEgr`
  CPU build, etc. excluded).
- Not a generalization of `scripts/batch_test.py` (that script stays
  CHROMA-CLI-specific; this is a separate, self-contained driver).
- The arch CLI flag is added **only to `sweep_gpu_sotas.py`** (user decision).
  Existing scripts run prebuilt binaries and are left unchanged.

## 3. Tool registry (12 entries, all GPU)

Every entry is a declarative record with a uniform interface:
`name`, `kind` (`sota`|`jp`), `build_unit`, `binary` path, `argv(graph)`
builder, and a `parse(stdout) -> (colors:int|None, total_exec_ms:float|None)`
function. The sweep engine is generic and never special-cases a tool inline.

All 12 binaries: accept `.egr`, read the graph **before** their timer starts
(so reported time **excludes graph loading** — verified per-source), and run
non-interactively on `.egr` input.

| name | build_unit | binary (relative to repo root) | argv (G = abs `.egr` path) | colors regex | time regex → ms |
|---|---|---|---|---|---|
| `csrcolor` | `csrcolor` | `External/csrcolor/bin/csrcolor` | `<bin> G` | `^\s*colors used:\s*(\d+)` | `^\s*runtime:\s+([0-9]+(?:\.[0-9]+)?)\s*ms` |
| `data_wlc` | `csrcolor_data` | `External/csrcolor/bin/data_wlc` | `<bin> G` | `^\s*colors used:\s*(\d+)` | `^\s*runtime:\s+([0-9]+(?:\.[0-9]+)?)\s*ms` |
| `data_pq` | `csrcolor_data` | `External/csrcolor/bin/data_pq` | `<bin> G` | `^\s*colors used:\s*(\d+)` | `^\s*runtime:\s+([0-9]+(?:\.[0-9]+)?)\s*ms` |
| `kokkos_VB` | `kokkos` | `External/kokkos-kernels/build/perf_test/graph/graph_color` | `<bin> --cuda 0 --amtx G --algorithm COLORING_VB --repeat 1` | `Num colors:\s*(\d+)` | `Average time over \d+ trials:\s*([0-9.eE+\-]+)\s*sec` ×1000 (fallback `Time:\s*([0-9.eE+\-]+)\s*sec`) |
| `kokkos_VBBIT` | `kokkos` | same as `kokkos_VB` | `<bin> --cuda 0 --amtx G --algorithm COLORING_VBBIT --repeat 1` | `Num colors:\s*(\d+)` | same as `kokkos_VB` ×1000 |
| `pgc_parallel` | `pgc` | `External/Parallel-Graph-Colouring/pgc_parallel` | `<bin> G` | `Number of colours used \(chromatic number\) ==>\s*(\d+)` | `Time Taken \(Parallel\)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms` |
| `Picasso` | `picasso` | `External/Picasso/build/apps/palcolEgrG` | `<bin> --in G --target 16 --recurse --order LIST --check` | `^#\s*of Final colors:\s*(\d+)` | `^\s*Pure Compute Time:\s*([0-9.eE+\-]+)\s*$` ×1000 |
| `ECL-GC` | `ecl-gc` | `External/ECL-GC/ecl-gc` | `<bin> G` | `^\s*colors used:\s*(\d+)` | `^\s*runtime:\s+([0-9.]+)\s+s\s*$` ×1000 |
| `ECL-GC-R` | `ecl-gc-r` | `External/ECL-GC/ecl-gc-r` | `<bin> G` | `colors used after improvement heuristic:\s*(\d+)` | (`^\s*runtime:\s+([0-9.]+)\s+s\s*$` + `^\s*reduce[12] runtime:\s+([0-9.]+)\s+s\s*$`) ×1000 |
| `cuSL` | `jp-series` | `JP-Series/JP-Series` | `<bin> -f G -a cuSL` | `colors\s+used:\s*(\d+)` | `runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms` |
| `JP-ADG` | `jp-series` | `JP-Series/JP-Series` | `<bin> -f G -a JP-ADG` | `colors\s+used:\s*(\d+)` | `runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms` |
| `JP-SLL` | `jp-series` | `JP-Series/JP-Series` | `<bin> -f G -a JP-SLL` | `colors\s+used:\s*(\d+)` | `runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms` |

Regex flags: `re.MULTILINE | re.IGNORECASE`. All times normalised to **ms**:
ms-native tools pass through; `sec`-native tools (`kokkos_*`, `Picasso`,
`ECL-GC`, `ECL-GC-R`) are ×1000.

### Picasso specifics (verified in source)

- IPDPS'24 palette coloring (PNNL). GPU `.egr` binary = `palcolEgrG`. CLI is
  `cxxopts` (named flags, no positional, **no interactive stdin**); `.egr`
  only (`input.readEGR`). `--target 16` = palette size (absolute), `--recurse`
  = recursive refinement of invalid vertices, `--order LIST`, `--check` =
  verify proper coloring (CPU, after coloring; excluded from compute time).
- Picasso prints **per-level** `Num Colors:` / `Assign Time:` lines AND final
  summary lines. The parser MUST use the **final** lines only:
  - colors = `# of Final colors: <N>` (final, after recursion + naive
    fallback) — distinct from per-level `Num Colors:`.
  - time = `Pure Compute Time: <seconds>` = sum of compute across all levels,
    **excludes** graph load (`EGR Load Time`), CUDA warmup, and H2D/D2H/alloc
    (`GPU Copy/Alloc Time`). This is the most apples-to-apples figure vs the
    other tools, whose reported runtime likewise excludes graph load and H2D
    copy (csrcolor/ECL-GC start their timer after the device copy).
- `--target 16` is the palette size, **not** the final color count;
  `# of Final colors:` may differ. Recorded metric is `# of Final colors:`.

### ECL-GC vs ECL-GC-R specifics (verified in source)

- `ECL-GC` = base `External/ECL-GC/ECL-GC_12.cu`. Prints `runtime: %.6f s`
  (timer wraps `init`/`runLarge`/`runSmall`; graph read + H2D copy happen
  before `timer.start()`), then `colors used: %d`.
- `ECL-GC-R` = `External/ECL-GC/ECL-GC-ColorReduction_12.cu` (the prebuilt
  repo-root `ecl-gc-r` is this source). Prints base `runtime: %.6f s`, then
  exactly one of `reduce1 runtime: %.6f s` / `reduce2 runtime: %.6f s`
  (chosen by `avg_degree > 10`), and final `colors used after improvement
  heuristic: %d`. `ECL-GC-R` total time = base `runtime` + whichever
  `reduce{1,2} runtime` appeared. If no reduce line appears (unexpected),
  fall back to base runtime only and set `runs[].error` to a note (still
  `ok` if colors+base time parsed). The base-count line
  `colors used by the original heuristic : %d` is **ignored** for `ECL-GC-R`.
- These are **two separate binaries** built from two separate sources.

## 4. Build phase ("build all fresh first")

Default: rebuild every needed build unit from scratch before sweeping. Only
build units required by the (post-`--only`/`--exclude`) tool set are built.
`--skip-build` skips the entire build phase and uses existing binaries as-is.

Repo root = `/home/chsieh45/PunchShadow/CHROMA` (script derives it from its own
location: `scripts/run_sotas/` → `../../`).

### Architecture flag (numeric)

`--arch` accepts a **numeric** compute-capability, e.g. `89` (also tolerates
`sm_89` and normalises to `89`). The script keeps the canonical number `NN`
and derives the form each builder needs:

- nvcc / make tools → `sm_<NN>` (`-arch=sm_89`, `COMPUTECAPABILITY=sm_89`,
  `ARCH=sm_89`).
- CMake tools (Picasso) → `-DCMAKE_CUDA_ARCHITECTURES=<NN>`.
- `kokkos`: arch is **baked into the prebuilt Kokkos** at
  `/home/chsieh45/local/kokkos-cuda` (`sm_86`); `--arch` is **not applied** to
  the kokkos unit (documented; do not pass an override).

Default `<NN>`: auto-detect via
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` (first GPU; e.g.
`8.9` → `89`); fallback `89` if `nvidia-smi` is missing/fails. Recorded in
`config.arch` (the number) and `config.arch_source`
(`--arch` | `nvidia-smi` | `fallback`).

### Build units (8)

Each run with `subprocess`, stdout+stderr captured, wall time recorded:

| build_unit | command(s) (cwd) | output |
|---|---|---|
| `csrcolor` | `make -C External/csrcolor/src/csrcolor clean` (ignore fail) then `make -C External/csrcolor/src/csrcolor COMPUTECAPABILITY=sm_<NN>` | `External/csrcolor/bin/csrcolor` |
| `csrcolor_data` | `make -C External/csrcolor/src/data clean` (ignore fail) then `make -C External/csrcolor/src/data COMPUTECAPABILITY=sm_<NN>` | `External/csrcolor/bin/data_wlc`, `…/data_pq` |
| `kokkos` | `rm -rf External/kokkos-kernels/build`; `cmake -S External/kokkos-kernels -B External/kokkos-kernels/build -DCMAKE_CXX_COMPILER=/home/chsieh45/local/kokkos-cuda/bin/nvcc_wrapper -DKokkos_ROOT=/home/chsieh45/local/kokkos-cuda -DKokkosKernels_ENABLE_PERFTESTS=ON -DCMAKE_BUILD_TYPE=Release`; `cmake --build External/kokkos-kernels/build --target graph_color -j` | `External/kokkos-kernels/build/perf_test/graph/graph_color` |
| `pgc` | (cwd `External/Parallel-Graph-Colouring`) `nvcc -O3 -std=c++14 -arch=sm_<NN> parallel.cu -o pgc_parallel`; on failure retry once without `-arch` | `External/Parallel-Graph-Colouring/pgc_parallel` |
| `picasso` | `rm -rf External/Picasso/build`; `cmake -S External/Picasso -B External/Picasso/build -DCMAKE_CUDA_ARCHITECTURES=<NN> -DCMAKE_BUILD_TYPE=Release`; `cmake --build External/Picasso/build --target palcolEgrG -j` (fallback: `cmake --build External/Picasso/build -j` if the target name is unknown) | `External/Picasso/build/apps/palcolEgrG` |
| `ecl-gc` | (cwd `External/ECL-GC`) `nvcc -O3 -std=c++17 -arch=sm_<NN> ECL-GC_12.cu -o ecl-gc` | `External/ECL-GC/ecl-gc` |
| `ecl-gc-r` | (cwd `External/ECL-GC`) `nvcc -O3 -std=c++17 -arch=sm_<NN> ECL-GC-ColorReduction_12.cu -o ecl-gc-r` | `External/ECL-GC/ecl-gc-r` |
| `jp-series` | `make -C JP-Series clean` (ignore fail) then `make -C JP-Series ARCH=sm_<NN>` | `JP-Series/JP-Series` |

Notes / risks (documented, handled gracefully — never abort the whole run):

- **kokkos & picasso are the slow/fragile CMake units** (full configure +
  compile, minutes).
  - kokkos needs the prebuilt Kokkos at `/home/chsieh45/local/kokkos-cuda`
    (arch fixed `sm_86`). Missing dir or cmake/make failure → `kokkos` unit
    failed → `kokkos_VB`/`kokkos_VBBIT` rows `ok:false`; sweep continues.
  - picasso needs CMake + CUDA≥11 + OpenMP. cmake/make failure → `picasso`
    unit failed → `Picasso` rows `ok:false`; sweep continues.
- csrcolor: only the two needed sub-Makefiles are built (`src/csrcolor`,
  `src/data`) — the root `make` is **avoided** because it also builds
  `GM`/`topo`/`serial` which may fail to compile and are unused.
  `COMPUTECAPABILITY=sm_<NN>` is a make command-line override (wins over the
  `=` assignment in `src/common.mk`).
- A build-unit failure marks all its tools' rows errored and continues. Build
  outcomes are summarised to stdout and recorded in JSON `builds[]`.

## 5. CLI

`argparse`, PEP 8, type hints. Flags:

- `--dataset-dir PATH` — **required**, no default. Directory of graphs.
- `--pattern STR` — glob, default `*.egr`.
- `--recursive` — recurse into subdirectories (default off).
- `--runs N` — invocations per `(tool, dataset)`, default `1`.
- `--timeout SEC` — per-invocation timeout, default `600`.
- `--out PATH` — default `scripts/run_sotas/gpu_sotas_results.json`.
- `--arch NN` — numeric compute capability, e.g. `89` (also accepts `sm_89`).
  Default: auto-detect (see §4).
- `--skip-build` — reuse existing binaries; skip the build phase.
- `--only NAME[,NAME…]` — restrict to these tool names.
- `--exclude NAME[,NAME…]` — drop these tool names.
  (`--only`/`--exclude` operate on the 12 tool names; the needed build-unit
  set is derived from the resulting tool set.)

No Picasso knobs are exposed — Picasso is fixed to
`--target 16 --recurse --order LIST --check` (user decision).

## 6. Sweep engine / data flow

1. Resolve repo root, arch (`NN`), tool set (apply `--only`/`--exclude`).
2. Discover datasets: `sorted(glob(dataset_dir / pattern))` (or `**/pattern`
   if `--recursive`), files only. Error out if `--dataset-dir` missing/empty.
3. Build phase (unless `--skip-build`): build each needed unit; record
   `builds[]`; compute per-tool availability.
4. Sweep: for each `tool` in registry order, for each `dataset`:
   - if tool unavailable (build failed / binary missing) → one errored row.
   - else run `--runs` invocations:
     `subprocess.run(argv, stdout=PIPE, stderr=STDOUT, text=True,
     timeout=timeout, check=False)` (absolute graph path; cwd = repo root).
     Parse stdout with the tool's regexes. A run is `ok` iff returncode==0 and
     both colors and time parsed. On `TimeoutExpired` or nonzero exit or
     parse-failure → run `error` set; **on timeout, skip remaining runs for
     that cell** (mirrors `batch_test.py`).
5. Pick best per cell and write JSON (atomic: write temp then replace).

## 7. Best-of-N selection

`pick_best`: among `ok` runs, sort by `(colors ASC, total_exec_ms ASC)`, take
first (identical tie-break to `scripts/batch_test.py:pick_best`). If no `ok`
run → `best_*` are `null`, row `ok:false`, `error` = last run's error.

## 8. JSON schema (single aggregated file)

```jsonc
{
  "config": {
    "timestamp": "2026-05-17T12:34:56Z",
    "repo_root": "/home/chsieh45/PunchShadow/CHROMA",
    "dataset_dir": "<abs>",
    "pattern": "*.egr",
    "recursive": false,
    "runs_per_cell": 1,
    "timeout_sec": 600,
    "arch": 89,
    "arch_source": "nvidia-smi | --arch | fallback",
    "skip_build": false,
    "tools": ["csrcolor","data_wlc","data_pq","kokkos_VB","kokkos_VBBIT",
              "pgc_parallel","Picasso","ECL-GC","ECL-GC-R",
              "cuSL","JP-ADG","JP-SLL"]
  },
  "builds": [
    {"unit": "csrcolor", "ok": true,
     "cmd": "make -C … COMPUTECAPABILITY=sm_89", "seconds": 12.3,
     "error": null}
    // one per built unit; --skip-build → builds: []
  ],
  "tools": [
    {"name": "csrcolor", "kind": "sota", "build_unit": "csrcolor",
     "binary": "<abs>", "algorithm": null, "time_unit_src": "ms",
     "available": true, "unavailable_reason": null},
    {"name": "Picasso", "kind": "sota", "build_unit": "picasso",
     "binary": "<abs>/External/Picasso/build/apps/palcolEgrG",
     "algorithm": "target=16,recurse,LIST", "time_unit_src": "s",
     "available": true, "unavailable_reason": null},
    {"name": "cuSL", "kind": "jp", "build_unit": "jp-series",
     "binary": "<abs>/JP-Series/JP-Series", "algorithm": "cuSL",
     "time_unit_src": "ms", "available": true, "unavailable_reason": null}
  ],
  "datasets": ["facebook.egr", "youtube.egr"],
  "rows": [
    {
      "tool": "csrcolor",
      "dataset": "facebook.egr",
      "dataset_path": "<abs>",
      "ok": true,
      "best_total_exec_ms": 12.34,   // excludes graph load;
                                     // ECL-GC-R = base+reduce;
                                     // Picasso = Pure Compute Time ×1000
      "best_colors": 72,
      "runs": [
        {"ok": true, "total_exec_ms": 12.40, "colors": 72,
         "returncode": 0, "error": null}
      ],
      "error": null
    }
  ]
}
```

`rows` is a flat list (one row per `tool × dataset`, registry order then
dataset order) — pandas/plot-friendly, consistent with
`scripts/batch_exe_breakdown_profile.py`. No `ca_ms`/`pa_ms` keys anywhere.

## 9. Error handling (summary)

- Missing/empty `--dataset-dir` → exit non-zero with a clear message.
- Build-unit failure → its tools `available:false`; their rows
  `ok:false, error:"build failed (<unit>): <tail of log>"`; sweep continues.
- `--skip-build` and binary missing → tool `available:false`, errored rows.
- Run nonzero exit / unparseable / timeout → run `error`; cell may have no
  `ok` runs → `best_* = null`. Timeout also skips that cell's remaining runs.
- All failures are captured in JSON; the script exits 0 if it produced a
  result file (a sweep with some failed tools is still a valid result), and
  prints a concise stdout summary (built units, per-tool ok/total cells).

## 10. File layout

```
scripts/run_sotas/
  sweep_gpu_sotas.py   # the driver (single file, registry-driven)
  README.md            # usage, tool list, JSON schema, examples
```

(Default `--out` writes `scripts/run_sotas/gpu_sotas_results.json`; not
committed.)

## 11. Testing / verification

No unit-test framework (per CLAUDE.md — validate via output). Acceptance:

1. Build all 12 tools fresh (auto-detected arch, e.g. `89`).
2. Run:
   `python3 scripts/run_sotas/sweep_gpu_sotas.py --dataset-dir Datasets/test`
   (= `facebook.egr`, `youtube.egr`).
3. Confirm the JSON: for every `available` tool, both test datasets have
   `best_colors > 0` and `best_total_exec_ms > 0`; any build/run failure is
   recorded (not crashing the script); stdout summary printed.
4. Sanity spot-checks (not hard gates):
   - `ECL-GC-R` `best_colors` ≤ `ECL-GC` `best_colors`, and `ECL-GC-R`
     `best_total_exec_ms` ≥ `ECL-GC` `best_total_exec_ms`.
   - `Picasso` `best_colors` is a small positive integer (final colors, not
     the palette size 16) and `best_total_exec_ms` parsed from
     `Pure Compute Time` (not a per-level `Assign Time`).
   - `--arch 86` produces `sm_86` for nvcc/make units and
     `-DCMAKE_CUDA_ARCHITECTURES=86` for Picasso.
```

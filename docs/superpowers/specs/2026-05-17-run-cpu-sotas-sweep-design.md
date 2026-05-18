# Design Spec — CPU SOTA Coloring Sweep (`scripts/run_sotas/sweep_cpu_sotas.py`)

- **Date:** 2026-05-17
- **Topic:** new `scripts/run_sotas/sweep_cpu_sotas.py` (sibling of `sweep_gpu_sotas.py`)
- **Status:** Approved (brainstorming → spec)

## 1. Goal

Add a second registry-driven sweep, `scripts/run_sotas/sweep_cpu_sotas.py`,
modeled on `sweep_gpu_sotas.py`, that builds-fresh and sweeps the **6 CPU
graph-coloring configs** in `CPU/Sequential/` and `CPU/Parallel/` over a
directory of `.egr` graphs, writing **one aggregated JSON** of **total
execution time (ms, excluding graph load)** + **color count** per
`(tool, dataset)`. Same deliverable contract, JSON schema, and engine
behavior as the GPU sweep; CPU-specific differences only.

## 2. The 6 CPU configs (mapping source-verified)

| name (verbatim, symbols kept per user) | kind | build unit | binary (rel to repo root) | argv (after binary) |
|---|---|---|---|---|
| `DSatur` | cpu | `cpu_sequential` | `CPU/Sequential/cpu_Dstura` | `{G}` |
| `Greedy` | cpu | `cpu_sequential` | `CPU/Sequential/cpu_greedy` | `{G}` |
| `JP-SL^M` | cpu | `cpu_sequential` | `CPU/Sequential/cpu_SDL` | `{G}` |
| `JP-SL^A` | cpu | `cpu_parallel` | `CPU/Parallel/cpu_SL` | `{G} {T}` |
| `ADG` | cpu | `cpu_parallel` | `CPU/Parallel/cpu_ADG` | `{G} {T}` |
| `SLL` | cpu | `cpu_parallel` | `CPU/Parallel/cpu_SLL` | `{G} {T}` |

Mapping evidence: `Dstura.cpp` class `DSaturColoring`; `greedy.cpp` "Simple
Greedy"; `JP-SL^M`↔`cpu_SDL` from `scripts/run_pa_sweep.py:64`
`Config("JP-SL_M","SDL",...)`; `JP-SL^A`↔`cpu_SL` from `CPU/Parallel/SL.cpp:361`
("JP-SL^A ordering key"); `ADG.cpp` `compute_ADG`; `SLL.cpp` `compute_SLL`.
`cpu_ecl` and `greedy_kokkos.cpp` are out of scope (excluded).

Registry order: sequential block (`DSatur`, `Greedy`, `JP-SL^M`) then
parallel block (`JP-SL^A`, `ADG`, `SLL`). `kind="cpu"`, `usrc="ms"` for all
6. `{G}` = absolute `.egr` path; `{T}` = the resolved thread count (string),
present only in the 3 parallel argv templates.

**Naming caveat (README):** `^` in `JP-SL^M`/`JP-SL^A` is kept verbatim; it
is a valid JSON key and not a bash glob metacharacter, and the
`--only`/`--exclude` CSV split handles it. No code change needed.

## 3. Parsers (verified — all 6 identical, reuse GPU JP-family patterns)

Every binary prints `runtime:    %.6f ms` and `colors used: %d`, with the
timer started **after** `readECLgraph()` (graph load excluded), in **ms**,
one authoritative color line (no reduction):

- colors: `r"colors\s+used:\s*(\d+)"`
- time: `("ms", r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")`

Same `parse_colors` / `parse_time_ms` (`"ms"` kind) logic as the GPU script.
No new time-spec kinds, no reduction handling.

## 4. Build units (2; g++, no GPU/arch)

| unit | steps (cwd = repo root) | output |
|---|---|---|
| `cpu_sequential` | `make -C CPU/Sequential clean` (ignore_fail) ; `make -C CPU/Sequential` | `CPU/Sequential/cpu_Dstura`, `cpu_greedy`, `cpu_SDL` |
| `cpu_parallel` | `make -C CPU/Parallel clean` (ignore_fail) ; `make -C CPU/Parallel` | `CPU/Parallel/cpu_SL`, `cpu_ADG`, `cpu_SLL` (also `cpu_ecl`, unused/harmless) |

The Makefiles set `g++ -O3 -std=c++17 -Wall` (+ `-fopenmp` for Parallel);
the sweep's build step is just `make` — no arch/PRE_MODEL/kokkos. Build-all-
fresh by default; a build-unit failure marks its tools `unavailable`
(`build failed (<unit>)`) and the sweep continues (same
`compute_availability`/graceful-degradation engine as the GPU sweep).
`--skip-build` reuses existing binaries.

## 5. CLI

Same flags as the GPU sweep **minus** `--arch` and `--kokkos-root`,
**plus** `--threads`:

- `--dataset-dir PATH` — **required**.
- `--pattern STR` — default `*.egr`.
- `--recursive` — recurse into subdirs.
- `--runs N` — invocations per `(tool, dataset)`, default `1`.
- `--timeout SEC` — per-invocation timeout, default `600`.
- `--out PATH` — default `scripts/run_sotas/cpu_sotas_results.json`.
- `--threads N` — OpenMP thread count passed as `{T}` to the 3 parallel
  configs; default = `os.cpu_count()` (sequential configs ignore it).
- `--skip-build`, `--only NAME[,NAME…]`, `--exclude NAME[,NAME…]`,
  `--selftest`.

## 6. Engine, JSON schema, errors (identical to GPU sweep)

Reuse the GPU sweep's structure verbatim in spirit: `select_tools` /
`needed_units` / `discover_datasets` / `pick_best` (best by
`(colors ASC, total_exec_ms ASC)`), `build_unit_steps` /
`build_one_unit` / `run_build_phase`, `assemble_run` / `run_cell`
(timeout → skip remaining runs for the cell), `compute_availability` /
`build_json_doc` / `write_json_atomic` / `print_summary`, `run_sweep`,
`--selftest` harness. JSON schema is identical:

```jsonc
{
  "config": { "timestamp","repo_root","dataset_dir","pattern","recursive",
              "runs_per_cell","timeout_sec","threads","skip_build",
              "tools":[...] },              // threads replaces arch/arch_source
  "builds":  [ {"unit","ok","cmd","seconds","error"} ],
  "tools":   [ {"name","kind":"cpu","build_unit","binary","algorithm",
                "time_unit_src":"ms","available","unavailable_reason"} ],
  "datasets":[ ... ],
  "rows":    [ {"tool","dataset","dataset_path","ok",
                "best_total_exec_ms","best_colors","runs":[...],"error"} ]
}
```

No `ca_ms`/`pa_ms` anywhere. `build_argv` substitutes `{G}` → graph path and
`{T}` → `str(threads)`; `run_sweep` resolves `threads = args.threads or
os.cpu_count()` and records `config.threads`. Exit-code semantics unchanged
(0 if a result file was produced; 2 only for the arg-validation failures:
missing/not-a-dir/empty `--dataset-dir`, unknown `--only`/`--exclude`).

## 7. Selftest (mirrors GPU structure, scaled to CPU)

No pytest (project constraint) — `--selftest` is the mechanism (red/green
via program output). Checks:

- **parsers (golden):** CPU sample stdout
  `"runtime:    3.500000 ms\ncolors used: 40\n"` → `parse_colors`==40,
  `parse_time_ms`==3.5 (2 checks).
- **registry:** size==6; names unique==6 (2).
- **filters:** `select_tools` only-filter (order preserved), exclude count,
  no-filter==6, unknown name → `SystemExit`; `needed_units` for a
  sequential-only and a parallel-only selection (≈6).
- **pick_best:** colors-then-time tie-break; all-failed → None (3).
- **build cmds:** `build_unit_steps("cpu_sequential")` == clean(ignore_fail
  True)+`make -C CPU/Sequential`; `build_unit_steps("cpu_parallel")`
  likewise for `CPU/Parallel`; unknown unit → ValueError (≈3).
- **build_argv / threads:** sequential entry → `[bin, G]`; parallel entry
  with threads=8 → `[bin, G, "8"]` (`{T}` substituted) (2).
- **assemble_run:** ok / nonzero-rc / unparseable / timeout (4).
- **json shape:** top keys == `[builds,config,datasets,rows,tools]`;
  `tools[].kind=="cpu"`; `config` carries `threads`; no `ca_ms`/`pa_ms`
  (≈4).

The plan pins the exact `SELFTEST: PASS (N/N checks)` literal and includes a
stop-and-report guard if the implemented count differs (per the GPU plan's
count-correction precedent).

## 8. Docs

`scripts/run_sotas/README.md` gains a short **"## CPU sweep
(`sweep_cpu_sotas.py`)"** section: the 6 tools, the 2 build units, the
`--threads` default (`os.cpu_count()`), the JSON/`--only` naming note, and
that it is GPU-sweep-independent.

## 9. Testing / acceptance

1. `--selftest` → `SELFTEST: PASS (N/N checks)`, exit 0.
2. Build both CPU units fresh and run
   `python3 scripts/run_sotas/sweep_cpu_sotas.py --dataset-dir Datasets/test`:
   `cpu_sequential` + `cpu_parallel` build OK; all 6 tools `available`;
   every cell on `facebook.egr`+`youtube.egr` has `best_colors>0` and
   `best_total_exec_ms>0`; script exits 0.
3. Spot-check: a parallel config's recorded run used `--threads`
   (`config.threads` set; the binary received `<graph> <threads>`);
   `best_total_exec_ms` parsed from `runtime:` (excludes graph load).

Implementation proceeds via the established subagent-driven loop
(implementer → spec-compliance review → code-quality review) with
concurrent-session-safe scoped (pathspec) commits — same as the GPU sweep.
This script is independent of `sweep_gpu_sotas.py`; the GPU script is not
modified.

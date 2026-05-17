# GPU SOTA + JP-Series Coloring Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/run_sotas/sweep_gpu_sotas.py`, a single registry-driven
driver that builds-fresh and sweeps 12 GPU graph-coloring tools over a
directory of `.egr` graphs, writing one JSON of total exec time (excl. graph
load) + color count per `(tool, dataset)`.

**Architecture:** One self-contained Python file. A declarative `REGISTRY` of
12 tool records (binary path, argv template, stdout colors/time regexes); a
generic engine: resolve arch → build needed units → sweep `tool × dataset ×
runs` → pick best → emit JSON. Pure logic (arch normalization, parsers,
pick_best, argv/build-command construction, JSON assembly) is exercised by an
embedded `--selftest` mode (validated via program output, per `CLAUDE.md`,
which states there is no unit-test framework — so we use a dependency-free
in-script selftest for red/green instead of pytest). The real build+sweep is
acceptance-validated on `Datasets/test`.

**Tech Stack:** Python 3 stdlib only (`argparse`, `subprocess`, `re`, `json`,
`glob`, `os`, `shutil`, `dataclasses`-free dict rows). Builds shell out to
`make` / `cmake` / `nvcc`. Spec: `docs/superpowers/specs/2026-05-17-run-gpu-sotas-sweep-design.md`.

---

## File Structure

- **Create** `scripts/run_sotas/sweep_gpu_sotas.py` — the entire driver
  (single file, sections in this order: header/imports → constants →
  arch → registry + parsers → discovery/pick_best → argv + build commands →
  build phase → sweep → json + summary → selftest → argparse/main).
- **Create** `scripts/run_sotas/README.md` — usage, tool list, JSON schema,
  examples.

Single file matches the spec (§10) and the `scripts/` convention (every
existing script is a single file). Do not split into a package.

All shell commands below are run from the repo root
`/home/chsieh45/PunchShadow/CHROMA` unless a task says otherwise. The script
derives the repo root from its own location: `scripts/run_sotas/ → ../../`.

---

## Task 1: Scaffold directory, README, and runnable skeleton

**Files:**
- Create: `scripts/run_sotas/sweep_gpu_sotas.py`
- Create: `scripts/run_sotas/README.md`

- [ ] **Step 1: Create the README**

Create `scripts/run_sotas/README.md`:

```markdown
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
```

- [ ] **Step 2: Create the script skeleton**

Create `scripts/run_sotas/sweep_gpu_sotas.py`:

```python
#!/usr/bin/env python3
"""Build-and-sweep 12 GPU graph-coloring tools over a directory of .egr graphs.

Records total execution time (excluding graph loading) and color count per
(tool, dataset) into one JSON. See
docs/superpowers/specs/2026-05-17-run-gpu-sotas-sweep-design.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional

# Repo root = two levels up from this file (scripts/run_sotas/ -> ../../).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BUILD_TIMEOUT_SEC = 3600  # hard cap per build-unit (cmake/make can run minutes)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        return run_selftest()
    return run_sweep(args)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sweep_gpu_sotas.py",
        description="Build + sweep GPU SOTA / JP-Series coloring tools.",
    )
    p.add_argument("--dataset-dir", help="Directory of .egr graphs (required "
                   "unless --selftest).")
    p.add_argument("--pattern", default="*.egr", help="Glob (default *.egr).")
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into subdirectories.")
    p.add_argument("--runs", type=int, default=1,
                   help="Invocations per (tool, dataset) (default 1).")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-invocation timeout seconds (default 600).")
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "scripts", "run_sotas",
                                        "gpu_sotas_results.json"),
                   help="Output JSON path.")
    p.add_argument("--arch", default=None,
                   help="Numeric compute capability, e.g. 89 (also accepts "
                        "sm_89). Default: nvidia-smi auto-detect.")
    p.add_argument("--skip-build", action="store_true",
                   help="Reuse existing binaries; skip the build phase.")
    p.add_argument("--only", default=None,
                   help="Comma-separated tool names to include.")
    p.add_argument("--exclude", default=None,
                   help="Comma-separated tool names to exclude.")
    p.add_argument("--selftest", action="store_true",
                   help="Run embedded logic checks and exit.")
    return p


def run_sweep(args: argparse.Namespace) -> int:
    raise NotImplementedError  # filled in Task 8


def run_selftest() -> int:
    raise NotImplementedError  # filled in Task 2+


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Verify the skeleton runs**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --help`
Expected: argparse help text listing all flags; exit 0.

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `NotImplementedError` traceback (selftest not implemented yet) —
confirms dispatch reaches `run_selftest`.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_sotas/sweep_gpu_sotas.py scripts/run_sotas/README.md
git commit -m "scripts/run_sotas: scaffold sweep_gpu_sotas.py skeleton + README

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Arch resolution + selftest harness

**Files:**
- Modify: `scripts/run_sotas/sweep_gpu_sotas.py`

- [ ] **Step 1: Write the selftest harness and the first failing cases**

Replace the placeholder `def run_selftest(): raise NotImplementedError` with
this harness plus arch cases (insert near the bottom, above `if __name__`):

```python
def _check(results: list, name: str, got, expected) -> None:
    ok = got == expected
    results.append((ok, name, got, expected))


def run_selftest() -> int:
    results: list = []
    selftest_arch(results)
    failed = [r for r in results if not r[0]]
    for ok, name, got, expected in results:
        if not ok:
            print(f"FAIL {name}: got={got!r} expected={expected!r}")
    print(f"SELFTEST: {'PASS' if not failed else 'FAIL'} "
          f"({len(results) - len(failed)}/{len(results)} checks)")
    return 0 if not failed else 1


def selftest_arch(results: list) -> None:
    _check(results, "normalize 89", normalize_arch("89"), "89")
    _check(results, "normalize sm_86", normalize_arch("sm_86"), "86")
    _check(results, "normalize 8.9", normalize_arch("8.9"), "89")
    _check(results, "normalize SM_90 spaced",
           normalize_arch("  SM_90 "), "90")
    raised = False
    try:
        normalize_arch("abc")
    except ValueError:
        raised = True
    _check(results, "normalize invalid raises", raised, True)
```

- [ ] **Step 2: Run selftest to verify it fails**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `NameError: name 'normalize_arch' is not defined` (function not
written yet).

- [ ] **Step 3: Implement arch functions**

Add (place in the arch section, after the constants):

```python
def normalize_arch(value: str) -> str:
    """'89' / 'sm_89' / '8.9' -> '89'. Raises ValueError if not numeric."""
    s = value.strip().lower()
    if s.startswith("sm_"):
        s = s[3:]
    s = s.replace(".", "")
    if not s or not s.isdigit():
        raise ValueError(f"invalid arch: {value!r}")
    return s


def detect_arch() -> Optional[str]:
    """First GPU compute capability via nvidia-smi, e.g. '8.9' -> '89'."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                return normalize_arch(line)
            except ValueError:
                return None
    return None


def resolve_arch(cli_arch: Optional[str]) -> tuple[str, str]:
    """Return (NN, source) where source in {--arch, nvidia-smi, fallback}."""
    if cli_arch:
        return normalize_arch(cli_arch), "--arch"
    detected = detect_arch()
    if detected:
        return detected, "nvidia-smi"
    return "89", "fallback"
```

- [ ] **Step 4: Run selftest to verify arch cases pass**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `SELFTEST: PASS (5/5 checks)`; exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sotas/sweep_gpu_sotas.py
git commit -m "scripts/run_sotas: arch resolution + selftest harness

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Tool registry + stdout parsers

**Files:**
- Modify: `scripts/run_sotas/sweep_gpu_sotas.py`

- [ ] **Step 1: Add failing parser selftest cases**

Add this function and call it from `run_selftest` (add
`selftest_parsers(results)` right after `selftest_arch(results)`):

```python
def selftest_parsers(results: list) -> None:
    by_name = {t["name"]: t for t in REGISTRY}

    csr = ("num_vertices 4039 num_edges 176468\n"
           "runtime:    1.234560 ms\ncolors used: 72\ncorrect.\n")
    _check(results, "csrcolor colors",
           parse_colors(by_name["csrcolor"], csr), 72)
    _check(results, "csrcolor ms",
           parse_time_ms(by_name["csrcolor"], csr), 1.23456)

    kok = ("algorithm: 3\n\nTime:0.005599 sec. Num colors:5 Num Phases:5\n"
           "\t5 4 3 2 1 \nAverage time over 1 trials: 0.005599 sec.\n")
    _check(results, "kokkos colors",
           parse_colors(by_name["kokkos_VB"], kok), 5)
    _check(results, "kokkos ms",
           round(parse_time_ms(by_name["kokkos_VB"], kok), 4), 5.599)

    pgc = ("Read .egr: nodes=4039, edges=176468\n\n"
           "Number of colours used (chromatic number) ==> 70\n"
           "Time Taken (Parallel) = 2.500000 ms\n")
    _check(results, "pgc colors",
           parse_colors(by_name["pgc_parallel"], pgc), 70)
    _check(results, "pgc ms",
           parse_time_ms(by_name["pgc_parallel"], pgc), 2.5)

    pic = ("EGR Load Time: 0.0012\n***********Level 0*******\n"
           "Num Nodes: 4039\nNum Colors: 15\nAssign Time: 1.30097\n"
           "# of Final colors: 26\nPure Compute Time: 1.335427\n"
           "GPU Copy/Alloc Time: 0.0123\n")
    _check(results, "picasso colors",
           parse_colors(by_name["Picasso"], pic), 26)
    _check(results, "picasso ms (final, not per-level)",
           round(parse_time_ms(by_name["Picasso"], pic), 3), 1335.427)

    eclg = ("ECL-GC v1.2 (ECL-GC_12.cu)\ninput: facebook.egr\nnodes: 4039\n"
            "runtime:    0.001234 s\nresult verification passed\n"
            "colors used: 71\ncol  0: 100\n")
    _check(results, "ecl-gc colors",
           parse_colors(by_name["ECL-GC"], eclg), 71)
    _check(results, "ecl-gc ms",
           round(parse_time_ms(by_name["ECL-GC"], eclg), 4), 1.234)

    eclr = ("colors used by the original heuristic : 71\n"
            "runtime:    0.002000 s\nreduce1 runtime:    0.000500 s\n"
            "colors used after improvement heuristic: 65\n")
    _check(results, "ecl-gc-r colors (after improvement)",
           parse_colors(by_name["ECL-GC-R"], eclr), 65)
    _check(results, "ecl-gc-r ms (base+reduce)",
           round(parse_time_ms(by_name["ECL-GC-R"], eclr), 4), 2.5)

    jp = ("Input file: facebook.egr\nAlgorithm: cuSL\n"
          "Resilient number: 0\nNodes: 4039\nEdges: 176468\n"
          "runtime:    15.666178 ms\nresult verification passed\n"
          "colors used: 72\n")
    _check(results, "jp cuSL colors",
           parse_colors(by_name["cuSL"], jp), 72)
    _check(results, "jp cuSL ms",
           parse_time_ms(by_name["cuSL"], jp), 15.666178)

    _check(results, "registry size", len(REGISTRY), 12)
    _check(results, "registry names unique",
           len({t["name"] for t in REGISTRY}), 12)
```

- [ ] **Step 2: Run selftest to verify it fails**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `NameError: name 'REGISTRY' is not defined`.

- [ ] **Step 3: Implement the registry and parsers**

Add the registry section (after arch functions):

```python
def _find_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return int(m.group(1)) if m else None


def _find_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_colors(tool: dict, text: str) -> Optional[int]:
    return _find_int(tool["colors"], text)


def parse_time_ms(tool: dict, text: str) -> Optional[float]:
    spec = tool["time"]
    kind = spec[0]
    if kind == "ms":
        return _find_float(spec[1], text)
    if kind == "sec":
        v = _find_float(spec[1], text)
        return None if v is None else v * 1000.0
    if kind == "sec_first":
        for pat in spec[1]:
            v = _find_float(pat, text)
            if v is not None:
                return v * 1000.0
        return None
    if kind == "sec_sum":
        base = _find_float(spec[1], text)
        if base is None:
            return None
        red = _find_float(spec[2], text)
        return (base + (red or 0.0)) * 1000.0
    raise ValueError(f"bad time spec: {spec!r}")


# Time-format strings verified against each tool's source.
_MS = r"^\s*runtime:\s+([0-9]+(?:\.[0-9]+)?)\s*ms"
_COLORS_USED = r"^\s*colors used:\s*(\d+)"

REGISTRY: list[dict] = [
    dict(name="csrcolor", kind="sota", unit="csrcolor",
         binrel="External/csrcolor/bin/csrcolor", argv=["{G}"],
         colors=_COLORS_USED, time=("ms", _MS), algo=None, usrc="ms"),
    dict(name="data_wlc", kind="sota", unit="csrcolor_data",
         binrel="External/csrcolor/bin/data_wlc", argv=["{G}"],
         colors=_COLORS_USED, time=("ms", _MS), algo=None, usrc="ms"),
    dict(name="data_pq", kind="sota", unit="csrcolor_data",
         binrel="External/csrcolor/bin/data_pq", argv=["{G}"],
         colors=_COLORS_USED, time=("ms", _MS), algo=None, usrc="ms"),
    dict(name="kokkos_VB", kind="sota", unit="kokkos",
         binrel="External/kokkos-kernels/build/perf_test/graph/graph_color",
         argv=["--cuda", "0", "--amtx", "{G}", "--algorithm",
               "COLORING_VB", "--repeat", "1"],
         colors=r"Num colors:\s*(\d+)",
         time=("sec_first",
               [r"Average time over \d+ trials:\s*([0-9.eE+\-]+)\s*sec",
                r"^\s*Time:\s*([0-9.eE+\-]+)\s*sec"]),
         algo="COLORING_VB", usrc="s"),
    dict(name="kokkos_VBBIT", kind="sota", unit="kokkos",
         binrel="External/kokkos-kernels/build/perf_test/graph/graph_color",
         argv=["--cuda", "0", "--amtx", "{G}", "--algorithm",
               "COLORING_VBBIT", "--repeat", "1"],
         colors=r"Num colors:\s*(\d+)",
         time=("sec_first",
               [r"Average time over \d+ trials:\s*([0-9.eE+\-]+)\s*sec",
                r"^\s*Time:\s*([0-9.eE+\-]+)\s*sec"]),
         algo="COLORING_VBBIT", usrc="s"),
    dict(name="pgc_parallel", kind="sota", unit="pgc",
         binrel="External/Parallel-Graph-Colouring/pgc_parallel",
         argv=["{G}"],
         colors=r"Number of colours used \(chromatic number\)\s*==>\s*(\d+)",
         time=("ms",
               r"Time Taken \(Parallel\)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
         algo=None, usrc="ms"),
    dict(name="Picasso", kind="sota", unit="picasso",
         binrel="External/Picasso/build/apps/palcolEgrG",
         argv=["--in", "{G}", "--target", "16", "--recurse",
               "--order", "LIST", "--check"],
         colors=r"^#\s*of Final colors:\s*(\d+)",
         time=("sec", r"^\s*Pure Compute Time:\s*([0-9.eE+\-]+)\s*$"),
         algo="target=16,recurse,LIST", usrc="s"),
    dict(name="ECL-GC", kind="sota", unit="ecl-gc",
         binrel="External/ECL-GC/ecl-gc", argv=["{G}"],
         colors=_COLORS_USED,
         time=("sec", r"^\s*runtime:\s+([0-9.]+)\s+s\s*$"),
         algo=None, usrc="s"),
    dict(name="ECL-GC-R", kind="sota", unit="ecl-gc-r",
         binrel="External/ECL-GC/ecl-gc-r", argv=["{G}"],
         colors=r"colors used after improvement heuristic:\s*(\d+)",
         time=("sec_sum",
               r"^\s*runtime:\s+([0-9.]+)\s+s\s*$",
               r"^\s*reduce[12] runtime:\s+([0-9.]+)\s+s\s*$"),
         algo=None, usrc="s"),
    dict(name="cuSL", kind="jp", unit="jp-series",
         binrel="JP-Series/JP-Series",
         argv=["-f", "{G}", "-a", "cuSL"],
         colors=r"colors\s+used:\s*(\d+)",
         time=("ms", r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
         algo="cuSL", usrc="ms"),
    dict(name="JP-ADG", kind="jp", unit="jp-series",
         binrel="JP-Series/JP-Series",
         argv=["-f", "{G}", "-a", "JP-ADG"],
         colors=r"colors\s+used:\s*(\d+)",
         time=("ms", r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
         algo="JP-ADG", usrc="ms"),
    dict(name="JP-SLL", kind="jp", unit="jp-series",
         binrel="JP-Series/JP-Series",
         argv=["-f", "{G}", "-a", "JP-SLL"],
         colors=r"colors\s+used:\s*(\d+)",
         time=("ms", r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
         algo="JP-SLL", usrc="ms"),
]
```

- [ ] **Step 4: Run selftest to verify parser cases pass**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `SELFTEST: PASS (21/21 checks)` (5 arch + 16 parser/registry).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sotas/sweep_gpu_sotas.py
git commit -m "scripts/run_sotas: 12-tool registry + verified stdout parsers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Tool filtering, dataset discovery, pick_best

**Files:**
- Modify: `scripts/run_sotas/sweep_gpu_sotas.py`

- [ ] **Step 1: Add failing selftest cases**

Add and wire `selftest_engine(results)` into `run_selftest` (call it after
`selftest_parsers(results)`):

```python
def selftest_engine(results: list) -> None:
    all_names = [t["name"] for t in REGISTRY]

    sel = select_tools(only="cuSL,ECL-GC", exclude=None)
    _check(results, "only filter", [t["name"] for t in sel],
           ["ECL-GC", "cuSL"])  # registry order preserved

    sel2 = select_tools(only=None, exclude="data_wlc,data_pq")
    _check(results, "exclude filter count", len(sel2), 10)
    _check(results, "exclude removed", "data_wlc" in
           [t["name"] for t in sel2], False)

    sel3 = select_tools(only=None, exclude=None)
    _check(results, "no filter = all", len(sel3), 12)

    raised = False
    try:
        select_tools(only="NoSuchTool", exclude=None)
    except SystemExit:
        raised = True
    _check(results, "unknown --only errors", raised, True)

    _check(results, "needed units (cuSL only)",
           sorted(needed_units(select_tools("cuSL", None))),
           ["jp-series"])
    _check(results, "needed units (kokkos_VB,data_pq)",
           sorted(needed_units(select_tools("kokkos_VB,data_pq", None))),
           ["csrcolor_data", "kokkos"])

    runs = [
        {"ok": True, "colors": 30, "total_exec_ms": 9.0},
        {"ok": True, "colors": 28, "total_exec_ms": 12.0},
        {"ok": True, "colors": 28, "total_exec_ms": 11.0},
        {"ok": False, "colors": None, "total_exec_ms": None},
    ]
    best = pick_best(runs)
    _check(results, "pick_best colors", best["colors"], 28)
    _check(results, "pick_best tie->faster", best["total_exec_ms"], 11.0)
    _check(results, "pick_best none", pick_best(
        [{"ok": False, "colors": None, "total_exec_ms": None}]), None)
```

- [ ] **Step 2: Run selftest to verify it fails**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `NameError: name 'select_tools' is not defined`.

- [ ] **Step 3: Implement filtering, discovery, pick_best**

Add (discovery/pick_best section):

```python
def select_tools(only: Optional[str],
                  exclude: Optional[str]) -> list[dict]:
    """Subset REGISTRY (order preserved). Errors on unknown names."""
    known = {t["name"] for t in REGISTRY}

    def parse_csv(v: Optional[str]) -> list[str]:
        return [x.strip() for x in v.split(",") if x.strip()] if v else []

    only_l = parse_csv(only)
    excl_l = parse_csv(exclude)
    for n in only_l + excl_l:
        if n not in known:
            sys.stderr.write(
                f"error: unknown tool name {n!r}; valid: "
                f"{', '.join(t['name'] for t in REGISTRY)}\n")
            raise SystemExit(2)
    out = []
    for t in REGISTRY:
        if only_l and t["name"] not in only_l:
            continue
        if t["name"] in excl_l:
            continue
        out.append(t)
    return out


def needed_units(tools: list[dict]) -> set[str]:
    return {t["unit"] for t in tools}


def discover_datasets(dataset_dir: str, pattern: str,
                      recursive: bool) -> list[str]:
    base = os.path.abspath(dataset_dir)
    if recursive:
        hits = glob.glob(os.path.join(base, "**", pattern), recursive=True)
    else:
        hits = glob.glob(os.path.join(base, pattern))
    return sorted(p for p in hits if os.path.isfile(p))


def pick_best(runs: list[dict]) -> Optional[dict]:
    valid = [r for r in runs if r.get("ok")]
    if not valid:
        return None
    return sorted(valid,
                  key=lambda r: (r["colors"], r["total_exec_ms"]))[0]
```

- [ ] **Step 4: Run selftest to verify pass**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `SELFTEST: PASS (31/31 checks)`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sotas/sweep_gpu_sotas.py
git commit -m "scripts/run_sotas: tool filtering, dataset discovery, pick_best

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: argv construction + build-command construction

**Files:**
- Modify: `scripts/run_sotas/sweep_gpu_sotas.py`

- [ ] **Step 1: Add failing selftest cases**

Add and wire `selftest_commands(results)` (call after
`selftest_engine(results)`):

```python
def selftest_commands(results: list) -> None:
    by_name = {t["name"]: t for t in REGISTRY}

    _check(results, "argv csrcolor",
           build_argv(by_name["csrcolor"], "/b/csr", "/g/x.egr"),
           ["/b/csr", "/g/x.egr"])
    _check(results, "argv Picasso",
           build_argv(by_name["Picasso"], "/b/p", "/g/x.egr"),
           ["/b/p", "--in", "/g/x.egr", "--target", "16", "--recurse",
            "--order", "LIST", "--check"])
    _check(results, "argv kokkos_VBBIT",
           build_argv(by_name["kokkos_VBBIT"], "/b/k", "/g/x.egr"),
           ["/b/k", "--cuda", "0", "--amtx", "/g/x.egr", "--algorithm",
            "COLORING_VBBIT", "--repeat", "1"])
    _check(results, "argv cuSL",
           build_argv(by_name["cuSL"], "/b/jp", "/g/x.egr"),
           ["/b/jp", "-f", "/g/x.egr", "-a", "cuSL"])

    eg = build_unit_steps("ecl-gc", "89")
    _check(results, "ecl-gc steps len", len(eg), 1)
    _check(results, "ecl-gc cwd", eg[0]["cwd"], "External/ECL-GC")
    _check(results, "ecl-gc cmd", eg[0]["cmd"],
           ["nvcc", "-O3", "-std=c++17", "-arch=sm_89",
            "ECL-GC_12.cu", "-o", "ecl-gc"])

    pc = build_unit_steps("picasso", "90")
    _check(results, "picasso cmake arch flag",
           "-DCMAKE_CUDA_ARCHITECTURES=90" in pc[1]["cmd"], True)

    cs = build_unit_steps("csrcolor", "86")
    _check(results, "csrcolor clean ignore_fail",
           cs[0]["ignore_fail"], True)
    _check(results, "csrcolor make arch",
           "COMPUTECAPABILITY=sm_86" in cs[1]["cmd"], True)

    pg = build_unit_steps("pgc", "89")
    _check(results, "pgc retry present", pg[0].get("retry") is not None,
           True)
    _check(results, "pgc retry drops -arch",
           any(a.startswith("-arch=") for a in pg[0]["retry"]), False)

    ko = build_unit_steps("kokkos", "89")
    _check(results, "kokkos ignores arch (no sm_ in any cmd)",
           any("sm_89" in " ".join(s["cmd"]) for s in ko), False)
```

- [ ] **Step 2: Run selftest to verify it fails**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `NameError: name 'build_argv' is not defined`.

- [ ] **Step 3: Implement argv + build-command construction**

Add (argv + build-commands section). A "step" is
`{"cmd": [...], "cwd": <relpath>, "ignore_fail": bool,
"retry": Optional[list]}`. `cwd` is relative to `REPO_ROOT`.

```python
def build_argv(tool: dict, binary_abs: str, graph_abs: str) -> list[str]:
    return [binary_abs] + [graph_abs if a == "{G}" else a
                           for a in tool["argv"]]


def build_unit_steps(unit: str, nn: str) -> list[dict]:
    """Ordered build steps for a unit. nn = numeric arch (e.g. '89')."""
    sm = f"sm_{nn}"
    if unit == "csrcolor":
        d = "External/csrcolor/src/csrcolor"
        return [
            {"cmd": ["make", "-C", d, "clean"], "cwd": ".",
             "ignore_fail": True, "retry": None},
            {"cmd": ["make", "-C", d, f"COMPUTECAPABILITY={sm}"],
             "cwd": ".", "ignore_fail": False, "retry": None},
        ]
    if unit == "csrcolor_data":
        d = "External/csrcolor/src/data"
        return [
            {"cmd": ["make", "-C", d, "clean"], "cwd": ".",
             "ignore_fail": True, "retry": None},
            {"cmd": ["make", "-C", d, f"COMPUTECAPABILITY={sm}"],
             "cwd": ".", "ignore_fail": False, "retry": None},
        ]
    if unit == "kokkos":
        b = "External/kokkos-kernels/build"
        kk = "/home/chsieh45/local/kokkos-cuda"
        return [
            {"cmd": ["rm", "-rf", b], "cwd": ".",
             "ignore_fail": True, "retry": None},
            {"cmd": ["cmake", "-S", "External/kokkos-kernels", "-B", b,
                     f"-DCMAKE_CXX_COMPILER={kk}/bin/nvcc_wrapper",
                     f"-DKokkos_ROOT={kk}",
                     "-DKokkosKernels_ENABLE_PERFTESTS=ON",
                     "-DCMAKE_BUILD_TYPE=Release"],
             "cwd": ".", "ignore_fail": False, "retry": None},
            {"cmd": ["cmake", "--build", b, "--target", "graph_color",
                     "-j"], "cwd": ".", "ignore_fail": False,
             "retry": None},
        ]
    if unit == "pgc":
        return [
            {"cmd": ["nvcc", "-O3", "-std=c++14", f"-arch={sm}",
                     "parallel.cu", "-o", "pgc_parallel"],
             "cwd": "External/Parallel-Graph-Colouring",
             "ignore_fail": False,
             "retry": ["nvcc", "-O3", "-std=c++14", "parallel.cu",
                       "-o", "pgc_parallel"]},
        ]
    if unit == "picasso":
        b = "External/Picasso/build"
        return [
            {"cmd": ["rm", "-rf", b], "cwd": ".",
             "ignore_fail": True, "retry": None},
            {"cmd": ["cmake", "-S", "External/Picasso", "-B", b,
                     f"-DCMAKE_CUDA_ARCHITECTURES={nn}",
                     "-DCMAKE_BUILD_TYPE=Release"],
             "cwd": ".", "ignore_fail": False, "retry": None},
            {"cmd": ["cmake", "--build", b, "--target", "palcolEgrG",
                     "-j"], "cwd": ".", "ignore_fail": False,
             "retry": ["cmake", "--build", b, "-j"]},
        ]
    if unit == "ecl-gc":
        return [
            {"cmd": ["nvcc", "-O3", "-std=c++17", f"-arch={sm}",
                     "ECL-GC_12.cu", "-o", "ecl-gc"],
             "cwd": "External/ECL-GC", "ignore_fail": False,
             "retry": None},
        ]
    if unit == "ecl-gc-r":
        return [
            {"cmd": ["nvcc", "-O3", "-std=c++17", f"-arch={sm}",
                     "ECL-GC-ColorReduction_12.cu", "-o", "ecl-gc-r"],
             "cwd": "External/ECL-GC", "ignore_fail": False,
             "retry": None},
        ]
    if unit == "jp-series":
        return [
            {"cmd": ["make", "-C", "JP-Series", "clean"], "cwd": ".",
             "ignore_fail": True, "retry": None},
            {"cmd": ["make", "-C", "JP-Series", f"ARCH={sm}"],
             "cwd": ".", "ignore_fail": False, "retry": None},
        ]
    raise ValueError(f"unknown build unit: {unit}")
```

- [ ] **Step 4: Run selftest to verify pass**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `SELFTEST: PASS (44/44 checks)`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sotas/sweep_gpu_sotas.py
git commit -m "scripts/run_sotas: argv + per-unit build command construction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Build phase + single-invocation runner + sweep loop

**Files:**
- Modify: `scripts/run_sotas/sweep_gpu_sotas.py`

- [ ] **Step 1: Add failing selftest cases**

Add and wire `selftest_run(results)` (call after
`selftest_commands(results)`). This injects fake stdout via a stub to test
result assembly without a GPU:

```python
def selftest_run(results: list) -> None:
    by_name = {t["name"]: t for t in REGISTRY}

    good = ("runtime:    3.500000 ms\ncolors used: 40\n")
    r = assemble_run(by_name["csrcolor"], 0, good, None)
    _check(results, "assemble ok", r["ok"], True)
    _check(results, "assemble colors", r["colors"], 40)
    _check(results, "assemble ms", r["total_exec_ms"], 3.5)
    _check(results, "assemble err none", r["error"], None)

    r2 = assemble_run(by_name["csrcolor"], 1, "boom\n", None)
    _check(results, "assemble nonzero rc not ok", r2["ok"], False)
    _check(results, "assemble nonzero rc err",
           "exit code 1" in r2["error"], True)

    r3 = assemble_run(by_name["csrcolor"], 0, "no metrics here", None)
    _check(results, "assemble unparseable not ok", r3["ok"], False)

    r4 = assemble_run(by_name["csrcolor"], None, "", "timeout")
    _check(results, "assemble timeout flagged",
           r4["error"], "timeout")
```

- [ ] **Step 2: Run selftest to verify it fails**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `NameError: name 'assemble_run' is not defined`.

- [ ] **Step 3: Implement build phase, runner, sweep**

Add (build/sweep section):

```python
def _run_cmd(cmd: list[str], cwd_rel: str,
             timeout: int) -> tuple[int, str]:
    cwd = os.path.join(REPO_ROOT, cwd_rel)
    try:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               timeout=timeout, check=False)
        return proc.returncode, proc.stdout
    except subprocess.TimeoutExpired:
        return 124, f"(timeout after {timeout}s)"
    except OSError as e:
        return 127, f"(exec error: {e})"


def build_one_unit(unit: str, nn: str) -> dict:
    start = time.perf_counter()
    last_cmd = ""
    for step in build_unit_steps(unit, nn):
        last_cmd = " ".join(step["cmd"])
        rc, out = _run_cmd(step["cmd"], step["cwd"], BUILD_TIMEOUT_SEC)
        if rc != 0 and not step["ignore_fail"]:
            if step.get("retry"):
                last_cmd = " ".join(step["retry"])
                rc, out = _run_cmd(step["retry"], step["cwd"],
                                   BUILD_TIMEOUT_SEC)
            if rc != 0:
                return {"unit": unit, "ok": False, "cmd": last_cmd,
                        "seconds": round(time.perf_counter() - start, 2),
                        "error": out.strip()[-800:]}
    return {"unit": unit, "ok": True, "cmd": last_cmd,
            "seconds": round(time.perf_counter() - start, 2),
            "error": None}


def run_build_phase(units: set[str], nn: str) -> dict:
    """unit -> build result dict. Built in a stable order."""
    order = ["csrcolor", "csrcolor_data", "kokkos", "pgc", "picasso",
             "ecl-gc", "ecl-gc-r", "jp-series"]
    out = {}
    for unit in order:
        if unit in units:
            print(f"[build] {unit} ...", flush=True)
            res = build_one_unit(unit, nn)
            print(f"[build] {unit}: "
                  f"{'OK' if res['ok'] else 'FAIL'} "
                  f"({res['seconds']}s)", flush=True)
            out[unit] = res
    return out


def assemble_run(tool: dict, returncode: Optional[int], stdout: str,
                 timeout_err: Optional[str]) -> dict:
    if timeout_err is not None:
        return {"ok": False, "total_exec_ms": None, "colors": None,
                "returncode": returncode, "error": timeout_err}
    if returncode != 0:
        return {"ok": False, "total_exec_ms": None, "colors": None,
                "returncode": returncode,
                "error": f"exit code {returncode}: "
                         f"{stdout.strip()[-400:]}"}
    colors = parse_colors(tool, stdout)
    ms = parse_time_ms(tool, stdout)
    if colors is None or ms is None:
        return {"ok": False, "total_exec_ms": ms, "colors": colors,
                "returncode": returncode,
                "error": "could not parse colors/time from output"}
    return {"ok": True, "total_exec_ms": ms, "colors": colors,
            "returncode": returncode, "error": None}


def run_cell(tool: dict, binary_abs: str, graph_abs: str, runs: int,
             timeout: int) -> list[dict]:
    out = []
    for _ in range(runs):
        argv = build_argv(tool, binary_abs, graph_abs)
        try:
            proc = subprocess.run(argv, cwd=REPO_ROOT,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   timeout=timeout, check=False)
            out.append(assemble_run(tool, proc.returncode,
                                    proc.stdout, None))
        except subprocess.TimeoutExpired:
            out.append(assemble_run(tool, None, "",
                                    f"timeout after {timeout}s"))
            break  # skip remaining runs for this cell on timeout
        except OSError as e:
            out.append(assemble_run(tool, 127, "", f"exec error: {e}"))
            break
    return out
```

- [ ] **Step 4: Run selftest to verify pass**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `SELFTEST: PASS (52/52 checks)`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sotas/sweep_gpu_sotas.py
git commit -m "scripts/run_sotas: build phase, single-run assembly, cell runner

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: JSON assembly + atomic write + stdout summary

**Files:**
- Modify: `scripts/run_sotas/sweep_gpu_sotas.py`

- [ ] **Step 1: Add failing selftest cases**

Add and wire `selftest_json(results)` (call after `selftest_run(results)`):

```python
def selftest_json(results: list) -> None:
    tools = select_tools("cuSL,ECL-GC-R", None)
    builds = {"jp-series": {"unit": "jp-series", "ok": True,
                            "cmd": "make ...", "seconds": 9.0,
                            "error": None},
              "ecl-gc-r": {"unit": "ecl-gc-r", "ok": False,
                           "cmd": "nvcc ...", "seconds": 1.0,
                           "error": "boom"}}
    avail = compute_availability(tools, builds, skip_build=False)
    _check(results, "cuSL available", avail["cuSL"][0], True)
    _check(results, "ECL-GC-R unavailable (build fail)",
           avail["ECL-GC-R"][0], False)

    doc = build_json_doc(
        config={"runs_per_cell": 1}, builds=builds, tools=tools,
        availability=avail, datasets=["g1.egr"],
        rows=[{"tool": "cuSL", "dataset": "g1.egr",
               "dataset_path": "/d/g1.egr", "ok": True,
               "best_total_exec_ms": 5.0, "best_colors": 7,
               "runs": [], "error": None}])
    _check(results, "json top keys",
           sorted(doc.keys()),
           ["builds", "config", "datasets", "rows", "tools"])
    _check(results, "json builds is list", isinstance(doc["builds"],
           list), True)
    _check(results, "json tool meta has available",
           doc["tools"][0]["available"] in (True, False), True)
    _check(results, "json no ca/pa keys",
           any(k in doc["rows"][0] for k in ("ca_ms", "pa_ms")), False)
```

- [ ] **Step 2: Run selftest to verify it fails**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `NameError: name 'compute_availability' is not defined`.

- [ ] **Step 3: Implement availability, JSON doc, atomic write, summary**

Add (json/summary section):

```python
def compute_availability(tools: list[dict], builds: dict,
                          skip_build: bool) -> dict:
    """name -> (available: bool, reason: Optional[str])."""
    out = {}
    for t in tools:
        binary_abs = os.path.join(REPO_ROOT, t["binrel"])
        if not skip_build:
            b = builds.get(t["unit"])
            if b is not None and not b["ok"]:
                out[t["name"]] = (False,
                                  f"build failed ({t['unit']})")
                continue
        if not os.path.isfile(binary_abs):
            out[t["name"]] = (False, f"binary missing: {t['binrel']}")
            continue
        out[t["name"]] = (True, None)
    return out


def build_json_doc(config: dict, builds: dict, tools: list[dict],
                   availability: dict, datasets: list[str],
                   rows: list[dict]) -> dict:
    tool_meta = []
    for t in tools:
        avail, reason = availability.get(t["name"], (False, "unknown"))
        tool_meta.append({
            "name": t["name"], "kind": t["kind"],
            "build_unit": t["unit"],
            "binary": os.path.join(REPO_ROOT, t["binrel"]),
            "algorithm": t["algo"], "time_unit_src": t["usrc"],
            "available": avail, "unavailable_reason": reason,
        })
    return {
        "config": config,
        "builds": list(builds.values()),
        "tools": tool_meta,
        "datasets": datasets,
        "rows": rows,
    }


def write_json_atomic(path: str, doc: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def print_summary(doc: dict) -> None:
    print("\n=== build summary ===")
    for b in doc["builds"]:
        print(f"  {b['unit']:<14} {'OK' if b['ok'] else 'FAIL':<5} "
              f"{b['seconds']}s")
    if not doc["builds"]:
        print("  (skipped --skip-build)")
    print("=== sweep summary (ok cells / total) ===")
    n_ds = len(doc["datasets"]) or 1
    for tm in doc["tools"]:
        ok = sum(1 for r in doc["rows"]
                 if r["tool"] == tm["name"] and r["ok"])
        flag = "" if tm["available"] else \
            f"  [unavailable: {tm['unavailable_reason']}]"
        print(f"  {tm['name']:<14} {ok}/{n_ds}{flag}")
```

- [ ] **Step 4: Run selftest to verify pass**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `SELFTEST: PASS (58/58 checks)`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sotas/sweep_gpu_sotas.py
git commit -m "scripts/run_sotas: availability, JSON doc, atomic write, summary

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wire `run_sweep` end-to-end

**Files:**
- Modify: `scripts/run_sotas/sweep_gpu_sotas.py`

- [ ] **Step 1: Implement `run_sweep`**

Replace `def run_sweep(args): raise NotImplementedError` with:

```python
def run_sweep(args: argparse.Namespace) -> int:
    if not args.dataset_dir:
        sys.stderr.write("error: --dataset-dir is required\n")
        return 2
    if not os.path.isdir(args.dataset_dir):
        sys.stderr.write(
            f"error: --dataset-dir not a directory: {args.dataset_dir}\n")
        return 2
    datasets = discover_datasets(args.dataset_dir, args.pattern,
                                 args.recursive)
    if not datasets:
        sys.stderr.write(
            f"error: no files matching {args.pattern!r} in "
            f"{args.dataset_dir}\n")
        return 2

    nn, arch_source = resolve_arch(args.arch)
    tools = select_tools(args.only, args.exclude)
    units = needed_units(tools)

    builds: dict = {}
    if not args.skip_build:
        builds = run_build_phase(units, nn)

    availability = compute_availability(tools, builds, args.skip_build)

    rows: list[dict] = []
    for t in tools:
        binary_abs = os.path.join(REPO_ROOT, t["binrel"])
        avail, reason = availability[t["name"]]
        for ds in datasets:
            name = os.path.basename(ds)
            if not avail:
                rows.append({
                    "tool": t["name"], "dataset": name,
                    "dataset_path": ds, "ok": False,
                    "best_total_exec_ms": None, "best_colors": None,
                    "runs": [], "error": reason})
                continue
            print(f"[run] {t['name']} :: {name}", flush=True)
            cell = run_cell(t, binary_abs, ds, args.runs, args.timeout)
            best = pick_best(cell)
            rows.append({
                "tool": t["name"], "dataset": name,
                "dataset_path": ds,
                "ok": best is not None,
                "best_total_exec_ms":
                    best["total_exec_ms"] if best else None,
                "best_colors": best["colors"] if best else None,
                "runs": cell,
                "error": None if best else
                         (cell[-1]["error"] if cell else "no runs"),
            })

    config = {
        "timestamp": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": REPO_ROOT,
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "pattern": args.pattern,
        "recursive": args.recursive,
        "runs_per_cell": args.runs,
        "timeout_sec": args.timeout,
        "arch": int(nn) if nn.isdigit() else nn,
        "arch_source": arch_source,
        "skip_build": args.skip_build,
        "tools": [t["name"] for t in tools],
    }
    doc = build_json_doc(config, builds, tools, availability,
                         [os.path.basename(d) for d in datasets], rows)
    write_json_atomic(args.out, doc)
    print_summary(doc)
    print(f"\nwrote {args.out}")
    return 0
```

- [ ] **Step 2: Verify selftest still passes (no regressions)**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest`
Expected: `SELFTEST: PASS (58/58 checks)`.

- [ ] **Step 3: Verify argument validation works without a GPU**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py`
Expected: `error: --dataset-dir is required`; exit code 2.

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --dataset-dir /nonexistent`
Expected: `error: --dataset-dir not a directory: /nonexistent`; exit 2.

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --dataset-dir Datasets/test --only NoTool --skip-build`
Expected: `error: unknown tool name 'NoTool'`; exit 2.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_sotas/sweep_gpu_sotas.py
git commit -m "scripts/run_sotas: wire run_sweep end-to-end

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Acceptance — build fresh + sweep `Datasets/test`

**Files:** none (validation only). Per `CLAUDE.md` there is no unit-test
framework — integration is validated via program output (spec §11).

- [ ] **Step 1: Skip-build smoke (fast, reuses any existing binaries)**

Run:
```bash
python3 scripts/run_sotas/sweep_gpu_sotas.py \
    --dataset-dir Datasets/test --skip-build \
    --out /tmp/gpu_sotas_smoke.json
```
Expected: completes; prints build summary `(skipped --skip-build)`, a sweep
summary, and `wrote /tmp/gpu_sotas_smoke.json`. Tools whose binaries do not
yet exist appear as `[unavailable: binary missing: ...]` — that is fine here.

- [ ] **Step 2: Inspect the smoke JSON shape**

Run:
```bash
python3 -c "import json;d=json.load(open('/tmp/gpu_sotas_smoke.json'));\
print(sorted(d.keys()));\
print(d['config']['arch'], d['config']['arch_source']);\
print(len(d['rows']),'rows', len(d['tools']),'tools');\
print([ (r['tool'],r['dataset'],r['ok'],r['best_colors'],\
r['best_total_exec_ms']) for r in d['rows'][:6] ])"
```
Expected: keys `['builds','config','datasets','rows','tools']`; `rows`
count = `tools × datasets`; no `ca_ms`/`pa_ms` anywhere.

- [ ] **Step 3: Full build-all-fresh sweep on the smoke set**

Run (this rebuilds all 8 units; kokkos + picasso take minutes):
```bash
python3 scripts/run_sotas/sweep_gpu_sotas.py \
    --dataset-dir Datasets/test \
    --out /tmp/gpu_sotas_full.json
```
Expected: per-unit `[build] <unit>: OK/FAIL (Ns)` lines; then `[run]
<tool> :: facebook.egr` / `youtube.egr` lines; final summary; exit 0 even
if some units FAIL (graceful degradation).

- [ ] **Step 4: Validate acceptance criteria (spec §11)**

Run:
```bash
python3 -c "
import json
d=json.load(open('/tmp/gpu_sotas_full.json'))
avail={t['name'] for t in d['tools'] if t['available']}
bad=[]
for r in d['rows']:
    if r['tool'] in avail and not (r['ok'] and r['best_colors']>0
            and r['best_total_exec_ms']>0):
        bad.append((r['tool'],r['dataset'],r.get('error')))
print('available tools:',sorted(avail))
print('FAILED available cells:',bad if bad else 'NONE')
import collections
g=collections.defaultdict(dict)
for r in d['rows']: g[r['dataset']][r['tool']]=(r['best_colors'],
    r['best_total_exec_ms'])
for ds,row in g.items():
    if 'ECL-GC' in row and 'ECL-GC-R' in row and None not in row['ECL-GC']\
       and None not in row['ECL-GC-R']:
        print(ds,'ECL-GC',row['ECL-GC'],'ECL-GC-R',row['ECL-GC-R'])
"
```
Expected: every **available** tool has `best_colors>0` and
`best_total_exec_ms>0` on both `facebook.egr` and `youtube.egr`
(`FAILED available cells: NONE`). Spot-check: `ECL-GC-R` colors ≤ `ECL-GC`
colors and `ECL-GC-R` time ≥ `ECL-GC` time; `Picasso` `best_colors` is a
small positive int (final colors, not 16).

- [ ] **Step 5: Triage any FAIL, then re-validate**

If a build unit FAILs: read `builds[].error` in the JSON for that unit. Known
risks (spec §4): `kokkos` needs `/home/chsieh45/local/kokkos-cuda` (arch
fixed sm_86 — not overridden); `picasso` needs CMake + CUDA≥11 + OpenMP; a
failing unit only disables its own tools and the sweep still completes. If a
run FAILs for an *available* tool, inspect `rows[].runs[].error` and the
tool's regex in `REGISTRY` against the actual binary stdout
(`<binary> <Datasets/test/facebook.egr>`); fix the regex, re-run
`--selftest` (add a golden case for the real output), then repeat Step 3–4.
Do not mark this task complete while any available-tool cell fails on the
smoke set.

- [ ] **Step 6: Commit (script only; results are not committed)**

```bash
git add scripts/run_sotas/sweep_gpu_sotas.py scripts/run_sotas/README.md
git commit -m "scripts/run_sotas: finalize sweep_gpu_sotas.py (acceptance passed)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" \
  --allow-empty
```

---

## Self-Review (filled by plan author)

**Spec coverage:**
- §3 12-tool registry → Task 3 (`REGISTRY`, parsers) ✔
- §3 Picasso/ECL-GC/ECL-GC-R parsing specifics → Task 3 golden cases ✔
- §4 build-all-fresh, 8 units, numeric arch derivation, retries,
  kokkos-no-override, csrcolor-subdirs-only → Task 2 (arch), Task 5
  (`build_unit_steps`), Task 6 (`build_one_unit`) ✔
- §5 CLI flags (incl. `--arch NN`, `--skip-build`, `--only/--exclude`,
  required `--dataset-dir`) → Task 1 (`build_arg_parser`), Task 8
  (validation) ✔
- §6 data flow / timeout-skip → Task 6 (`run_cell` break-on-timeout),
  Task 8 ✔
- §7 pick_best (colors then ms) → Task 4 ✔
- §8 JSON schema (config/builds/tools/datasets/rows, no ca/pa) → Task 7 ✔
- §9 error handling / exit 0 with result file → Task 6/8 (graceful),
  Task 8 returns 0 after writing ✔
- §10 file layout (2 files) → Task 1 ✔
- §11 acceptance on Datasets/test + sanity checks → Task 9 ✔

**Placeholder scan:** no TBD/TODO; every code step shows complete code;
every command has an expected result. ✔

**Type consistency:** run-result dict keys
(`ok/total_exec_ms/colors/returncode/error`) consistent across
`assemble_run`/`run_cell`/`pick_best`/`run_sweep`; build-step dict keys
(`cmd/cwd/ignore_fail/retry`) consistent across `build_unit_steps`/
`build_one_unit`; tool dict keys (`name/kind/unit/binrel/argv/colors/time/
algo/usrc`) consistent across `REGISTRY`/parsers/`build_argv`/
`build_json_doc`. Selftest counts are cumulative (5→21→31→44→52→58) and
match each task's expected line. ✔
```

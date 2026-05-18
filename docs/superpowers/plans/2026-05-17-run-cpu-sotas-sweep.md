# CPU SOTA Coloring Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/run_sotas/sweep_cpu_sotas.py`, a registry-driven sweep
(sibling of `sweep_gpu_sotas.py`) over the 6 CPU coloring configs in
`CPU/Sequential/` + `CPU/Parallel/`, writing one JSON of total exec time
(ms, excl. graph load) + color count per `(tool, dataset)`.

**Architecture:** One new self-contained file mirroring the proven GPU-sweep
engine, minus GPU concerns (`--arch`, kokkos, PRE_MODEL) plus a `--threads`
flag injected into the 3 parallel configs. 6 declarative `REGISTRY` entries,
2 `make`-based build units, the JP-family stdout parsers, identical JSON
schema (`threads` replaces `arch`), `--selftest` as the test mechanism.

**Tech Stack:** Python 3 stdlib only. Spec:
`docs/superpowers/specs/2026-05-17-run-cpu-sotas-sweep-design.md`. Commands
run from repo root `/home/chsieh45/PunchShadow/CHROMA`.

---

## File Structure

- **Create** `scripts/run_sotas/sweep_cpu_sotas.py` — the entire CPU driver
  (single file; sections: imports/constants → threads resolver → parsers +
  REGISTRY → discovery/pick_best → argv + build commands → build phase →
  sweep → json + summary → cli/main → selftest → `__main__`).
- **Modify** `scripts/run_sotas/README.md` — append a "## CPU sweep" section.

`sweep_gpu_sotas.py` is **not modified**. Single-file per the `scripts/`
convention and the GPU precedent.

**Concurrency note (unchanged from the GPU work):** a parallel user session
also commits to this repo. NEVER `git add -A`/`.`/`-u`, NEVER
`git commit -a`/`-am`/`--amend`. Only `git add` the exact files and commit
with an explicit `-- <paths>` pathspec; verify with
`git show --stat --oneline <sha>` (locate your sha via `git log --oneline -3`
— HEAD may be a concurrent commit).

---

## Task 1: Create `sweep_cpu_sotas.py` + README section

**Files:**
- Create: `scripts/run_sotas/sweep_cpu_sotas.py`
- Modify: `scripts/run_sotas/README.md`

- [ ] **Step 1: Create the script (complete content)**

Create `scripts/run_sotas/sweep_cpu_sotas.py` with EXACTLY this content:

```python
#!/usr/bin/env python3
"""Build-and-sweep 6 CPU graph-coloring configs over a directory of .egr graphs.

Records total execution time (excluding graph loading) and color count per
(tool, dataset) into one JSON. Sibling of sweep_gpu_sotas.py. See
docs/superpowers/specs/2026-05-17-run-cpu-sotas-sweep-design.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional

# Repo root = two levels up from this file (scripts/run_sotas/ -> ../../).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BUILD_TIMEOUT_SEC = 3600  # hard cap per build-unit (make can run minutes)


def resolve_threads(cli_value: Optional[int]) -> int:
    """--threads if >0, else all logical cores (os.cpu_count(), >=1)."""
    if cli_value and cli_value > 0:
        return cli_value
    return os.cpu_count() or 1


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
    if spec[0] == "ms":
        return _find_float(spec[1], text)
    raise ValueError(f"bad time spec: {spec!r}")


# Verified: all 6 CPU binaries print `runtime: %.6f ms` and
# `colors used: %d`, timer started after readECLgraph (graph load excluded).
_MS = r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms"
_COLORS_USED = r"colors\s+used:\s*(\d+)"

REGISTRY: list[dict] = [
    dict(name="DSatur", kind="cpu", unit="cpu_sequential",
         binrel="CPU/Sequential/cpu_Dstura", argv=["{G}"],
         colors=_COLORS_USED, time=("ms", _MS), algo="DSatur", usrc="ms"),
    dict(name="Greedy", kind="cpu", unit="cpu_sequential",
         binrel="CPU/Sequential/cpu_greedy", argv=["{G}"],
         colors=_COLORS_USED, time=("ms", _MS), algo="greedy", usrc="ms"),
    dict(name="JP-SL^M", kind="cpu", unit="cpu_sequential",
         binrel="CPU/Sequential/cpu_SDL", argv=["{G}"],
         colors=_COLORS_USED, time=("ms", _MS), algo="SDL", usrc="ms"),
    dict(name="JP-SL^A", kind="cpu", unit="cpu_parallel",
         binrel="CPU/Parallel/cpu_SL", argv=["{G}", "{T}"],
         colors=_COLORS_USED, time=("ms", _MS), algo="SL", usrc="ms"),
    dict(name="ADG", kind="cpu", unit="cpu_parallel",
         binrel="CPU/Parallel/cpu_ADG", argv=["{G}", "{T}"],
         colors=_COLORS_USED, time=("ms", _MS), algo="ADG", usrc="ms"),
    dict(name="SLL", kind="cpu", unit="cpu_parallel",
         binrel="CPU/Parallel/cpu_SLL", argv=["{G}", "{T}"],
         colors=_COLORS_USED, time=("ms", _MS), algo="SLL", usrc="ms"),
]


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
    """Best run by (colors ASC, total_exec_ms ASC), or None if no ok run.

    Contract: every dict with ok=True MUST have numeric "colors" and
    "total_exec_ms" (guaranteed by assemble_run, which only sets ok=True
    when both parsed non-None).
    """
    valid = [r for r in runs if r.get("ok")]
    if not valid:
        return None
    return sorted(valid,
                  key=lambda r: (r["colors"], r["total_exec_ms"]))[0]


def build_argv(tool: dict, binary_abs: str, graph_abs: str,
               threads: int) -> list[str]:
    out = [binary_abs]
    for a in tool["argv"]:
        if a == "{G}":
            out.append(graph_abs)
        elif a == "{T}":
            out.append(str(threads))
        else:
            out.append(a)
    return out


def build_unit_steps(unit: str) -> list[dict]:
    """Ordered build steps for a CPU build unit (g++ via make; no arch)."""
    if unit == "cpu_sequential":
        d = "CPU/Sequential"
        return [
            {"cmd": ["make", "-C", d, "clean"], "cwd": ".",
             "ignore_fail": True, "retry": None},
            {"cmd": ["make", "-C", d], "cwd": ".",
             "ignore_fail": False, "retry": None},
        ]
    if unit == "cpu_parallel":
        d = "CPU/Parallel"
        return [
            {"cmd": ["make", "-C", d, "clean"], "cwd": ".",
             "ignore_fail": True, "retry": None},
            {"cmd": ["make", "-C", d], "cwd": ".",
             "ignore_fail": False, "retry": None},
        ]
    raise ValueError(f"unknown build unit: {unit}")


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


def build_one_unit(unit: str) -> dict:
    start = time.perf_counter()
    last_cmd = ""
    for step in build_unit_steps(unit):
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


def run_build_phase(units: set[str]) -> dict:
    """unit -> build result dict. Built in a stable order."""
    order = ["cpu_sequential", "cpu_parallel"]
    out = {}
    for unit in order:
        if unit in units:
            print(f"[build] {unit} ...", flush=True)
            res = build_one_unit(unit)
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
    if returncode is None:
        return {"ok": False, "total_exec_ms": None, "colors": None,
                "returncode": None, "error": "internal: no returncode"}
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
             timeout: int, threads: int) -> list[dict]:
    out = []
    for _ in range(runs):
        argv = build_argv(tool, binary_abs, graph_abs, threads)
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
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sweep_cpu_sotas.py",
        description="Build + sweep the 6 CPU SOTA coloring configs.",
    )
    p.add_argument("--dataset-dir", help="Directory of .egr graphs "
                   "(required unless --selftest).")
    p.add_argument("--pattern", default="*.egr", help="Glob (default *.egr).")
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into subdirectories.")
    p.add_argument("--runs", type=int, default=1,
                   help="Invocations per (tool, dataset) (default 1).")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-invocation timeout seconds (default 600).")
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "scripts", "run_sotas",
                                        "cpu_sotas_results.json"),
                   help="Output JSON path.")
    p.add_argument("--threads", type=int, default=None,
                   help="OpenMP threads for the 3 parallel configs "
                        "(default: all logical cores).")
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

    threads = resolve_threads(args.threads)
    tools = select_tools(args.only, args.exclude)
    units = needed_units(tools)

    builds: dict = {}
    if not args.skip_build:
        builds = run_build_phase(units)

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
            cell = run_cell(t, binary_abs, ds, args.runs, args.timeout,
                            threads)
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
        "threads": threads,
        "skip_build": args.skip_build,
        "tools": [t["name"] for t in tools],
    }
    doc = build_json_doc(config, builds, tools, availability,
                         [os.path.basename(d) for d in datasets], rows)
    write_json_atomic(args.out, doc)
    print_summary(doc)
    print(f"\nwrote {args.out}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.selftest:
        return run_selftest()
    return run_sweep(args)


def _check(results: list, name: str, got, expected) -> None:
    results.append((got == expected, name, got, expected))


def run_selftest() -> int:
    results: list = []
    selftest_parsers(results)
    selftest_engine(results)
    selftest_commands(results)
    selftest_run(results)
    selftest_json(results)
    failed = [r for r in results if not r[0]]
    for ok, name, got, expected in results:
        if not ok:
            print(f"FAIL {name}: got={got!r} expected={expected!r}")
    print(f"SELFTEST: {'PASS' if not failed else 'FAIL'} "
          f"({len(results) - len(failed)}/{len(results)} checks)")
    return 0 if not failed else 1


def selftest_parsers(results: list) -> None:
    by = {t["name"]: t for t in REGISTRY}
    s = "runtime:    3.500000 ms\ncolors used: 40\n"
    _check(results, "cpu colors", parse_colors(by["DSatur"], s), 40)
    _check(results, "cpu ms", parse_time_ms(by["DSatur"], s), 3.5)
    _check(results, "registry size", len(REGISTRY), 6)
    _check(results, "registry names unique",
           len({t["name"] for t in REGISTRY}), 6)


def selftest_engine(results: list) -> None:
    _check(results, "only filter",
           [t["name"] for t in select_tools("Greedy,DSatur", None)],
           ["DSatur", "Greedy"])
    _check(results, "exclude count",
           len(select_tools(None, "ADG,SLL")), 4)
    _check(results, "exclude removed", "ADG" in
           [t["name"] for t in select_tools(None, "ADG,SLL")], False)
    _check(results, "no filter = all",
           len(select_tools(None, None)), 6)
    raised = False
    try:
        select_tools("Nope", None)
    except SystemExit:
        raised = True
    _check(results, "unknown only errors", raised, True)
    _check(results, "needed units sequential",
           sorted(needed_units(
               select_tools("DSatur,Greedy,JP-SL^M", None))),
           ["cpu_sequential"])
    _check(results, "needed units parallel",
           sorted(needed_units(
               select_tools("JP-SL^A,ADG,SLL", None))),
           ["cpu_parallel"])
    runs = [
        {"ok": True, "colors": 30, "total_exec_ms": 9.0},
        {"ok": True, "colors": 28, "total_exec_ms": 12.0},
        {"ok": True, "colors": 28, "total_exec_ms": 11.0},
        {"ok": False, "colors": None, "total_exec_ms": None},
    ]
    b = pick_best(runs)
    _check(results, "pick_best colors", b["colors"], 28)
    _check(results, "pick_best tie->faster", b["total_exec_ms"], 11.0)
    _check(results, "pick_best none", pick_best(
        [{"ok": False, "colors": None, "total_exec_ms": None}]), None)


def selftest_commands(results: list) -> None:
    by = {t["name"]: t for t in REGISTRY}
    seq = build_unit_steps("cpu_sequential")
    _check(results, "cpu_seq steps len", len(seq), 2)
    _check(results, "cpu_seq clean ignore_fail",
           seq[0]["ignore_fail"], True)
    _check(results, "cpu_seq make cmd", seq[1]["cmd"],
           ["make", "-C", "CPU/Sequential"])
    _check(results, "cpu_par make cmd",
           build_unit_steps("cpu_parallel")[1]["cmd"],
           ["make", "-C", "CPU/Parallel"])
    raised = False
    try:
        build_unit_steps("zzz")
    except ValueError:
        raised = True
    _check(results, "unknown unit raises", raised, True)
    _check(results, "argv sequential",
           build_argv(by["DSatur"], "/b/d", "/g.egr", 8),
           ["/b/d", "/g.egr"])
    _check(results, "argv parallel injects threads",
           build_argv(by["JP-SL^A"], "/b/sl", "/g.egr", 8),
           ["/b/sl", "/g.egr", "8"])
    _check(results, "resolve_threads explicit",
           resolve_threads(8), 8)
    _check(results, "resolve_threads None -> cpu_count",
           resolve_threads(None), os.cpu_count() or 1)
    _check(results, "resolve_threads 0 -> cpu_count",
           resolve_threads(0), os.cpu_count() or 1)


def selftest_run(results: list) -> None:
    by = {t["name"]: t for t in REGISTRY}
    g = "runtime:    3.500000 ms\ncolors used: 40\n"
    r = assemble_run(by["DSatur"], 0, g, None)
    _check(results, "assemble ok", r["ok"], True)
    _check(results, "assemble colors", r["colors"], 40)
    _check(results, "assemble ms", r["total_exec_ms"], 3.5)
    _check(results, "assemble err none", r["error"], None)
    r2 = assemble_run(by["DSatur"], 1, "boom\n", None)
    _check(results, "assemble nonzero not ok", r2["ok"], False)
    _check(results, "assemble nonzero err",
           "exit code 1" in r2["error"], True)
    r3 = assemble_run(by["DSatur"], 0, "no metrics", None)
    _check(results, "assemble unparseable not ok", r3["ok"], False)
    r4 = assemble_run(by["DSatur"], None, "", "timeout")
    _check(results, "assemble timeout flagged", r4["error"], "timeout")
    r5 = assemble_run(by["DSatur"], None, "", None)
    _check(results, "assemble no-returncode guard",
           r5["error"], "internal: no returncode")


def selftest_json(results: list) -> None:
    tools = select_tools("DSatur,SLL", None)
    builds = {"cpu_sequential": {"unit": "cpu_sequential", "ok": False,
                                 "cmd": "make ...", "seconds": 1.0,
                                 "error": "boom"},
              "cpu_parallel": {"unit": "cpu_parallel", "ok": False,
                               "cmd": "make ...", "seconds": 1.0,
                               "error": "boom"}}
    avail = compute_availability(tools, builds, skip_build=False)
    _check(results, "DSatur unavailable (build fail)",
           avail["DSatur"][0], False)
    _check(results, "SLL unavailable (build fail)",
           avail["SLL"][0], False)
    doc = build_json_doc(
        config={"threads": 8}, builds=builds, tools=tools,
        availability=avail, datasets=["g1.egr"],
        rows=[{"tool": "DSatur", "dataset": "g1.egr",
               "dataset_path": "/d/g1.egr", "ok": True,
               "best_total_exec_ms": 5.0, "best_colors": 7,
               "runs": [], "error": None}])
    _check(results, "json top keys", sorted(doc.keys()),
           ["builds", "config", "datasets", "rows", "tools"])
    _check(results, "json builds is list",
           isinstance(doc["builds"], list), True)
    _check(results, "json tool kind cpu",
           doc["tools"][0]["kind"], "cpu")
    _check(results, "json no ca/pa keys",
           any(k in doc["rows"][0] for k in ("ca_ms", "pa_ms")), False)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the selftest passes (GREEN)**

Run: `python3 scripts/run_sotas/sweep_cpu_sotas.py --selftest; echo "exit=$?"`
Expected EXACTLY a line `SELFTEST: PASS (39/39 checks)` and `exit=0`.
(Counts: parsers 4 + engine 10 + commands 10 + run 9 + json 6 = 39.) If the
printed count is NOT 39, do NOT tweak code to force it — recount the
`_check` calls, report the true number, and STOP with status
DONE_WITH_CONCERNS so the plan annotation can be corrected (per the GPU
plan's count-correction precedent).

- [ ] **Step 3: Import + CLI sanity**

Run:
```bash
python3 -c "import ast; ast.parse(open('scripts/run_sotas/sweep_cpu_sotas.py').read()); print('ast OK')"
python3 scripts/run_sotas/sweep_cpu_sotas.py --help 2>&1 | grep -E -- '--threads|--dataset-dir|--arch' || true
python3 scripts/run_sotas/sweep_cpu_sotas.py 2>&1; echo "noargs exit=$?"
python3 -c "import importlib.util as u; s=u.spec_from_file_location('m','scripts/run_sotas/sweep_cpu_sotas.py'); m=u.module_from_spec(s); s.loader.exec_module(m); bn={t['name']:t for t in m.REGISTRY}; print(len(m.REGISTRY)); print(m.build_argv(bn['JP-SL^A'],'/b','/g.egr',16)); print(m.build_argv(bn['Greedy'],'/b','/g.egr',16)); print(sorted(m.needed_units(m.select_tools('DSatur,ADG',''))))"
```
Expected: `ast OK`; `--help` shows `--threads` and `--dataset-dir` and NO
`--arch`; no-args prints `error: --dataset-dir is required` and
`noargs exit=2`; then `6`, `['/b', '/g.egr', '16']` (parallel injects
threads), `['/b', '/g.egr']` (sequential, no threads), and
`['cpu_parallel', 'cpu_sequential']`.

- [ ] **Step 4: Append the README CPU section**

Append (to the END of) `scripts/run_sotas/README.md` exactly this block
(use an append — do not rely on a fragile mid-file anchor):

```markdown

## CPU sweep (`sweep_cpu_sotas.py`)

Sibling of `sweep_gpu_sotas.py` (same engine/JSON schema), independent of
it. Sweeps the 6 CPU coloring configs:

- Sequential (`CPU/Sequential/`, build unit `cpu_sequential`): `DSatur`
  (cpu_Dstura), `Greedy` (cpu_greedy), `JP-SL^M` (cpu_SDL).
- Parallel (`CPU/Parallel/`, build unit `cpu_parallel`, OpenMP): `JP-SL^A`
  (cpu_SL), `ADG` (cpu_ADG), `SLL` (cpu_SLL).

    python3 scripts/run_sotas/sweep_cpu_sotas.py --dataset-dir Datasets/test
    python3 scripts/run_sotas/sweep_cpu_sotas.py --dataset-dir Datasets/test --threads 16
    python3 scripts/run_sotas/sweep_cpu_sotas.py --selftest

`--threads N` is the OpenMP thread count passed to the 3 parallel configs
(default: all logical cores, `os.cpu_count()`); the 3 sequential configs
ignore it. No `--arch` (CPU/g++). `--dataset-dir` is required.
Build-all-fresh by default (`make -C CPU/Sequential` / `CPU/Parallel`);
`--skip-build` reuses existing binaries; any build/run failure is recorded
and the sweep continues. Default out:
`scripts/run_sotas/cpu_sotas_results.json`. `^` in `JP-SL^M`/`JP-SL^A` is a
valid JSON key and bash-safe in `--only`/`--exclude`.
```

A safe way to append exactly once:
```bash
printf '\n## CPU sweep (`sweep_cpu_sotas.py`)\n...' # (use the Edit/Write tool to append the block above verbatim; do NOT duplicate it if re-run)
```
(Implementer: append the markdown block above to the end of
`scripts/run_sotas/README.md` using the editing tool; verify with
`tail -25 scripts/run_sotas/README.md` that it appears exactly once.)

- [ ] **Step 5: Commit (SCOPED — concurrency-safe pathspec)**

```bash
git add scripts/run_sotas/sweep_cpu_sotas.py scripts/run_sotas/README.md
git commit -m "scripts/run_sotas: add sweep_cpu_sotas.py (6 CPU SOTA configs)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" -- scripts/run_sotas/sweep_cpu_sotas.py scripts/run_sotas/README.md
```
Then locate your commit (`git log --oneline -3`) and confirm
`git show --stat --oneline <sha>` lists ONLY those 2 files. Report
pre-commit `git diff --cached --name-status` (unrelated concurrent-session
files staged is expected and excluded by the pathspec). Branch unchanged.

---

## Task 2: Acceptance — build CPU units fresh + sweep `Datasets/test`

**Files:** none (validation only; DIAGNOSE-ONLY — report discrepancies, do
NOT edit code here; fixes route back through the implement→review loop).

- [ ] **Step 1: Selftest regression (no build needed)**

Run: `python3 scripts/run_sotas/sweep_cpu_sotas.py --selftest; echo $?`
Expected: `SELFTEST: PASS (39/39 checks)`, exit 0.

- [ ] **Step 2: Build both CPU units fresh + sweep all 6 on the smoke set**

Run (g++ builds are fast; no GPU needed):
```bash
cd /home/chsieh45/PunchShadow/CHROMA
timeout 1800 python3 scripts/run_sotas/sweep_cpu_sotas.py \
  --dataset-dir Datasets/test --threads 8 \
  --out /tmp/cpu_sotas.json > /tmp/cpu_sweep.log 2>&1; echo "EXIT=$?"
tail -30 /tmp/cpu_sweep.log
```
Expected: `[build] cpu_sequential: OK (Ns)`, `[build] cpu_parallel: OK
(Ns)`, then `[run] DSatur :: facebook.egr` … progress, a build+sweep
summary, exit 0. If a build unit FAILs, capture `builds[].error`.

- [ ] **Step 3: Validate acceptance (spec §9)**

```bash
python3 -c "
import json
d=json.load(open('/tmp/cpu_sotas.json'))
print('builds:', [(b['unit'],b['ok'],(b['error'] or '')[-200:]) for b in d['builds']])
print('avail:', [(t['name'],t['available'],t['unavailable_reason']) for t in d['tools']])
print('config.threads:', d['config']['threads'])
bad=[]
for r in d['rows']:
    if not (r['ok'] and r['best_colors'] and r['best_colors']>0 and r['best_total_exec_ms'] and r['best_total_exec_ms']>0):
        bad.append((r['tool'],r['dataset'],r.get('error')))
print('FAILED CELLS:', bad if bad else 'NONE')
for r in d['rows']: print(r['tool'], r['dataset'], r['best_colors'], r['best_total_exec_ms'])
"
```
Acceptance: both build units OK; all 6 tools `available`; `config.threads`
== 8; `FAILED CELLS: NONE` (every cell on facebook.egr + youtube.egr has
`best_colors>0` and `best_total_exec_ms>0`); script exited 0.

- [ ] **Step 4: Spot-check parallel threads + time source**

```bash
cd /home/chsieh45/PunchShadow/CHROMA
CPU/Parallel/cpu_SL Datasets/test/facebook.egr 8 2>&1 | grep -E 'runtime:|colors used:'
```
Confirm the JSON `JP-SL^A`/`facebook.egr` `best_colors`/`best_total_exec_ms`
correspond to the manual `colors used:` / `runtime: ms`, and that the
parallel binary accepted the thread argument (no usage error). Report raw
lines vs JSON. Diagnose-only.

- [ ] **Step 5: Triage / report**

If a build unit failed: paste `builds[].error`, judge env (g++/OpenMP
toolchain) vs a bug in the `cpu_*` build step. If an available tool's cell
failed: paste exact argv + one manual run's real stdout vs the recorded
error. `git log --oneline -3` + `git branch --show-current` (report only).
Bottom line: did acceptance PASS (both units built, 6 tools available &
valid on both graphs, exit 0, threads honored)?

---

## Self-Review (filled by plan author)

**Spec coverage:**
- §2 6 configs/mapping/order/kind=cpu/argv `{G}`/`{T}` → Task 1 Step 1
  `REGISTRY` ✔
- §3 reuse JP-family parsers (`_MS`/`_COLORS_USED`, `("ms",..)`) → Step 1 ✔
- §4 2 build units (`make -C CPU/Sequential` / `CPU/Parallel`, clean
  ignore_fail) + graceful → `build_unit_steps`/`build_one_unit`/
  `compute_availability` ✔
- §5 CLI (no `--arch`/`--kokkos-root`; `--threads` default os.cpu_count;
  `--dataset-dir` required) → `build_arg_parser`/`resolve_threads`/
  `run_sweep` ✔
- §6 engine + JSON schema identical, `threads` replaces arch, no ca/pa,
  exit codes → `run_sweep`/`build_json_doc`; Step 3 sanity ✔
- §7 selftest mirrors GPU, pinned `39/39` + stop-if-mismatch → Step 1
  `selftest_*` + Step 2 guard ✔
- §8 README CPU section → Task 1 Step 4 ✔
- §9 acceptance (build fresh + 6-tool sweep + colors>0/time>0 + threads +
  Total-time) → Task 2 ✔

**Placeholder scan:** complete file given verbatim; every step has exact
command + expected output; the only non-literals (`<sha>`, `N`s in build
seconds) are runtime values, not plan gaps. ✔

**Type consistency:** tool dict keys
(`name/kind/unit/binrel/argv/colors/time/algo/usrc`) consistent across
`REGISTRY`/parsers/`build_argv`/`build_json_doc`; `build_argv` takes
`threads`; `run_cell` takes `threads` and passes it to `build_argv`;
`build_unit_steps(unit)`/`build_one_unit(unit)`/`run_build_phase(units)`
have no `nn` (CPU, no arch); `run_sweep` resolves `threads` and records
`config.threads`; selftest count 4+10+10+9+6 = 39 matches Step 2. ✔

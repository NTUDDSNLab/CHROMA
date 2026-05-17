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


def _check(results: list, name: str, got, expected) -> None:
    ok = got == expected
    results.append((ok, name, got, expected))


def run_selftest() -> int:
    results: list = []
    selftest_arch(results)
    selftest_parsers(results)
    selftest_engine(results)
    failed = [r for r in results if not r[0]]
    for ok, name, got, expected in results:
        if not ok:
            print(f"FAIL {name}: got={got!r} expected={expected!r}")
    print(f"SELFTEST: {'PASS' if not failed else 'FAIL'} "
          f"({len(results) - len(failed)}/{len(results)} checks)")
    return 0 if not failed else 1


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


if __name__ == "__main__":
    raise SystemExit(main())

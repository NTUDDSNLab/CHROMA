#!/usr/bin/env python3
"""Build-and-sweep 18 GPU graph-coloring tools over a directory of .egr graphs.

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
KOKKOS_ROOT_DEFAULT = "/home/chsieh45/local/kokkos-cuda"


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
_CHROMA_TOTAL_MS = r"^\s*Total runtime:\s+([0-9]+(?:\.[0-9]+)?)\s*ms"

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
    dict(name="CHROMA", kind="chroma", unit="chroma",
         binrel="CHROMA/CHROMA",
         argv=["-f", "{G}", "-a", "0"],
         colors=_COLORS_USED, time=("ms", _CHROMA_TOTAL_MS),
         algo="a=0", usrc="ms"),
    dict(name="CHROMA+", kind="chroma", unit="chroma",
         binrel="CHROMA/CHROMA",
         argv=["-f", "{G}", "-a", "1"],
         colors=_COLORS_USED, time=("ms", _CHROMA_TOTAL_MS),
         algo="a=1", usrc="ms"),
    dict(name="CHROMA*", kind="chroma", unit="chroma",
         binrel="CHROMA/CHROMA",
         argv=["-f", "{G}", "-a", "1", "-p",
               "--predict-model", "v0_paper"],
         colors=_COLORS_USED, time=("ms", _CHROMA_TOTAL_MS),
         algo="a=1,predict=v0_paper", usrc="ms"),
    dict(name="CHROMA_v2-b-adw", kind="chroma", unit="chroma",
         binrel="CHROMA/CHROMA",
         argv=["-f", "{G}", "-a", "1", "-p",
               "--predict-model", "v3", "--no-dynamic-theta"],
         colors=_COLORS_USED, time=("ms", _CHROMA_TOTAL_MS),
         algo="a=1,predict=v3,no-dyn-theta", usrc="ms"),
    dict(name="CHROMA_v2-b", kind="chroma", unit="chroma",
         binrel="CHROMA/CHROMA",
         argv=["-f", "{G}", "-a", "10", "-p",
               "--predict-model", "v3", "--no-dynamic-theta"],
         colors=_COLORS_USED, time=("ms", _CHROMA_TOTAL_MS),
         algo="a=10,predict=v3,no-dyn-theta", usrc="ms"),
    dict(name="CHROMA_v2", kind="chroma", unit="chroma",
         binrel="CHROMA/CHROMA",
         argv=["-f", "{G}", "-a", "10", "-p",
               "--predict-model", "v3"],
         colors=_COLORS_USED, time=("ms", _CHROMA_TOTAL_MS),
         algo="a=10,predict=v3", usrc="ms"),
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


def build_argv(tool: dict, binary_abs: str, graph_abs: str) -> list[str]:
    return [binary_abs] + [graph_abs if a == "{G}" else a
                           for a in tool["argv"]]


def resolve_kokkos_root(cli_value: Optional[str]) -> str:
    """Kokkos install prefix: --kokkos-root > $KOKKOS_ROOT > built-in."""
    if cli_value:
        return cli_value
    return os.environ.get("KOKKOS_ROOT") or KOKKOS_ROOT_DEFAULT


def build_unit_steps(unit: str, nn: str,
                     kokkos_root: str = KOKKOS_ROOT_DEFAULT) -> list[dict]:
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
        kk = kokkos_root
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
    if unit == "chroma":
        return [
            {"cmd": ["make", "-C", "CHROMA", "clean"], "cwd": ".",
             "ignore_fail": True, "retry": None},
            {"cmd": ["make", "-C", "CHROMA", "PRE_MODEL=1",
                     f"ARCH={sm}"],
             "cwd": ".", "ignore_fail": False, "retry": None},
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


def build_one_unit(unit: str, nn: str,
                   kokkos_root: str = KOKKOS_ROOT_DEFAULT) -> dict:
    start = time.perf_counter()
    last_cmd = ""
    for step in build_unit_steps(unit, nn, kokkos_root):
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


def run_build_phase(units: set[str], nn: str,
                    kokkos_root: str = KOKKOS_ROOT_DEFAULT) -> dict:
    """unit -> build result dict. Built in a stable order."""
    order = ["csrcolor", "csrcolor_data", "kokkos", "pgc", "picasso",
             "ecl-gc", "ecl-gc-r", "jp-series", "chroma"]
    out = {}
    for unit in order:
        if unit in units:
            print(f"[build] {unit} ...", flush=True)
            res = build_one_unit(unit, nn, kokkos_root)
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
    p.add_argument("--kokkos-root", default=None,
                   help="Kokkos install prefix for the kokkos build unit "
                        "(default: $KOKKOS_ROOT, else the built-in path).")
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

    nn, arch_source = resolve_arch(args.arch)
    kokkos_root = resolve_kokkos_root(args.kokkos_root)
    tools = select_tools(args.only, args.exclude)
    units = needed_units(tools)

    builds: dict = {}
    if not args.skip_build:
        builds = run_build_phase(units, nn, kokkos_root)

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
        "kokkos_root": kokkos_root,
        "skip_build": args.skip_build,
        "tools": [t["name"] for t in tools],
    }
    doc = build_json_doc(config, builds, tools, availability,
                         [os.path.basename(d) for d in datasets], rows)
    write_json_atomic(args.out, doc)
    print_summary(doc)
    print(f"\nwrote {args.out}")
    return 0


def _check(results: list, name: str, got, expected) -> None:
    ok = got == expected
    results.append((ok, name, got, expected))


def run_selftest() -> int:
    results: list = []
    selftest_arch(results)
    selftest_parsers(results)
    selftest_engine(results)
    selftest_commands(results)
    selftest_run(results)
    selftest_json(results)
    selftest_chroma(results)
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

    _check(results, "registry size", len(REGISTRY), 18)
    _check(results, "registry names unique",
           len({t["name"] for t in REGISTRY}), 18)


def selftest_engine(results: list) -> None:
    sel = select_tools(only="cuSL,ECL-GC", exclude=None)
    _check(results, "only filter", [t["name"] for t in sel],
           ["ECL-GC", "cuSL"])  # registry order preserved

    sel2 = select_tools(only=None, exclude="data_wlc,data_pq")
    _check(results, "exclude filter count", len(sel2), 16)
    _check(results, "exclude removed", "data_wlc" in
           [t["name"] for t in sel2], False)

    sel3 = select_tools(only=None, exclude=None)
    _check(results, "no filter = all", len(sel3), 18)

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

    kko = build_unit_steps("kokkos", "89", "/opt/kk")[1]["cmd"]
    _check(results, "kokkos --kokkos-root override",
           "-DKokkos_ROOT=/opt/kk" in kko
           and "-DCMAKE_CXX_COMPILER=/opt/kk/bin/nvcc_wrapper" in kko,
           True)
    _check(results, "kokkos default root preserved",
           "-DKokkos_ROOT=" + KOKKOS_ROOT_DEFAULT
           in build_unit_steps("kokkos", "89")[1]["cmd"], True)


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


def selftest_json(results: list) -> None:
    tools = select_tools("cuSL,ECL-GC-R", None)
    builds = {"jp-series": {"unit": "jp-series", "ok": False,
                            "cmd": "make ...", "seconds": 9.0,
                            "error": "boom"},
              "ecl-gc-r": {"unit": "ecl-gc-r", "ok": False,
                           "cmd": "nvcc ...", "seconds": 1.0,
                           "error": "boom"}}
    avail = compute_availability(tools, builds, skip_build=False)
    _check(results, "cuSL unavailable (build fail)", avail["cuSL"][0],
           False)
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


def selftest_chroma(results: list) -> None:
    by_name = {t["name"]: t for t in REGISTRY}

    # Realistic CHROMA single-run stdout (reduction default-off; a predict
    # config also prints the "EGC θ: N (Predicted)" info line).
    out = ("EGC θ: 12 (Predicted)\nFinish PA\nFinish CA\n"
           "PA runtime: 5.000000 ms\nCA runtime: 2.000000 ms\n"
           "Post reduction runtime: 0.000000 ms\n"
           "Total runtime: 7.000000 ms\n"
           "colors before reduction: 50\ncolors after reduction: 50\n"
           "color reduction delta: 0\nresult verification passed\n"
           "colors used: 50\nIter count: 7\n")
    _check(results, "chroma colors (colors used, not reduction lines)",
           parse_colors(by_name["CHROMA"], out), 50)
    _check(results, "chroma ms (Total runtime, not PA/CA)",
           parse_time_ms(by_name["CHROMA"], out), 7.0)
    _check(results, "argv CHROMA_v2-b-adw",
           build_argv(by_name["CHROMA_v2-b-adw"], "/b/c", "/g/x.egr"),
           ["/b/c", "-f", "/g/x.egr", "-a", "1", "-p",
            "--predict-model", "v3", "--no-dynamic-theta"])


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

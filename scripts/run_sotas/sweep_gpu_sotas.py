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


def run_sweep(args: argparse.Namespace) -> int:
    raise NotImplementedError  # filled in Task 8


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


if __name__ == "__main__":
    raise SystemExit(main())

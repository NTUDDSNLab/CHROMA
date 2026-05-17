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

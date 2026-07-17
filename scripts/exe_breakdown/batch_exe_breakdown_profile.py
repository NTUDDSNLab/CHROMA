#!/usr/bin/env python3
"""Sweep the paper's CHROMA configurations over EGR datasets and capture
timings for an execution-time breakdown figure — all configs land in ONE
JSON so the plot script can pick any subset.

Each config bundles an algorithm (-a), extra CLI flags, and its own theta
regime (fixed -e vs --predict with a specific model), matching the paper's
configuration table:

  name             -a algo              EGC theta          AWD  Bumping
  CHROMA           cuSL_ELS             -e <N>             -    -
  CHROMA+          cuSL_ELS_SDC         -e <N>             -    -
  CHROMA_star      cuSL_ELS_SDC         predict v0_paper   -    -
  CHROMA_star_awd  cuSL_ELS_SDC_CTA_S   predict v0_paper   v    -
  CHROMA_v2-b-awd  cuSL_ELS_SDC         predict 3feat      -    -
  CHROMA_v2-b      cuSL_ELS_SDC_CTA_S   predict 3feat      v    -
  CHROMA_v2        cuSL_ELS_SDC_CTA_S   predict 3feat      v    v

Names not in CONFIG_SPECS pass through as a raw -a algorithm with the -e
theta (the SPLIT-mode diagnostic algos use this; their per-phase `PA scan`
/ `PA decrement` rows are captured when present and the plot script then
renders a three-segment stack).

For each (config, dataset) pair the script invokes the CHROMA binary with
--runs N (default 5), parses the `=== Statistics over N runs (ms) ===`
block from stdout, and writes a JSON suitable for
`scripts/exe_breakdown/plot_execution_breakdown.py`.

NOTE: the predict-based configs need a PRE_MODEL=1 binary, and CHROMA_v2
(bumping) additionally needs DYNAMIC_THETA=1 (the Makefile default).

Examples:
    python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py
    python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py \\
        --only facebook le450_25d --runs 3
    # Just the journal-version configs.
    python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py \\
        --configs CHROMA_v2-b-awd CHROMA_v2-b CHROMA_v2
    # Three-segment breakdown via SPLIT kernels (slower, diagnostic-only).
    python3 scripts/exe_breakdown/batch_exe_breakdown_profile.py \\
        --configs cuSL_ELS_SDC_SPLIT cuSL_ELS_SDC_CTA_SPLIT \\
                  cuSL_ELS_SDC_CTA_S_SPLIT
"""
from __future__ import annotations
import argparse
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

#
# Paper config name → (CHROMA -a value, extra CLI flags, theta spec).
# theta spec is ("elastic", None) — use the sweep-level -e value — or
# ("predict", <model>) — use --predict --predict-model <model>.
#
# All static-theta configs pass --no-dynamic-theta so the on-device θ
# controller (default ON in DYNAMIC_THETA=1 builds) doesn't contaminate
# them; only CHROMA_v2 leaves it enabled (the Bumping column).
# "_star" stands in for the paper's superscript-* (shell-safe).
#
CONFIG_SPECS = {
    "CHROMA":          ("cuSL_ELS",           ["--no-dynamic-theta"], ("elastic", None)),
    "CHROMA+":         ("cuSL_ELS_SDC",       ["--no-dynamic-theta"], ("elastic", None)),
    "CHROMA_star":     ("cuSL_ELS_SDC",       ["--no-dynamic-theta"], ("predict", "v0_paper")),
    "CHROMA_star_awd": ("cuSL_ELS_SDC_CTA_S", ["--no-dynamic-theta"], ("predict", "v0_paper")),
    "CHROMA_v2-b-awd": ("cuSL_ELS_SDC",       ["--no-dynamic-theta"], ("predict", "3feat")),
    "CHROMA_v2-b":     ("cuSL_ELS_SDC_CTA_S", ["--no-dynamic-theta"], ("predict", "3feat")),
    "CHROMA_v2":       ("cuSL_ELS_SDC_CTA_S", [],                     ("predict", "3feat")),
}

DEFAULT_CONFIGS = list(CONFIG_SPECS.keys())


def resolve_config(config: str) -> tuple[str, list, tuple]:
    """Return (CHROMA -a value, extra CLI flags, theta spec) for a config
    name. Names not in CONFIG_SPECS pass through unchanged as a raw -a
    algorithm with no extras and the -e theta (the SPLIT-mode diagnostic
    algos use this path)."""
    spec = CONFIG_SPECS.get(config)
    if spec is None:
        return (config, [], ("elastic", None))
    return spec

# stats_f produces  "avg=%9.3f  min=%9.3f  max=%9.3f" — \s* covers padding.
# Required rows are emitted by every algo; the optional pair is only
# emitted by SPLIT-mode algos and is captured when present.
STAT_RE_REQUIRED = {
    "ca_ms":       re.compile(r"^CA time\s*:\s*avg=\s*([0-9.]+)",     re.MULTILINE),
    "pa_ms":       re.compile(r"^PA time\s*:\s*avg=\s*([0-9.]+)",     re.MULTILINE),
    "total_ms":    re.compile(r"^Total time\s*:\s*avg=\s*([0-9.]+)",  re.MULTILINE),
    "colors_used": re.compile(r"^colors used\s*:\s*avg=\s*([0-9.]+)", re.MULTILINE),
}
STAT_RE_OPTIONAL = {
    "pa_scan_ms":      re.compile(r"^PA scan\s*:\s*avg=\s*([0-9.]+)",      re.MULTILINE),
    "pa_decrement_ms": re.compile(r"^PA decrement\s*:\s*avg=\s*([0-9.]+)", re.MULTILINE),
}


def read_egr_size(path: Path) -> tuple[int, int]:
    """Parse nodes/edges from the ECLgraph .egr binary header.

    Header layout (lib/io/ECLgraph.h): two int32 fields at offset 0:
      [0..3]  nodes
      [4..7]  edges
    """
    with open(path, "rb") as f:
        head = f.read(8)
    nodes, edges = struct.unpack("<ii", head)
    return nodes, edges


def parse_stats(stdout: str) -> Optional[dict]:
    out = {}
    for key, rx in STAT_RE_REQUIRED.items():
        m = rx.search(stdout)
        if m is None:
            return None
        out[key] = float(m.group(1))
    for key, rx in STAT_RE_OPTIONAL.items():
        m = rx.search(stdout)
        if m is not None:
            out[key] = float(m.group(1))
    return out


def build_cmd(binary: Path, egr: Path, config: str, runs: int,
              elastic: int) -> list:
    # --no-reduce: the figure measures pure CA + PA scan + PA decrement.
    # Color reduction is a separate post-CA phase and is not part of the
    # breakdown; disabling it also keeps CHROMA's runtime cleaner.
    algo, extras, (theta_mode, model) = resolve_config(config)
    cmd = [str(binary), "-f", str(egr), "-a", algo, "--no-reduce",
           "--runs", str(runs)]
    cmd.extend(extras)
    if theta_mode == "predict":
        cmd.extend(["--predict", "--predict-model", model])
    else:
        cmd.extend(["-e", str(elastic)])
    return cmd


def run_one(cmd: list, timeout: int):
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0, f"TIMEOUT after {timeout}s"
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").splitlines()[-3:]
        return None, dt, f"rc={r.returncode}: {' | '.join(tail)}"
    stats = parse_stats(r.stdout)
    if stats is None:
        tail = r.stdout.splitlines()[-5:]
        return None, dt, f"could not parse stats: ...{' / '.join(tail)}"
    return stats, dt, None


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary",      default=str(repo / "CHROMA" / "CHROMA"))
    ap.add_argument("--dataset-dir", default=str(repo / "Datasets" / "EGR"))
    ap.add_argument("--runs",        type=int, default=5,
                    help="Repeated runs per cell (must be >= 2; the stats "
                         "block is gated on num_runs > 1)")
    ap.add_argument("--timeout",     type=int, default=1200,
                    help="Per-cell timeout (seconds). SPLIT mode is slow.")
    ap.add_argument("--configs",     nargs="+", default=DEFAULT_CONFIGS,
                    help="Paper config names (see CONFIG_SPECS) and/or raw "
                         "CHROMA -a algorithm names. Default: all "
                         f"{len(DEFAULT_CONFIGS)} paper configs.")
    ap.add_argument("--only",        nargs="*", default=None,
                    help="Restrict to dataset stems (matched by .egr basename)")
    ap.add_argument("--skip",        nargs="*", default=[])
    ap.add_argument("--out",         default=str(repo / "scripts" / "exe_breakdown"
                                                 / "batch_profile_results.json"))
    ap.add_argument("-e", "--elastic", type=int, default=0,
                    help="Theta for the fixed-theta configs (CHROMA, CHROMA+ "
                         "and raw algo names). Predict-based configs ignore "
                         "it. Default: 0.")
    args = ap.parse_args()

    if args.runs < 2:
        sys.exit("ERROR: --runs must be >= 2 (CHROMA only prints the "
                 "statistics block when num_runs > 1)")
    elastic = args.elastic

    bin_path = Path(args.binary)
    if not bin_path.exists():
        print(f"ERROR: missing CHROMA binary at {bin_path}", file=sys.stderr)
        sys.exit(1)

    ds_dir = Path(args.dataset_dir)
    egrs = sorted(ds_dir.glob("*.egr"))
    if args.only:
        wanted = set(args.only)
        egrs = [p for p in egrs
                if p.stem.split('.')[0] in wanted or p.stem in wanted]
    if args.skip:
        skip = set(args.skip)
        egrs = [p for p in egrs
                if p.stem.split('.')[0] not in skip and p.stem not in skip]

    print(f"# {len(egrs)} datasets x {len(args.configs)} configs x "
          f"{args.runs} runs (e={elastic} for fixed-theta configs)",
          file=sys.stderr)

    datasets_meta = []
    seen_names = set()
    rows = []

    for egr in egrs:
        name = egr.stem
        try:
            nodes, edges = read_egr_size(egr)
        except Exception as e:
            print(f"# WARN: cannot read header of {egr}: {e}", file=sys.stderr)
            nodes, edges = 0, 0
        if name not in seen_names:
            datasets_meta.append({"name": name, "nodes": nodes, "edges": edges})
            seen_names.add(name)

        for cfg in args.configs:
            algo, _extras, (theta_mode, model) = resolve_config(cfg)
            predict = theta_mode == "predict"
            cmd = build_cmd(bin_path, egr, cfg, args.runs, elastic)
            stats, wall, err = run_one(cmd, args.timeout)
            row = {
                "config":        cfg,
                "algo":          algo,
                "dataset":       name,
                "nodes":         nodes,
                "edges":         edges,
                "runs":          args.runs,
                "wall_s":        wall,
                "elastic":       None if predict else elastic,
                "predict":       predict,
                "predict_model": model,
            }
            if err:
                row["error"] = err
                print(f"# {name:32s} {cfg:20s} FAIL  ({err})", file=sys.stderr)
            else:
                row.update(stats)
                if "pa_scan_ms" in stats and "pa_decrement_ms" in stats:
                    pa_str = (f"scan={stats['pa_scan_ms']:7.2f}ms "
                              f"dec={stats['pa_decrement_ms']:7.2f}ms")
                else:
                    pa_str = f"pa={stats['pa_ms']:7.2f}ms"
                print(f"# {name:32s} {cfg:20s} "
                      f"ca={stats['ca_ms']:7.2f}ms "
                      f"{pa_str} "
                      f"colors={stats['colors_used']:5.1f} "
                      f"wall={wall:6.1f}s", file=sys.stderr)
            rows.append(row)

    summary = {
        "config": {
            "elastic":   elastic,
            "runs":      args.runs,
            "no_reduce": True,
        },
        "configs":  list(args.configs),
        "datasets": datasets_meta,
        "rows":     rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())

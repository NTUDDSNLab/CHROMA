#!/usr/bin/env python3
"""Sweep three CHROMA frameworks over EGR datasets and capture timings
for an execution-time breakdown figure.

Defaults to the unified (non-SPLIT) cooperative kernels:
  cuSL_ELS_SDC, cuSL_ELS_SDC_CTA, cuSL_ELS_SDC_CTA_S
which produce a single `PA time` per run (scan + decrement fused inside
one cooperative-launch kernel — the production path).

If you pass SPLIT-mode framework names instead (e.g. cuSL_ELS_SDC_SPLIT)
the script also captures the separate `PA scan` / `PA decrement` rows
that CHROMA emits for those variants, and the plot script renders a
three-segment stack accordingly.

For each (framework, dataset) pair the script invokes the CHROMA binary
with --runs N (default 5), parses the `=== Statistics over N runs (ms) ===`
block from stdout, and writes a JSON suitable for
`scripts/plots/plot_execution_breakdown.py`.

The theta configuration is chosen via mutually-exclusive --elastic and
--predict flags so the same script can profile different theta regimes
into separate JSON files.

Examples:
    python3 scripts/batch_exe_breakdown_profile.py
    python3 scripts/batch_exe_breakdown_profile.py --only facebook le450_25d --runs 3
    python3 scripts/batch_exe_breakdown_profile.py --elastic 10
    python3 scripts/batch_exe_breakdown_profile.py --predict
    python3 scripts/batch_exe_breakdown_profile.py --predict --predict-model v3 \\
        --out scripts/breakdown_predict_v3.json
    # Three-segment breakdown via SPLIT kernels (slower, diagnostic-only).
    python3 scripts/batch_exe_breakdown_profile.py \\
        --frameworks cuSL_ELS_SDC_SPLIT cuSL_ELS_SDC_CTA_SPLIT \\
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

DEFAULT_FRAMEWORKS = [
    "cuSL_ELS_SDC",
    "cuSL_ELS_SDC_CTA",
    "cuSL_ELS_SDC_CTA_S",
]

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


def build_cmd(binary: Path, egr: Path, framework: str, runs: int,
              elastic: int, predict: bool, predict_model: str) -> list:
    # --no-reduce: the figure measures pure CA + PA scan + PA decrement.
    # Color reduction is a separate post-CA phase and is not part of the
    # breakdown; disabling it also keeps CHROMA's runtime cleaner.
    cmd = [str(binary), "-f", str(egr), "-a", framework, "--no-reduce",
           "--runs", str(runs)]
    if predict:
        cmd.extend(["--predict", "--predict-model", predict_model])
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
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary",      default=str(repo / "CHROMA" / "CHROMA"))
    ap.add_argument("--dataset-dir", default=str(repo / "Datasets" / "EGR"))
    ap.add_argument("--runs",        type=int, default=5,
                    help="Repeated runs per cell (must be >= 2; the stats "
                         "block is gated on num_runs > 1)")
    ap.add_argument("--timeout",     type=int, default=1200,
                    help="Per-cell timeout (seconds). SPLIT mode is slow.")
    ap.add_argument("--frameworks",  nargs="+", default=DEFAULT_FRAMEWORKS)
    ap.add_argument("--only",        nargs="*", default=None,
                    help="Restrict to dataset stems (matched by .egr basename)")
    ap.add_argument("--skip",        nargs="*", default=[])
    ap.add_argument("--out",         default=str(repo / "scripts" / "batch_profile_results.json"))

    # Theta configuration. --elastic and --predict are mutually exclusive;
    # default is --elastic 0.
    ap.add_argument("-e", "--elastic", type=int, default=None,
                    help="Set theta via CHROMA's -e flag. Default: 0. "
                         "Mutually exclusive with --predict.")
    ap.add_argument("-p", "--predict", action="store_true",
                    help="Use the linked-in ML model (CHROMA --predict) "
                         "instead of a fixed -e. Mutually exclusive with "
                         "--elastic.")
    ap.add_argument("--predict-model", choices=["v3", "v0_paper"],
                    default="v3",
                    help="When --predict is set, pick which model CHROMA "
                         "should use. 'v3' (default, 9-feature random forest) "
                         "or 'v0_paper' (paper-era RF on V/E only). Forwarded "
                         "verbatim as --predict-model <name>. Ignored when "
                         "--predict is not set.")
    args = ap.parse_args()

    if args.runs < 2:
        sys.exit("ERROR: --runs must be >= 2 (CHROMA only prints the "
                 "statistics block when num_runs > 1)")

    if args.predict and args.elastic is not None:
        sys.exit("ERROR: --elastic and --predict are mutually exclusive.")
    elastic = args.elastic if args.elastic is not None else 0

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

    config_label = ("predict=" + args.predict_model) if args.predict else f"e={elastic}"
    print(f"# {len(egrs)} datasets x {len(args.frameworks)} frameworks x "
          f"{args.runs} runs ({config_label})", file=sys.stderr)

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

        for fw in args.frameworks:
            cmd = build_cmd(bin_path, egr, fw, args.runs,
                            elastic, args.predict, args.predict_model)
            stats, wall, err = run_one(cmd, args.timeout)
            row = {
                "framework":     fw,
                "dataset":       name,
                "nodes":         nodes,
                "edges":         edges,
                "runs":          args.runs,
                "wall_s":        wall,
                "elastic":       None if args.predict else elastic,
                "predict":       args.predict,
                "predict_model": args.predict_model if args.predict else None,
            }
            if err:
                row["error"] = err
                print(f"# {name:32s} {fw:30s} FAIL  ({err})", file=sys.stderr)
            else:
                row.update(stats)
                if "pa_scan_ms" in stats and "pa_decrement_ms" in stats:
                    pa_str = (f"scan={stats['pa_scan_ms']:7.2f}ms "
                              f"dec={stats['pa_decrement_ms']:7.2f}ms")
                else:
                    pa_str = f"pa={stats['pa_ms']:7.2f}ms"
                print(f"# {name:32s} {fw:30s} "
                      f"ca={stats['ca_ms']:7.2f}ms "
                      f"{pa_str} "
                      f"colors={stats['colors_used']:5.1f} "
                      f"wall={wall:6.1f}s", file=sys.stderr)
            rows.append(row)

    summary = {
        "config": {
            "elastic":       None if args.predict else elastic,
            "predict":       args.predict,
            "predict_model": args.predict_model if args.predict else None,
            "runs":          args.runs,
            "no_reduce":     True,
        },
        "frameworks": list(args.frameworks),
        "datasets":   datasets_meta,
        "rows":       rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())

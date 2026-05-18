#!/usr/bin/env python3
"""Sweep priority-ordering consistency vs the JP-SL^A reference.

For each EGR dataset: dump the JP-SL^A reference priority list with
cpu_SL, dump/synthesize each comparison framework's priority list, run
scripts/consistency_metric <egr> <ref.bin> <test.bin>, and collect the
consistency ratio. Writes a JSON consumed by
plot_priority_consistency.py.

Framework -> priority source:
  JP-SL^A         cpu_SL --dump compared against itself (per-dataset
                  consistency ceiling: the best any framework can score)
  JP-SLL          pa_dumper -a JP_SLL
  JP-ADG          cpu_ADG <egr> <threads> --dump  (pa_dumper JP_ADG emits no per-vertex order)
  CHROMA          pa_dumper -a cuSL_ELS        -e 0
  CHROMA+         pa_dumper -a cuSL_ELS_SDC    -e 0
  CHROMA*         CHROMA -a cuSL_ELS_SDC --predict --predict-model v0_paper
                         --no-dynamic-theta
  CHROMA_v2_raw   CHROMA -a cuSL_ELS_SDC --predict --predict-model v3
                         --no-dynamic-theta   (offline predictor only)
  CHROMA_v2_bump  CHROMA -a cuSL_ELS_SDC --predict --predict-model v3
                         (predictor + on-device dynamic-theta bumping)

Examples:
    python3 scripts/plots/priority_consistency/sweep_priority_consistency.py
    python3 scripts/plots/priority_consistency/sweep_priority_consistency.py \\
        --only facebook le450_25d
"""
from __future__ import annotations
import argparse
import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Internal key -> (kind, spec). kind drives how the test dump is made.
#   "pa"     : pa_dumper -a <algo> -e 0
#   "cpuadg" : cpu_ADG <egr> <threads> --dump  (pa_dumper JP_ADG has no per-vertex order)
#   "chroma" : CHROMA -a cuSL_ELS_SDC --predict --predict-model <model>
#              [--no-dynamic-theta if spec["bump"] is False]
#   "self"   : the JP-SL^A reference compared against itself — the
#              per-dataset consistency ceiling (no separate dump; the
#              metric runs ref vs ref)
# v3_raw = offline predictor only (online dynamic-theta bumping OFF);
# v3_bump = predictor + online bumping (CHROMA default in
# DYNAMIC_THETA=1 builds). v0_paper is the paper-era predictor, no bump.
FRAMEWORKS = {
    "JP-SL^A":        ("self",   {}),
    "JP-SLL":         ("pa",     {"algo": "JP_SLL"}),
    "JP-ADG":         ("cpuadg", {}),
    "CHROMA":         ("pa",     {"algo": "cuSL_ELS"}),
    "CHROMA+":        ("pa",     {"algo": "cuSL_ELS_SDC"}),
    "CHROMA*":        ("chroma", {"model": "v0_paper", "bump": False}),
    "CHROMA_v2_raw":  ("chroma", {"model": "v3",       "bump": False}),
    "CHROMA_v2_bump": ("chroma", {"model": "v3",       "bump": True}),
}
DEFAULT_FRAMEWORKS = list(FRAMEWORKS.keys())


def read_egr_header(path: Path) -> tuple:
    """Return (nodes, edges) from the .egr CSR binary header."""
    with open(path, "rb") as f:
        nodes, edges = struct.unpack("<ii", f.read(8))
    return nodes, edges


def run(cmd: list, timeout: int):
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0, f"TIMEOUT after {timeout}s"
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").splitlines()[-3:]
        return None, dt, f"rc={r.returncode}: {' | '.join(tail)}"
    return r, dt, None


def make_test_dump(args, key, egr, out_bin) -> tuple:
    kind, spec = FRAMEWORKS[key]
    if kind == "pa":
        cmd = [args.pa_dumper, "-f", str(egr), "-a", spec["algo"],
               "-e", "0", "--dump-priority", str(out_bin)]
    elif kind == "cpuadg":
        cmd = [args.cpu_adg, str(egr), str(args.threads),
               "--dump", str(out_bin)]
    else:  # chroma
        cmd = [args.binary, "-f", str(egr), "-a", "cuSL_ELS_SDC",
               "--no-reduce", "--predict", "--predict-model", spec["model"]]
        if not spec.get("bump", False):
            cmd.append("--no-dynamic-theta")  # raw predictor, no online bump
        cmd += ["--dump-priority", str(out_bin)]
    _r, dt, err = run(cmd, args.timeout)
    return dt, err


def parse_metric(stdout: str) -> Optional[dict]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def main():
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary",      default=str(repo / "CHROMA" / "CHROMA"))
    ap.add_argument("--pa-dumper",   default=str(repo / "tools" / "pa_dumper" / "pa_dumper"))
    ap.add_argument("--cpu-sl",      default=str(repo / "CPU" / "Parallel" / "cpu_SL"))
    ap.add_argument("--cpu-adg",     default=str(repo / "CPU" / "Parallel" / "cpu_ADG"))
    ap.add_argument("--metric-bin",  default=str(repo / "scripts" / "consistency_metric"))
    ap.add_argument("--dataset-dir", default=str(repo / "Datasets" / "EGR"))
    ap.add_argument("--threads",     type=int, default=32,
                    help="OpenMP threads for cpu_SL (JP-SL^A reference)")
    ap.add_argument("--frameworks",  nargs="+", default=DEFAULT_FRAMEWORKS)
    ap.add_argument("--only",        nargs="*", default=None)
    ap.add_argument("--skip",        nargs="*", default=[])
    ap.add_argument("--timeout",     type=int, default=1800,
                    help="Per dump / metric call timeout (seconds)")
    ap.add_argument("--keep-dumps",  action="store_true")
    ap.add_argument("--out", default=str(
        repo / "scripts" / "plots" / "priority_consistency" /
        "consistency_results.json"))
    args = ap.parse_args()

    bad = [fw for fw in args.frameworks if fw not in FRAMEWORKS]
    if bad:
        sys.exit(f"ERROR: unknown frameworks {bad}; "
                 f"valid: {list(FRAMEWORKS)}")

    kinds = {FRAMEWORKS[fw][0] for fw in args.frameworks}
    needed = [("cpu_SL", args.cpu_sl), ("consistency_metric", args.metric_bin)]
    if "pa" in kinds:
        needed.append(("pa_dumper", args.pa_dumper))
    if "cpuadg" in kinds:
        needed.append(("cpu_ADG", args.cpu_adg))
    if "chroma" in kinds:
        needed.append(("CHROMA", args.binary))
    for label, p in needed:
        if not Path(p).exists():
            sys.exit(f"ERROR: missing {label} at {p}")

    ds_dir = Path(args.dataset_dir)
    egrs = sorted(ds_dir.glob("*.egr"))
    if args.only:
        want = set(args.only)
        egrs = [p for p in egrs
                if p.stem.split('.')[0] in want or p.stem in want]
    if args.skip:
        sk = set(args.skip)
        egrs = [p for p in egrs
                if p.stem.split('.')[0] not in sk and p.stem not in sk]

    print(f"# {len(egrs)} datasets x {len(args.frameworks)} frameworks "
          f"(baseline JP-SL^A via cpu_SL, {args.threads} threads)",
          file=sys.stderr)

    datasets_meta = []
    rows = []
    tmp_root = Path(tempfile.mkdtemp(prefix="prio_consist_"))

    for egr in egrs:
        name = egr.stem
        try:
            nodes, edges = read_egr_header(egr)
        except Exception as e:
            print(f"# WARN {egr}: cannot read header: {e}", file=sys.stderr)
            continue
        datasets_meta.append({"name": name, "nodes": nodes, "edges": edges})

        ref_bin = tmp_root / f"{name}__JP-SL_A.bin"
        _r, ref_dt, ref_err = run(
            [args.cpu_sl, str(egr), str(args.threads), "--dump", str(ref_bin)],
            args.timeout)
        if ref_err:
            print(f"# {name:28s} JP-SL^A REF FAIL ({ref_err})", file=sys.stderr)
            for fw in args.frameworks:
                rows.append({"framework": fw, "dataset": name,
                             "nodes": nodes, "edges": edges,
                             "error": f"ref failed: {ref_err}"})
            continue

        for fw in args.frameworks:
            kind = FRAMEWORKS[fw][0]
            if kind == "self":
                # JP-SL^A vs itself: per-dataset consistency ceiling.
                test_bin = ref_bin
                dump_dt, dump_err = 0.0, None
            else:
                test_bin = tmp_root / f"{name}__{fw}.bin"
                dump_dt, dump_err = make_test_dump(args, fw, egr, test_bin)
            row = {"framework": fw, "dataset": name,
                   "nodes": nodes, "edges": edges,
                   "ref_wall_s": ref_dt, "dump_wall_s": dump_dt}
            if dump_err:
                row["error"] = f"dump failed: {dump_err}"
                print(f"# {name:28s} {fw:14s} DUMP FAIL ({dump_err})",
                      file=sys.stderr)
                rows.append(row)
                continue
            mr, m_dt, m_err = run(
                [args.metric_bin, str(egr), str(ref_bin), str(test_bin)],
                args.timeout)
            if m_err:
                row["error"] = f"metric failed: {m_err}"
                print(f"# {name:28s} {fw:14s} METRIC FAIL ({m_err})",
                      file=sys.stderr)
            else:
                metric = parse_metric(mr.stdout)
                if metric is None:
                    row["error"] = "metric: no JSON line"
                    print(f"# {name:28s} {fw:14s} METRIC PARSE FAIL",
                          file=sys.stderr)
                else:
                    for k in ("consistency_ratio", "tie_ratio_ref",
                              "tie_ratio_test", "distinct_ref",
                              "distinct_test", "concordant_unord_pairs"):
                        if k in metric:
                            row[k] = metric[k]
                    row["metric_wall_s"] = m_dt
                    print(f"# {name:28s} {fw:14s} "
                          f"cons={row.get('consistency_ratio', float('nan')):.4f} "
                          f"ref={ref_dt:6.1f}s", file=sys.stderr)
            rows.append(row)
            if not args.keep_dumps and kind != "self":
                # never unlink here when test_bin IS ref_bin (self) —
                # ref_bin is still needed by later frameworks and is
                # cleaned up once after the framework loop.
                test_bin.unlink(missing_ok=True)
        if not args.keep_dumps:
            ref_bin.unlink(missing_ok=True)

    summary = {
        "baseline":   "JP-SL^A",
        "frameworks": list(args.frameworks),
        "datasets":   datasets_meta,
        "rows":       rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)
    if not args.keep_dumps:
        try:
            tmp_root.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Sweep all .egr datasets, compute consistency ratio (JP-SL^M baseline vs
cuSL_ELS_SDC + EGC theta) and adjacent-pair tie ratios.

Outputs: JSON per dataset to results, plus a summary table on stdout.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_args():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=str(repo / "Datasets" / "EGR"))
    p.add_argument("--ref-bin", default=str(repo / "CPU" / "Sequential" / "cpu_SDL"))
    p.add_argument("--test-bin", default=str(repo / "CHROMA" / "CHROMA"))
    p.add_argument("--metric-bin", default=str(repo / "scripts" / "consistency_metric"))
    p.add_argument("--algo", default="cuSL_ELS_SDC")
    p.add_argument("--theta", type=int, default=5)
    p.add_argument("--out", default=str(repo / "scripts" / "consistency_results.json"))
    p.add_argument("--ref-timeout", type=int, default=1800)
    p.add_argument("--test-timeout", type=int, default=600)
    p.add_argument("--metric-timeout", type=int, default=900)
    p.add_argument("--only", nargs="*", default=None,
                   help="If set, only run on these dataset stems (without .egr)")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Skip these dataset stems")
    return p.parse_args()


def run_cmd(cmd, timeout, label):
    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0, f"TIMEOUT after {timeout}s"
    dt = time.perf_counter() - t0
    if out.returncode != 0:
        tail = (out.stderr or out.stdout or "").splitlines()[-3:]
        return None, dt, f"{label} returncode={out.returncode}: {' | '.join(tail)}"
    return out, dt, None


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: dataset dir not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)
    for path in (args.ref_bin, args.test_bin, args.metric_bin):
        if not Path(path).exists():
            print(f"ERROR: missing binary: {path}", file=sys.stderr)
            sys.exit(1)

    egrs = sorted(dataset_dir.glob("*.egr"), key=lambda p: p.stat().st_size)
    if args.only:
        wanted = set(args.only)
        egrs = [p for p in egrs if p.stem.split('.')[0] in wanted or p.stem in wanted]
    if args.skip:
        skip = set(args.skip)
        egrs = [p for p in egrs if p.stem.split('.')[0] not in skip and p.stem not in skip]
    print(f"# {len(egrs)} datasets to process (sorted by size, smallest first)",
          file=sys.stderr)

    tmp_root = Path(tempfile.mkdtemp(prefix="chroma_consist_"))
    print(f"# tmp: {tmp_root}", file=sys.stderr)

    results = []
    fmt = ("{stem:32s}  cons={cons:6.4f}  tie_ref={tr:6.4f}  tie_test={tt:6.4f}  "
           "distinct_test={dt:>8d}  ref_ms={r:>8.0f}  test_ms={s:>8.0f}  "
           "metric_ms={m:>7.0f}")
    for egr in egrs:
        stem = egr.stem
        ref_path  = tmp_root / f"ref_{stem}.bin"
        test_path = tmp_root / f"test_{stem}.bin"

        # --- 1) JP-SL^M (cpu_SDL)
        out_r, ref_dt, err = run_cmd(
            [args.ref_bin, str(egr), "--dump-priority", str(ref_path)],
            args.ref_timeout, "ref")
        if err:
            print(f"# SKIP {stem}: ref failed: {err}", file=sys.stderr)
            results.append({"graph": stem, "error": f"ref: {err}",
                            "ref_wall_s": ref_dt})
            continue

        # --- 2) cuSL_ELS_SDC -e <theta>
        out_t, test_dt, err = run_cmd(
            [args.test_bin, "-f", str(egr),
             "-a", args.algo, "-e", str(args.theta),
             "--dump-priority", str(test_path)],
            args.test_timeout, "test")
        if err:
            print(f"# SKIP {stem}: test failed: {err}", file=sys.stderr)
            results.append({"graph": stem, "error": f"test: {err}",
                            "ref_wall_s": ref_dt, "test_wall_s": test_dt})
            ref_path.unlink(missing_ok=True)
            continue

        # --- 3) metric tool
        out_m, metric_dt, err = run_cmd(
            [args.metric_bin, str(egr), str(ref_path), str(test_path)],
            args.metric_timeout, "metric")
        ref_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)
        if err:
            print(f"# SKIP {stem}: metric failed: {err}", file=sys.stderr)
            results.append({"graph": stem, "error": f"metric: {err}",
                            "ref_wall_s": ref_dt, "test_wall_s": test_dt,
                            "metric_wall_s": metric_dt})
            continue

        try:
            data = json.loads(out_m.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as e:
            print(f"# SKIP {stem}: metric parse: {e}", file=sys.stderr)
            results.append({"graph": stem, "error": f"metric parse: {e}",
                            "raw": out_m.stdout, "ref_wall_s": ref_dt,
                            "test_wall_s": test_dt, "metric_wall_s": metric_dt})
            continue

        data["graph"] = stem
        data["ref_wall_s"]    = ref_dt
        data["test_wall_s"]   = test_dt
        data["metric_wall_s"] = metric_dt
        data["theta"]         = args.theta
        data["algo"]          = args.algo
        results.append(data)

        print(fmt.format(
            stem=stem,
            cons=data["consistency_ratio"],
            tr=data["tie_ratio_ref"],
            tt=data["tie_ratio_test"],
            dt=data["distinct_test"],
            r=ref_dt * 1000, s=test_dt * 1000, m=metric_dt * 1000),
              file=sys.stderr, flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"# wrote {len(results)} results to {args.out}", file=sys.stderr)
    shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Multi-framework PA / colour sweep driver.

For each (configuration, dataset) pair, calls `tools/pa_dumper/pa_dumper`
to dump the priority list, parses the machine-readable result block to
extract colour count + PA/CA timings, then optionally computes the
consistency ratio + adjacent-pair tie ratio between every non-baseline
configuration and a designated baseline configuration on each dataset.

Per-configuration entry format (CLI: --config "<label>:<algo>[:theta]" or
"<label>:<algo>:<theta>:baseline"; can repeat):

    --config "JP-SL_M:SDL:0:baseline" \
    --config "cuSL_ELS_SDC_t5:cuSL_ELS_SDC:5"

Without --config flags, the DEFAULT_CONFIGS list below is used.

Run:
    python3 scripts/run_pa_sweep.py            # all defaults, all .egr
    python3 scripts/run_pa_sweep.py --only facebook le450_25d
    python3 scripts/run_pa_sweep.py --no-consistency
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    label:    str
    algo:     str
    theta:    int  = 0
    predict:  bool = False     # if True, ignore `theta` and pass --predict
    baseline: bool = False

    @classmethod
    def parse(cls, spec: str) -> "Config":
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(
                f"Bad --config spec '{spec}'. "
                "Use label:algo[:theta_or_predict[:baseline]]")
        label = parts[0]
        algo  = parts[1]
        theta_str = parts[2] if len(parts) > 2 else ""
        predict = (theta_str.lower() == "predict")
        theta   = 0 if predict or theta_str == "" else int(theta_str)
        baseline = (len(parts) > 3 and parts[3].lower() == "baseline")
        return cls(label=label, algo=algo, theta=theta,
                   predict=predict, baseline=baseline)


# Edit this list to change the default sweep set.
DEFAULT_CONFIGS = [
    Config("JP-SL_M",            "SDL",              0,  baseline=True),
    Config("cuSL_ELS",           "cuSL_ELS",         0),
    Config("cuSL_ELS_SDC",       "cuSL_ELS_SDC",     0),
    Config("cuSL_ELS_SDC_t5",    "cuSL_ELS_SDC",     5),
    Config("cuSL_ELS_SDC_t10",   "cuSL_ELS_SDC",    10),
    Config("JP_SL",              "JP_SL",            0),
    Config("JP_SLL",             "JP_SLL",           0),
    Config("JP_ADG",             "JP_ADG",           0),
]


RESULT_BLOCK_RE = re.compile(r"---PA_DUMPER_RESULT---\n(.*?)\n---PA_DUMPER_END---",
                             re.DOTALL)


def parse_args():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=str(repo / "Datasets" / "EGR"))
    p.add_argument("--pa-dumper",
                   default=str(repo / "tools" / "pa_dumper" / "pa_dumper"))
    p.add_argument("--metric-bin",
                   default=str(repo / "scripts" / "consistency_metric"))
    p.add_argument("--out", default=str(repo / "scripts" / "pa_sweep_results.json"))
    p.add_argument("--config", action="append", default=None,
                   help="Override DEFAULT_CONFIGS; "
                        "format: label:algo[:theta[:baseline]]; can repeat")
    p.add_argument("--only", nargs="*", default=None,
                   help="Restrict to dataset stems (e.g. facebook le450_25d)")
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--no-consistency", action="store_true",
                   help="Skip pairwise consistency vs baseline (only collect "
                        "colour usage + PA/CA timing)")
    p.add_argument("--keep-priority", action="store_true",
                   help="Keep dumped priority files for inspection (in --tmp-dir)")
    p.add_argument("--tmp-dir", default=None,
                   help="Temp dir for priority dumps (default: mkdtemp)")
    p.add_argument("--gpu-timeout", type=int, default=600)
    p.add_argument("--cpu-timeout", type=int, default=2400)
    p.add_argument("--metric-timeout", type=int, default=900)
    return p.parse_args()


def parse_result_block(stdout: str) -> Optional[dict]:
    m = RESULT_BLOCK_RE.search(stdout)
    if not m:
        return None
    out = {}
    for line in m.group(1).splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    # Coerce numeric fields
    for k in ("nodes", "edges", "theta", "colors_used",
              "colors_before_reduction", "colors_after_reduction"):
        if k in out:
            try: out[k] = int(out[k])
            except ValueError: pass
    for k in ("pa_ms", "ca_ms", "reduce_ms", "total_ms"):
        if k in out:
            try: out[k] = float(out[k])
            except ValueError: pass
    return out


def run_pa_dumper(args, cfg: Config, egr: Path, dump_path: Path):
    is_cpu = (cfg.algo == "SDL")
    timeout = args.cpu_timeout if is_cpu else args.gpu_timeout
    cmd = [args.pa_dumper, "-f", str(egr), "-a", cfg.algo,
           "--dump-priority", str(dump_path)]
    if cfg.predict:
        cmd.append("--predict")
    else:
        cmd += ["-e", str(cfg.theta)]
    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0, f"TIMEOUT after {timeout}s"
    dt = time.perf_counter() - t0
    if out.returncode != 0:
        tail = (out.stderr or out.stdout or "").splitlines()[-3:]
        return None, dt, f"rc={out.returncode}: {' | '.join(tail)}"
    block = parse_result_block(out.stdout)
    if block is None:
        return None, dt, "no result block in stdout"
    return block, dt, None


def run_metric(args, egr: Path, ref: Path, test: Path):
    cmd = [args.metric_bin, str(egr), str(ref), str(test)]
    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=args.metric_timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0, f"TIMEOUT after {args.metric_timeout}s"
    dt = time.perf_counter() - t0
    if out.returncode != 0:
        tail = (out.stderr or out.stdout or "").splitlines()[-3:]
        return None, dt, f"rc={out.returncode}: {' | '.join(tail)}"
    try:
        return json.loads(out.stdout.strip().splitlines()[-1]), dt, None
    except Exception as e:
        return None, dt, f"parse: {e}"


def main():
    args = parse_args()

    configs = ([Config.parse(s) for s in args.config]
               if args.config else list(DEFAULT_CONFIGS))
    if not args.no_consistency:
        baselines = [c for c in configs if c.baseline]
        if len(baselines) != 1:
            print(f"ERROR: need exactly one baseline (got {len(baselines)}). "
                  "Mark with ':baseline' suffix.", file=sys.stderr)
            sys.exit(1)
        baseline = baselines[0]
    else:
        baseline = None

    for path in (args.pa_dumper, args.metric_bin if not args.no_consistency else None):
        if path and not Path(path).exists():
            print(f"ERROR: missing binary: {path}", file=sys.stderr)
            sys.exit(1)

    dataset_dir = Path(args.dataset_dir)
    egrs = sorted(dataset_dir.glob("*.egr"), key=lambda p: p.stat().st_size)
    if args.only:
        wanted = set(args.only)
        egrs = [p for p in egrs
                if p.stem.split('.')[0] in wanted or p.stem in wanted]
    if args.skip:
        skip = set(args.skip)
        egrs = [p for p in egrs
                if p.stem.split('.')[0] not in skip and p.stem not in skip]

    print(f"# {len(egrs)} datasets x {len(configs)} configs",
          file=sys.stderr)
    print(f"# baseline: {baseline.label if baseline else '(none)'}",
          file=sys.stderr)

    if args.tmp_dir:
        tmp_root = Path(args.tmp_dir); tmp_root.mkdir(parents=True, exist_ok=True)
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="pa_sweep_"))
    print(f"# tmp dir: {tmp_root}", file=sys.stderr)

    rows = []   # one row per (dataset, config)
    cmp_rows = []  # one row per (dataset, non-baseline config) consistency

    for egr in egrs:
        stem = egr.stem
        per_config_dump = {}   # cfg.label → dump path (only if pa_dumper succeeded)
        per_config_block = {}  # cfg.label → result dict
        for cfg in configs:
            dump_path = tmp_root / f"{stem}__{cfg.label}.bin"
            block, wall, err = run_pa_dumper(args, cfg, egr, dump_path)
            row = {
                "graph":      stem,
                "config":     cfg.label,
                "algo":       cfg.algo,
                "theta":      cfg.theta,
                "baseline":   cfg.baseline,
                "wall_s":     wall,
            }
            if err:
                row["error"] = err
                print(f"# {stem} / {cfg.label}: FAIL ({err})", file=sys.stderr)
            else:
                row.update({k: block.get(k) for k in (
                    "nodes", "edges", "pa_ms", "ca_ms", "reduce_ms", "total_ms",
                    "colors_used", "colors_before_reduction",
                    "colors_after_reduction", "framework",
                )})
                # When --predict is set, the actual θ used is decided inside
                # pa_dumper. Pull it out of the result block so the JSON
                # reflects what really ran (the cfg.theta is meaningless then).
                if cfg.predict and block.get("theta") is not None:
                    row["theta"] = block["theta"]
                    row["predict_used"] = True
                per_config_dump[cfg.label]  = dump_path
                per_config_block[cfg.label] = row
                print(f"# {stem:32s} {cfg.label:24s} colors={row['colors_used']:4d} "
                      f"pa={row['pa_ms']:8.2f}ms ca={row['ca_ms']:8.2f}ms "
                      f"wall={wall:6.2f}s", file=sys.stderr)
            rows.append(row)

        # Pairwise consistency vs baseline (skip baseline-vs-baseline)
        if baseline and baseline.label in per_config_dump:
            ref_dump = per_config_dump[baseline.label]
            for cfg in configs:
                if cfg.label == baseline.label:
                    continue
                if cfg.label not in per_config_dump:
                    continue
                test_dump = per_config_dump[cfg.label]
                metric, mwall, merr = run_metric(args, egr, ref_dump, test_dump)
                cmp_row = {
                    "graph":         stem,
                    "baseline":      baseline.label,
                    "test":          cfg.label,
                    "algo":          cfg.algo,
                    "theta":         cfg.theta,
                    "metric_wall_s": mwall,
                }
                if merr:
                    cmp_row["error"] = merr
                    print(f"# {stem} / {cfg.label} vs {baseline.label}: "
                          f"metric FAIL ({merr})", file=sys.stderr)
                else:
                    cmp_row.update({k: metric.get(k) for k in (
                        "consistency_ratio", "tie_ratio_ref", "tie_ratio_test",
                        "distinct_ref", "distinct_test",
                        "concordant_unord_pairs",
                    )})
                    print(f"# {stem:32s} {cfg.label:24s} "
                          f"vs {baseline.label}: cons={cmp_row['consistency_ratio']:.4f} "
                          f"tie_test={cmp_row['tie_ratio_test']:.4f}",
                          file=sys.stderr)
                cmp_rows.append(cmp_row)

        # Cleanup priority dumps for this dataset (unless --keep-priority)
        if not args.keep_priority:
            for p in per_config_dump.values():
                try: p.unlink()
                except FileNotFoundError: pass

    # Save aggregated results
    summary = {
        "configs":       [asdict(c) for c in configs],
        "baseline":      baseline.label if baseline else None,
        "per_run":       rows,
        "consistency":   cmp_rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)
    print(f"#   {len(rows)} per-run rows, {len(cmp_rows)} consistency rows",
          file=sys.stderr)

    # ── Pretty print: colour-count matrix (datasets × configs) ──────────
    by_graph = {}
    for r in rows:
        by_graph.setdefault(r["graph"], {})[r["config"]] = r
    cmp_lookup = {}
    for r in cmp_rows:
        cmp_lookup[(r["graph"], r["test"])] = r

    print()
    print("================================================================================")
    print(" Colors used (best of run; '-' = error/timeout)")
    print("================================================================================")
    header = f"{'dataset':<28s}" + "".join(f"{c.label:>14s}" for c in configs)
    print(header)
    for graph in sorted(by_graph.keys(),
                        key=lambda g: by_graph[g].get(configs[0].label, {}).get("nodes", 0)):
        line = f"{graph:<28s}"
        for c in configs:
            r = by_graph[graph].get(c.label)
            if r is None or "error" in r or r.get("colors_used") is None:
                line += f"{'-':>14s}"
            else:
                line += f"{r['colors_used']:>14d}"
        print(line)

    if baseline and cmp_rows:
        print()
        print("================================================================================")
        print(f" Consistency Ratio vs baseline '{baseline.label}'")
        print("================================================================================")
        cs = [c for c in configs if c.label != baseline.label]
        header = f"{'dataset':<28s}" + "".join(f"{c.label:>14s}" for c in cs)
        print(header)
        for graph in sorted(by_graph.keys(),
                            key=lambda g: by_graph[g].get(configs[0].label, {}).get("nodes", 0)):
            line = f"{graph:<28s}"
            for c in cs:
                r = cmp_lookup.get((graph, c.label))
                if r is None or "error" in r or r.get("consistency_ratio") is None:
                    line += f"{'-':>14s}"
                else:
                    line += f"{r['consistency_ratio']:>14.4f}"
            print(line)

        print()
        print("================================================================================")
        print(f" Adjacent-pair tie ratio (test list; ref = {baseline.label} is always 0)")
        print("================================================================================")
        header = f"{'dataset':<28s}" + "".join(f"{c.label:>14s}" for c in cs)
        print(header)
        for graph in sorted(by_graph.keys(),
                            key=lambda g: by_graph[g].get(configs[0].label, {}).get("nodes", 0)):
            line = f"{graph:<28s}"
            for c in cs:
                r = cmp_lookup.get((graph, c.label))
                if r is None or "error" in r or r.get("tie_ratio_test") is None:
                    line += f"{'-':>14s}"
                else:
                    line += f"{r['tie_ratio_test']:>14.4f}"
            print(line)

    if not args.keep_priority:
        try: tmp_root.rmdir()
        except OSError: pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

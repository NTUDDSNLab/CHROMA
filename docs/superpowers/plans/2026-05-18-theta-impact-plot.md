# θ-Impact Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained `scripts/plots/theta_impact/` pipeline that sweeps CHROMA's `cuSL_ELS_SDC` over θ=0–20 for as-skitter/cit-Patents/europe_osm and redraws paper Fig. 6 with the predicted-θ star relabelled **CEP theta (v0_paper)** plus a new diamond **AEP theta (v3_raw)**.

**Architecture:** Two decoupled scripts joined by a JSON contract. `theta_impact.py` invokes the CHROMA binary (θ sweep keep-best + two deterministic predicted-θ reads) and writes `theta_impact_results.json`. `plot_theta_impact.py` consumes only that JSON and renders a 3-subplot twin-axis figure. A `README.md` documents the 2-step workflow. Figure/JSON are gitignored on this branch (regenerable); the committed deliverable is the pipeline + sweep log.

**Tech Stack:** Python 3 (argparse, json, re, struct, subprocess), matplotlib (Agg), numpy. Reuses `scripts/grid_elastic.py`'s output regexes and keep-best rule; mirrors the structure of `scripts/plots/priority_consistency/`.

**Spec:** `docs/superpowers/specs/2026-05-18-theta-impact-plot-design.md`

---

### Task 1: Sweep driver `theta_impact.py`

**Files:**
- Create: `scripts/plots/theta_impact/theta_impact.py`

- [ ] **Step 1: Create the sweep driver**

Create `scripts/plots/theta_impact/theta_impact.py` with exactly this content:

```python
#!/usr/bin/env python3
"""Sweep θ-impact for the paper Fig. 6 redraw.

For each dataset (default: as-skitter, cit-Patents, europe_osm):
  - θ = 0..THETA_MAX: run `CHROMA -a cuSL_ELS_SDC -e <θ>` RUNS times,
    keep the best run (min colors, tie -> min runtime); record
    {color, runtime_ms, iter_count}.
  - CEP θ: one run  CHROMA -a cuSL_ELS_SDC --predict --predict-model v0_paper
  - AEP θ: one run  CHROMA -a cuSL_ELS_SDC --predict --predict-model v3
                           --no-dynamic-theta
Writes theta_impact_results.json consumed by plot_theta_impact.py.

The `EGC θ: N (Predicted)` line reports the predictor's *initial* θ
(before any online bumping), so the parsed value is independent of
whether the dynamic-θ controller is on; CEP follows the paper's
v0_paper path, AEP adds --no-dynamic-theta per the v3_raw definition.

Examples:
    python3 scripts/plots/theta_impact/theta_impact.py
    python3 scripts/plots/theta_impact/theta_impact.py \\
        --datasets cit-Patents --theta-max 3 --runs 2 --out /tmp/ti.json
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

# Patterns for CHROMA verbose single-run output. CHROMA prints phase
# timings ("PA runtime:", "CA runtime:", ...) BEFORE the "Total
# runtime:" line, so RUNTIME_RE is anchored to "Total runtime:" — a
# bare "runtime:" (as grid_elastic.py uses) would match PA time first.
# COLORS_RE / ITER_RE mirror grid_elastic.py; PRED_RE matches
# "EGC θ: <d> (Predicted)".
RUNTIME_RE = re.compile(r"Total\s+runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
COLORS_RE = re.compile(r"colors\s+used:\s*(\d+)", re.IGNORECASE)
ITER_RE = re.compile(r"Iter\s+count:\s*(\d+)", re.IGNORECASE)
PRED_RE = re.compile(r"EGC[^:]*:\s*(\d+)\s*\(Predicted\)")


def read_egr_header(path: Path) -> tuple:
    """Return (nodes, edges) from the .egr CSR binary header."""
    with open(path, "rb") as f:
        nodes, edges = struct.unpack("<ii", f.read(8))
    return nodes, edges


def resolve_egr(ds_dir: Path, stem: str) -> Optional[Path]:
    """Stem -> .egr, .col double-suffix aware (mirrors run_pa_sweep.py)."""
    for p in sorted(ds_dir.glob("*.egr")):
        if p.stem.split('.')[0] == stem or p.stem == stem:
            return p
    return None


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


def parse_sweep(stdout: str) -> Optional[dict]:
    mr, mc = RUNTIME_RE.search(stdout), COLORS_RE.search(stdout)
    if not (mr and mc):
        return None
    mi = ITER_RE.search(stdout)
    return {
        "runtime_ms": float(mr.group(1)),
        "color": int(mc.group(1)),
        "iter_count": int(mi.group(1)) if mi else None,
    }


def best_of(runs: list) -> Optional[dict]:
    """grid_elastic keep-best: min colors, tie -> min runtime."""
    if not runs:
        return None
    return sorted(runs, key=lambda x: (x["color"], x["runtime_ms"]))[0]


def predicted_theta(binary, egr, algo, model, no_bump, timeout) -> tuple:
    cmd = [binary, "-f", str(egr), "-a", algo, "--predict",
           "--predict-model", model]
    if no_bump:
        cmd.append("--no-dynamic-theta")
    r, _dt, err = run(cmd, timeout)
    if err:
        return None, err
    m = PRED_RE.search(r.stdout)
    if not m:
        return None, "no '(Predicted)' line in output"
    return int(m.group(1)), None


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--binary", default=str(repo / "CHROMA" / "CHROMA"))
    ap.add_argument("--dataset-dir", default=str(repo / "Datasets" / "EGR"))
    ap.add_argument("--datasets", nargs="+",
                    default=["as-skitter", "cit-Patents", "europe_osm"])
    ap.add_argument("--algo", default="cuSL_ELS_SDC")
    ap.add_argument("--theta-max", type=int, default=20)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1200,
                    help="Per-CHROMA-invocation timeout (seconds)")
    ap.add_argument("--out", default=str(here / "theta_impact_results.json"))
    args = ap.parse_args()

    if not Path(args.binary).exists():
        sys.exit(f"ERROR: CHROMA binary not found at {args.binary} "
                 f"(build CHROMA/ with PRE_MODEL=1 for --predict)")
    ds_dir = Path(args.dataset_dir)

    data = {}
    for stem in args.datasets:
        egr = resolve_egr(ds_dir, stem)
        if egr is None:
            print(f"# {stem:16s} SKIP: no .egr in {ds_dir}", file=sys.stderr)
            data[stem] = {"error": f"no .egr for {stem} in {ds_dir}"}
            continue
        nodes, edges = read_egr_header(egr)
        entry = {"nodes": nodes, "edges": edges, "sweep": {}}
        for theta in range(0, args.theta_max + 1):
            recs = []
            last_err = None
            for _ in range(args.runs):
                r, _dt, err = run(
                    [args.binary, "-f", str(egr), "-a", args.algo,
                     "-e", str(theta)], args.timeout)
                if err:
                    last_err = err
                    continue
                rec = parse_sweep(r.stdout)
                if rec is None:
                    last_err = "parse-failed (no runtime/colors line)"
                    continue
                recs.append(rec)
            best = best_of(recs)
            if best is None:
                entry["sweep"][str(theta)] = {
                    "error": last_err or "all runs failed"}
                print(f"# {stem:16s} θ={theta:2d} FAIL ({last_err})",
                      file=sys.stderr)
            else:
                entry["sweep"][str(theta)] = best
                print(f"# {stem:16s} θ={theta:2d} colors={best['color']:3d} "
                      f"rt={best['runtime_ms']:9.3f}ms "
                      f"iter={best['iter_count']}", file=sys.stderr)
        cep, cep_err = predicted_theta(args.binary, egr, args.algo,
                                       "v0_paper", False, args.timeout)
        aep, aep_err = predicted_theta(args.binary, egr, args.algo,
                                       "v3", True, args.timeout)
        entry["cep_theta"] = cep
        entry["aep_theta"] = aep
        print(f"# {stem:16s} CEP θ={cep} ({cep_err or 'ok'}) | "
              f"AEP θ={aep} ({aep_err or 'ok'})", file=sys.stderr)
        data[stem] = entry

    out = {
        "datasets": args.datasets,
        "algo": args.algo,
        "theta_max": args.theta_max,
        "runs": args.runs,
        "data": data,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Syntax check**

Run: `python3 -m py_compile scripts/plots/theta_impact/theta_impact.py`
Expected: exit 0, no output.

- [ ] **Step 3: CLI smoke (no binary needed)**

Run: `python3 scripts/plots/theta_impact/theta_impact.py --help`
Expected: usage text listing `--binary --dataset-dir --datasets --algo --theta-max --runs --timeout --out`, exit 0.

- [ ] **Step 4: Missing-binary guard**

Run: `python3 scripts/plots/theta_impact/theta_impact.py --binary /no/such/chroma --datasets cit-Patents --out /tmp/ti_x.json`
Expected: non-zero exit, stderr contains `ERROR: CHROMA binary not found` and `PRE_MODEL=1`.

- [ ] **Step 5: Real reduced sweep (requires CHROMA built with PRE_MODEL=1)**

First confirm the binary exists and supports the flags:
Run: `test -x CHROMA/CHROMA && grep -q -- '--predict-model' CHROMA/CHROMA.cu && echo OK`
Expected: `OK`. (If `CHROMA/CHROMA` is missing, build it: `cd CHROMA && make ARCH=sm_89 PRE_MODEL=1` — ARCH must match the GPU.)

Run: `python3 scripts/plots/theta_impact/theta_impact.py --datasets cit-Patents --theta-max 3 --runs 2 --out /tmp/ti_smoke.json`
Expected: stderr shows `cit-Patents θ= 0..3` lines plus a `CEP θ=<int> ... | AEP θ=<int> ...` line and `# wrote /tmp/ti_smoke.json`; exit 0.

- [ ] **Step 6: Validate JSON contract**

Run:
```bash
python3 - <<'EOF'
import json
d = json.load(open("/tmp/ti_smoke.json"))
assert d["datasets"] == ["cit-Patents"], d["datasets"]
assert d["algo"] == "cuSL_ELS_SDC" and d["theta_max"] == 3 and d["runs"] == 2
e = d["data"]["cit-Patents"]
assert e["nodes"] > 0 and e["edges"] > 0, e
sw = e["sweep"]
assert sorted(sw) == ["0", "1", "2", "3"], sorted(sw)
for k, v in sw.items():
    assert "error" not in v, (k, v)
    assert isinstance(v["color"], int) and v["color"] > 0, (k, v)
    assert isinstance(v["runtime_ms"], float) and v["runtime_ms"] > 0, (k, v)
    assert v["iter_count"] is None or isinstance(v["iter_count"], int), (k, v)
assert isinstance(e["cep_theta"], int) and e["cep_theta"] >= 0, e["cep_theta"]
assert isinstance(e["aep_theta"], int) and e["aep_theta"] >= 0, e["aep_theta"]
print("OK", "cep", e["cep_theta"], "aep", e["aep_theta"])
EOF
```
Expected: prints `OK cep <int> aep <int>`, exit 0.

- [ ] **Step 7: Commit**

```bash
git reset -q
git add scripts/plots/theta_impact/theta_impact.py
git commit -m "scripts/plots/theta_impact: add theta-sweep + CEP/AEP-theta driver

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Plot renderer `plot_theta_impact.py`

**Files:**
- Create: `scripts/plots/theta_impact/plot_theta_impact.py`

- [ ] **Step 1: Create the plot script**

Create `scripts/plots/theta_impact/plot_theta_impact.py` with exactly this content:

```python
#!/usr/bin/env python3
"""Render the θ-impact figure (paper Fig. 6 redraw).

One row of subplots (default datasets: as-skitter, cit-Patents,
europe_osm). Per subplot: x = θ; left y = runtime (ms) as bars coloured
by #colors used (discrete per-subplot legend `color = N`); right y =
iteration count (line + markers). A star marks CEP θ (v0_paper) and a
diamond marks AEP θ (v3_raw), drawn near y=0 on the left axis; a marker
is omitted when its predicted θ is null.

Examples:
    python3 scripts/plots/theta_impact/plot_theta_impact.py
    python3 scripts/plots/theta_impact/plot_theta_impact.py \\
        --in scripts/plots/theta_impact/theta_impact_results.json \\
        --figsize 16 4
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

SUBPLOT_TAGS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
COLOR_CYCLE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
               "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
CEP_COLOR, AEP_COLOR = "#D62728", "#17BECF"
TICK_FS, LABEL_FS, LEGEND_FS, TAG_FS = 10, 12, 9, 12


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--in", dest="in_path",
                    default=str(here / "theta_impact_results.json"))
    ap.add_argument("--out-prefix", default=str(here / "theta_impact"))
    ap.add_argument("--figsize", nargs=2, type=float, default=[15.0, 4.0],
                    metavar=("W", "H"))
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run theta_impact.py first.",
              file=sys.stderr)
        return 1

    d = json.loads(in_path.read_text())
    datasets = d["datasets"]
    theta_max = d["theta_max"]
    thetas = list(range(0, theta_max + 1))

    fig, axes = plt.subplots(1, len(datasets), figsize=tuple(args.figsize),
                             constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]

    for si, stem in enumerate(datasets):
        ax = axes[si]
        ax2 = ax.twinx()
        entry = d["data"].get(stem, {})
        sweep = entry.get("sweep", {})

        runtimes, iters, colors = [], [], []
        for t in thetas:
            cell = sweep.get(str(t), {})
            if not cell or "error" in cell:
                runtimes.append(0.0)
                iters.append(np.nan)
                colors.append(None)
            else:
                runtimes.append(cell["runtime_ms"])
                ic = cell.get("iter_count")
                iters.append(ic if ic is not None else np.nan)
                colors.append(cell["color"])

        uniq = sorted({c for c in colors if c is not None})
        cmap = {c: COLOR_CYCLE[i % len(COLOR_CYCLE)]
                for i, c in enumerate(uniq)}
        bar_colors = [cmap.get(c, "#CCCCCC") for c in colors]
        ax.bar(thetas, runtimes, width=0.8, color=bar_colors,
               edgecolor="black", linewidth=0.4, zorder=2)
        ax2.plot(thetas, iters, color="#1f1f1f", marker="o",
                 markersize=3, linewidth=1.3, zorder=3)

        cep = entry.get("cep_theta")
        aep = entry.get("aep_theta")
        ymax = max([r for r in runtimes if r > 0] or [1.0])
        y0 = ymax * 0.02
        if cep is not None:
            ax.scatter([cep], [y0], marker="*", s=240, color=CEP_COLOR,
                       edgecolor="black", linewidth=0.5, zorder=5)
        if aep is not None:
            ax.scatter([aep], [y0], marker="D", s=90, color=AEP_COLOR,
                       edgecolor="black", linewidth=0.5, zorder=5)

        ax.set_xlim(-0.6, theta_max + 0.6)
        ax.set_xticks(range(0, theta_max + 1, max(1, theta_max // 10)))
        ax.set_ylim(bottom=0.0)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax2.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_xlabel(r"$\theta$", fontsize=LABEL_FS)
        if si == 0:
            ax.set_ylabel("runtime (ms)", fontsize=LABEL_FS)
        if si == len(datasets) - 1:
            ax2.set_ylabel("iteration count", fontsize=LABEL_FS)
        ax.set_title(f"{SUBPLOT_TAGS[si]} {stem}", fontsize=TAG_FS)

        handles = [mpatches.Patch(facecolor=cmap[c], edgecolor="black",
                                  linewidth=0.4, label=f"color = {c}")
                   for c in uniq]
        if cep is not None:
            handles.append(mlines.Line2D(
                [], [], linestyle="none", marker="*", markersize=13,
                color=CEP_COLOR, markeredgecolor="black",
                label="CEP theta (v0_paper)"))
        if aep is not None:
            handles.append(mlines.Line2D(
                [], [], linestyle="none", marker="D", markersize=8,
                color=AEP_COLOR, markeredgecolor="black",
                label="AEP theta (v3_raw)"))
        if handles:
            ax.legend(handles=handles, fontsize=LEGEND_FS, frameon=True,
                      loc="upper right", handlelength=1.2, borderpad=0.4)

    pdf = Path(args.out_prefix + ".pdf")
    png = Path(args.out_prefix + ".png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"# wrote {pdf}\n# wrote {png}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Syntax check**

Run: `python3 -m py_compile scripts/plots/theta_impact/plot_theta_impact.py`
Expected: exit 0, no output.

- [ ] **Step 3: Missing-input guard**

Run: `python3 scripts/plots/theta_impact/plot_theta_impact.py --in /no/such.json --out-prefix /tmp/ti_none`
Expected: non-zero exit (1), stderr contains `ERROR:` and `Run theta_impact.py first`.

- [ ] **Step 4: Synthetic-JSON render smoke**

Run:
```bash
python3 - <<'EOF'
import json
d = {"datasets": ["as-skitter", "cit-Patents", "europe_osm"],
     "algo": "cuSL_ELS_SDC", "theta_max": 5, "runs": 5, "data": {}}
for i, s in enumerate(d["datasets"]):
    sw = {}
    for t in range(6):
        sw[str(t)] = {"color": 70 - (t // 2) + i,
                      "runtime_ms": 400.0 / (t + 1) + i,
                      "iter_count": 900 - 60 * t}
    sw["3"] = {"error": "synthetic gap"}        # exercise the gap path
    d["data"][s] = {"nodes": 1000 + i, "edges": 5000 + i, "sweep": sw,
                    "cep_theta": 2, "aep_theta": 4 if i else None}
json.dump(d, open("/tmp/ti_syn.json", "w"))
print("wrote /tmp/ti_syn.json")
EOF
python3 scripts/plots/theta_impact/plot_theta_impact.py \
    --in /tmp/ti_syn.json --out-prefix /tmp/ti_syn
test -s /tmp/ti_syn.pdf && test -s /tmp/ti_syn.png && echo "FILES OK"
```
Expected: `wrote /tmp/ti_syn.json`, `# wrote /tmp/ti_syn.pdf`, `# wrote /tmp/ti_syn.png`, `FILES OK`; exit 0. (The `aep_theta: null` on as-skitter exercises the omit-marker path; the `"3": error` cell exercises the gap path.)

- [ ] **Step 5: Commit**

```bash
git reset -q
git add scripts/plots/theta_impact/plot_theta_impact.py
git commit -m "scripts/plots/theta_impact: add Fig.6 renderer (CEP star + AEP diamond)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `README.md`

**Files:**
- Create: `scripts/plots/theta_impact/README.md`

- [ ] **Step 1: Create the README**

Create `scripts/plots/theta_impact/README.md` with exactly this content:

````markdown
# θ-Impact Plot (paper Fig. 6 redraw)

Redraws Fig. 6 of `CHROMA_IPDPSW_26.pdf` — `cuSL_ELS_SDC` runtime, color
count, and iteration count across θ — for **as-skitter**, **cit-Patents**,
**europe_osm**. The single paper "Predicted θ" star becomes two markers:

| Marker | Meaning | How it is obtained |
|--------|---------|--------------------|
| ★ CEP theta (v0_paper) | paper-era predictor θ | `CHROMA -a cuSL_ELS_SDC --predict --predict-model v0_paper` |
| ◆ AEP theta (v3_raw)   | v3 predictor θ, online bumping off | `CHROMA -a cuSL_ELS_SDC --predict --predict-model v3 --no-dynamic-theta` |

`EGC θ: N (Predicted)` reports the predictor's *initial* θ (before any
online bumping), so each predicted-θ is a single deterministic run.

## Prerequisites

- `CHROMA/CHROMA` built with `PRE_MODEL=1` (needed for `--predict`;
  supports `--predict-model {v3,v0_paper}` and `--no-dynamic-theta`):
  `cd CHROMA && make ARCH=sm_89 PRE_MODEL=1` (set `ARCH` to your GPU).
- `Datasets/EGR/{as-skitter,cit-Patents,europe_osm}.egr` present.

## Step 1 — Sweep

```
python3 scripts/plots/theta_impact/theta_impact.py
```

For each dataset: θ = 0…20 runs `CHROMA -a cuSL_ELS_SDC -e <θ>` 5×,
keeps the best run (min colors, tie → min runtime), then one
deterministic CEP and one AEP predicted-θ run. Key flags: `--datasets`,
`--algo` (default `cuSL_ELS_SDC`), `--theta-max` (default 20), `--runs`
(default 5), `--timeout` (per-invocation seconds, default 1200),
`--binary`, `--dataset-dir`, `--out`. Writes
`scripts/plots/theta_impact/theta_impact_results.json` (gitignored under
the project `*.json` rule; regenerable). A failed θ cell is recorded
with an `error` field and drawn as a 0-height gap; a failed predicted-θ
run stores `null` and that marker is omitted.

## Step 2 — Plot

```
python3 scripts/plots/theta_impact/plot_theta_impact.py
```

Flags: `--in`, `--out-prefix`, `--figsize` (default 15×4). Writes
`theta_impact.{pdf,png}`. Each subplot: x = θ, left y = runtime (ms)
bars coloured by #colors (per-subplot `color = N` legend), right y =
iteration-count line, ★ CEP / ◆ AEP near y=0.

## Notes

- `europe_osm` is the slow part of the sweep (θ=0 / small-θ runs on the
  largest graph dominate wall time); raise `--timeout` if cells fail.
  The per-cell timeout absorbs hangs — a timed-out cell becomes a gap.
- Predicted-θ is deterministic, so CEP/AEP are 1 run each (not `--runs`).
- Smoke a single dataset fast:
  `python3 scripts/plots/theta_impact/theta_impact.py --datasets
  cit-Patents --theta-max 3 --runs 2 --out /tmp/ti.json`.
````

- [ ] **Step 2: Sanity check the README renders the two-step workflow**

Run: `grep -c -E 'theta_impact\.py|plot_theta_impact\.py' scripts/plots/theta_impact/README.md`
Expected: a count ≥ 3 (Step 1 sweep cmd + Step 2 plot cmd + the Notes
smoke cmd). The Prerequisites section names the `CHROMA/CHROMA` binary
and the dataset paths, not the `.py` scripts.

- [ ] **Step 3: Commit**

```bash
git reset -q
git add scripts/plots/theta_impact/README.md
git commit -m "scripts/plots/theta_impact: add README (2-step workflow, CEP/AEP)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: End-to-end validation + final figure

**Files:** (no source changes — integration run)

- [ ] **Step 1: Confirm build prerequisite**

Run: `test -x CHROMA/CHROMA && grep -q -- '--predict-model' CHROMA/CHROMA.cu && grep -q -- '--no-dynamic-theta' CHROMA/CHROMA.cu && echo READY`
Expected: `READY`. If not, build: `cd CHROMA && make ARCH=sm_89 PRE_MODEL=1` (ARCH must match the GPU).

- [ ] **Step 2: Reduced end-to-end (real binary, fast config)**

Run:
```bash
python3 scripts/plots/theta_impact/theta_impact.py \
    --datasets cit-Patents --theta-max 4 --runs 2 --out /tmp/ti_e2e.json
python3 scripts/plots/theta_impact/plot_theta_impact.py \
    --in /tmp/ti_e2e.json --out-prefix /tmp/ti_e2e
test -s /tmp/ti_e2e.pdf && test -s /tmp/ti_e2e.png && echo "E2E OK"
```
Expected: sweep stderr shows θ=0..4 + a `CEP θ=… | AEP θ=…` line; plot writes pdf+png; `E2E OK`; exit 0.

- [ ] **Step 3: Eyeball the reduced figure**

Open `/tmp/ti_e2e.png`. Sanity-check vs paper Fig. 6 intuition: runtime bars are tallest at small θ and shrink as θ grows; the iteration-count line trends downward; the ★ (CEP) and ◆ (AEP) sit at small θ near the x-axis; the per-subplot legend lists `color = N` entries plus the two θ markers. Note any discrepancy; if the figure is structurally wrong, fix the offending script and re-run Steps 2–3 before continuing.

- [ ] **Step 4: Full sweep (the deliverable run — long; europe_osm dominates)**

Run:
```bash
python3 scripts/plots/theta_impact/theta_impact.py \
    2>&1 | tee /tmp/theta_impact_sweep.log
python3 scripts/plots/theta_impact/plot_theta_impact.py
```
Expected: `scripts/plots/theta_impact/theta_impact_results.json` written with `data` for all three datasets (each `sweep` having keys `"0".."20"` and integer `cep_theta`/`aep_theta`), then `theta_impact.{pdf,png}` written. This is the heavy run (3 datasets × 21 θ × 5 + 6 predicted invocations); europe_osm small-θ runs dominate wall time — leave it running. If a θ cell times out it is recorded as an `error` gap and the sweep continues.

- [ ] **Step 5: Commit the sweep record**

The JSON, PDF and PNG are gitignored on this branch (regenerable). The
sweep stderr already contains every per-θ `colors/runtime_ms/iter` line
plus the `CEP θ=… | AEP θ=…` line, so it is a sufficient reproducibility
record. NOTE: `*.log` is also gitignored on this branch — store the
record as a non-ignored `.md` file (a fenced log), not `.log`:
```bash
git reset -q
mkdir -p scripts/plots/theta_impact/logs
{
  echo "# θ-Impact full-sweep record"
  echo
  echo "\`scripts/plots/theta_impact/theta_impact.py\` (defaults: 3 datasets,"
  echo "θ=0..20, 5 runs/θ keep-best) + \`plot_theta_impact.py\`. Generated"
  echo "$(date -Is). JSON/PDF/PNG are gitignored & regenerable."
  echo
  echo '```'
  cat /tmp/theta_impact_sweep.log
  echo '```'
} > scripts/plots/theta_impact/logs/theta_impact_sweep.md
git add scripts/plots/theta_impact/logs/theta_impact_sweep.md
git commit -m "scripts/plots/theta_impact: record full-sweep results

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- theta_impact.py (sweep + CEP/AEP θ → JSON) → Task 1 ✓
- plot_theta_impact.py (JSON → 3-subplot Fig. 6) → Task 2 ✓
- README.md → Task 3 ✓
- JSON contract shape (datasets/algo/theta_max/runs/data; per-θ color/runtime_ms/iter_count; cep_theta/aep_theta; error gaps; null predicted) → Task 1 Steps 5–6 + Task 2 Step 4 ✓
- Spec testing strategy items 1–5 → Task 1 Step 5 (build check), Task 1 Steps 5–6 (sweep + parse), Task 2 Steps 3–4 (plot smoke + missing input), Task 4 Steps 2–4 (full sweep + eyeball) ✓
- CEP plain `--predict-model v0_paper`; AEP adds `--no-dynamic-theta` → `predicted_theta()` called with `no_bump=False` (CEP) / `True` (AEP) in Task 1 ✓
- Sweep uses `-e <θ>` (CHROMA elastic, not `-r`) → Task 1 sweep cmd ✓
- Figure/JSON gitignored, pipeline+log committed → Task 4 Step 5 ✓

**Placeholder scan:** No TBD/TODO; every code step contains complete file content; every command has expected output. ✓

**Type consistency:** `parse_sweep` returns `{runtime_ms,color,iter_count}`; `best_of` sorts by `(color, runtime_ms)`; plot reads `cell["runtime_ms"]/cell["color"]/cell.get("iter_count")` and `entry["cep_theta"]/entry["aep_theta"]/entry["sweep"]` — all consistent with the JSON written by `main()`. `resolve_egr`/`read_egr_header`/`run`/`predicted_theta` signatures match their call sites. ✓

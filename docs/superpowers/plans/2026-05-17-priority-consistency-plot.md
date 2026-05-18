# Priority Consistency Ratio Plot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained sweep + plot pipeline under `scripts/plots/priority_consistency/` that reproduces paper Fig. 5 (priority-ordering consistency vs the JP-SL^A reference) with 8 comparison frameworks (incl. CHROMA_v2 v3_raw vs v3_bump).

**Architecture:** Add a `--dump <path>` flag to `CPU/Parallel/SL.cpp` so `cpu_SL` can emit the JP-SL^A reference priority list. A Python sweep driver dumps each framework's priority list (pa_dumper for JP/CHROMA/CHROMA⁺, the CHROMA binary for the predict-model variants, in-Python LDF synthesis for ECL-GC), runs the existing `scripts/consistency_metric` binary against the reference, and writes a JSON contract. A separate plot script renders the paper-Fig.5-style grouped bar chart from that JSON.

**Tech Stack:** C++17/OpenMP (cpu_SL), Python 3 (matplotlib + stdlib), existing `pa_dumper` / `CHROMA` / `consistency_metric` binaries.

**Spec:** `docs/superpowers/specs/2026-05-17-priority-consistency-plot-design.md`

---

## File map

| File | Action | Responsibility |
|------|--------|----------------|
| `CPU/Parallel/SL.cpp` | modify | Add `--dump <path>` → write JP-SL^A `priority[]` as uint32, then exit before CA |
| `scripts/plots/priority_consistency/sweep_priority_consistency.py` | create | Orchestrate dumps + consistency_metric → JSON |
| `scripts/plots/priority_consistency/plot_priority_consistency.py` | create | JSON → paper-Fig.5 grouped bar PDF/PNG |
| `scripts/plots/priority_consistency/README.md` | create | Usage guide |

`.egr` binary format (little-endian, from `lib/io/ECLgraph.h`): `int32 nodes`, `int32 edges`, `int32 nindex[nodes+1]`, `int32 nlist[edges]`, `int32 eweight[edges]` (eweight optional). ECL-GC LDF synthesis only needs `nodes` + `nindex`.

`consistency_metric` CLI: `consistency_metric <graph.egr> <ref.bin> <test.bin>` → one JSON line on stdout with keys incl. `consistency_ratio`, `tie_ratio_test`, `distinct_test`, `concordant_unord_pairs`. Each `.bin` is `nodes` little-endian `uint32` priority values; the metric dense-ranks each list independently before counting concordant pairs.

`pa_dumper` CLI: `pa_dumper -f <egr> -a <algo> --dump-priority <path> [-e <theta>] [--predict]` (algos incl. `JP_SLL`, `JP_ADG`, `cuSL_ELS`, `cuSL_ELS_SDC`). `CHROMA` CLI: `CHROMA -f <egr> -a cuSL_ELS_SDC --no-reduce --predict --predict-model {v0_paper,v3} --dump-priority <path>`.

---

## Task 1 — Add `--dump <path>` to `CPU/Parallel/SL.cpp`

**Why:** `cpu_SL` computes the JP-SL^A `priority[]` but `main()` only takes `<graph.egr> <threads>` and always runs the full coloring afterward. We need it to emit the priority list (uint32×nodes) and skip the CA stage when dumping. Non-dump behaviour must stay byte-identical.

**Files:**
- Modify: `CPU/Parallel/SL.cpp` (top includes; `main()` arg parsing; insertion after `compute_SL`)

- [ ] **Step 1: Ensure `<cstring>` is included**

Open `CPU/Parallel/SL.cpp`. Look at the top `#include` block. If `#include <cstring>` is absent, add it immediately after the first existing `#include` line (it is needed for `strcmp`/`strncmp`). `<cstdio>` is already available transitively via `ECLgraph.h`; `<vector>` is already used.

- [ ] **Step 2: Replace the `main()` argument-parsing head**

Find this exact block (note the single leading space on each line — preserve the file's existing indentation when you replace):

```cpp
 int main(int argc, char** argv)
 {
   if (argc != 3) {
     fprintf(stderr, "USAGE: %s <graph.egr> <threads>\n", argv[0]);
     return 0;
   }
   int threads = atoi(argv[2]);
   if (threads < 1) { fprintf(stderr, "threads must be >= 1\n"); return 0; }
 
   ECLgraph g = readECLgraph(argv[1]);
```

Replace it with:

```cpp
 int main(int argc, char** argv)
 {
   const char* dump_path  = nullptr;
   const char* graph_path = nullptr;
   int threads = 0;
   int pos = 0;
   for (int i = 1; i < argc; ++i) {
     if (strcmp(argv[i], "--dump") == 0) {
       if (i + 1 >= argc) { fprintf(stderr, "ERROR: --dump needs a path\n"); return 1; }
       dump_path = argv[++i];
     } else if (strncmp(argv[i], "--dump=", 7) == 0) {
       dump_path = argv[i] + 7;
     } else if (pos == 0) { graph_path = argv[i]; pos = 1; }
     else if (pos == 1) { threads = atoi(argv[i]); pos = 2; }
   }
   if (graph_path == nullptr || threads < 1) {
     fprintf(stderr, "USAGE: %s <graph.egr> <threads> [--dump <path>]\n", argv[0]);
     return 0;
   }
 
   ECLgraph g = readECLgraph(graph_path);
```

- [ ] **Step 3: Emit the priority dump and exit before CA**

Find this block:

```cpp
     PA_time.start();
     compute_SL(g, threads, priority);
     PA_time.stop();
     
 
     printf("Start init \n");
```

Replace it with:

```cpp
     PA_time.start();
     compute_SL(g, threads, priority);
     PA_time.stop();

     if (dump_path != nullptr) {
       // priority[] holds the JP-SL^A ordering key. consistency_metric
       // reads `nodes` uint32 values and dense-ranks them, so the raw
       // int bit pattern is exactly what it needs (ascending value =
       // earlier in SL order, matching pa_dumper / CHROMA dumps).
       FILE* df = fopen(dump_path, "wb");
       if (df == nullptr) {
         fprintf(stderr, "ERROR: cannot open %s for write\n", dump_path);
         return 1;
       }
       size_t w = fwrite(priority.data(), sizeof(int), (size_t)g.nodes, df);
       fclose(df);
       if (w != (size_t)g.nodes) {
         fprintf(stderr, "ERROR: short write to %s\n", dump_path);
         return 1;
       }
       printf("dumped %d priorities to %s\n", g.nodes, dump_path);
       return 0;
     }
 
     printf("Start init \n");
```

- [ ] **Step 4: Rebuild cpu_SL**

Run: `cd CPU/Parallel && make 2>&1 | tail -15 && cd ../..`

Expected: clean build producing `CPU/Parallel/cpu_SL`. If the Makefile target name differs, run `cd CPU/Parallel && make cpu_SL`.

- [ ] **Step 5: Smoke — non-dump path unchanged**

Run: `CPU/Parallel/cpu_SL Datasets/test/facebook.egr 8 2>&1 | tail -5`

Expected: still prints `result verification passed` and `colors used:` (behaviour identical to before this task).

- [ ] **Step 6: Smoke — dump path works**

Run:

```bash
CPU/Parallel/cpu_SL Datasets/test/facebook.egr 8 --dump /tmp/jpsla_fb.bin 2>&1 | tail -2
python3 -c "import os; n=os.path.getsize('/tmp/jpsla_fb.bin'); import struct; f=open('/home/chsieh45/PunchShadow/CHROMA/Datasets/test/facebook.egr','rb'); nodes=struct.unpack('<i',f.read(4))[0]; print('bytes',n,'nodes',nodes,'ok',n==nodes*4)"
```

Expected: prints `dumped <N> priorities to /tmp/jpsla_fb.bin` and `... ok True`. It must NOT print `Start init` (it returns before CA).

- [ ] **Step 7: Commit**

```bash
git add CPU/Parallel/SL.cpp
git commit -m "$(cat <<'EOF'
CPU/Parallel/SL.cpp: add --dump <path> to emit JP-SL^A priority list

cpu_SL now accepts an optional --dump <path> (or --dump=<path>); when
given, it writes the computed JP-SL^A priority[] as uint32-per-vertex
and exits before the color-allocation stage. Without --dump the
behaviour and CLI are byte-identical. Enables priority-consistency
sweeps to use cpu_SL as the JP-SL^A reference.
EOF
)"
```

---

## Task 2 — Validate the dump encoding (deterministic, no inversion)

**Why:** `consistency_metric` is directional (it counts concordant ordered pairs). JP-SL^A is a **partial order** — batch removal gives every vertex peeled in the same round the same priority — so a dump's self-consistency is `1 − tied_pair_fraction` (≈ 0.993 on facebook), NOT 1.0. The real correctness gate is: (a) two `cpu_SL` runs produce byte-identical dumps (determinism), (b) self-comparison equals that maximum-achievable ratio, (c) a cross-check vs an all-distinct SL dump stays high (no encoding inversion).

**Files:** none (verification only)

- [ ] **Step 1: Confirm the metric binary exists**

Run: `test -x scripts/consistency_metric && echo OK || echo "MISSING: build via the comment header in scripts/consistency_metric.cpp"`

Expected: `OK`. If MISSING: `g++ -O3 -std=c++17 -Ilib/io scripts/consistency_metric.cpp -o scripts/consistency_metric`.

- [ ] **Step 2: Ref-vs-ref consistency**

Run:

```bash
CPU/Parallel/cpu_SL Datasets/test/facebook.egr 8 --dump /tmp/ref_fb.bin >/dev/null 2>&1
scripts/consistency_metric Datasets/test/facebook.egr /tmp/ref_fb.bin /tmp/ref_fb.bin
```

Expected: a single JSON line. `consistency_ratio` is ≈ 0.993 on facebook (NOT 1.0) because JP-SL^A has tied priorities; verify instead that `1 - concordant_unord_pairs/T` matches `tie_ratio_ref` and that two `cpu_SL` runs produce byte-identical dumps (`cmp` two dumps). A near-zero ratio WOULD indicate a broken/inverted encoding — only then STOP and revisit Task 1.

- [ ] **Step 3: Cross-check vs an independent dumper**

Run:

```bash
tools/pa_dumper/pa_dumper -f Datasets/test/facebook.egr -a SDL --dump-priority /tmp/sdl_fb.bin >/dev/null 2>&1 \
  && scripts/consistency_metric Datasets/test/facebook.egr /tmp/ref_fb.bin /tmp/sdl_fb.bin \
  || echo "pa_dumper SDL unavailable — skip (non-blocking)"
```

Expected: a JSON line with a high `consistency_ratio` (JP-SL^A vs JP-SL^M `SDL` are both SL-family; ratio should be well above 0.8, not near 0). A near-0 ratio would indicate an inverted ordering convention. Non-blocking if `pa_dumper` is missing.

- [ ] **Step 4: Cleanup**

Run: `rm -f /tmp/ref_fb.bin /tmp/sdl_fb.bin /tmp/jpsla_fb.bin`

- [ ] **Step 5: Commit nothing** (verification only)

---

## Task 3 — Create `scripts/plots/priority_consistency/sweep_priority_consistency.py`

**Why:** Orchestrates the per-(framework,dataset) priority dumps + `consistency_metric` runs vs the JP-SL^A reference, and writes the JSON contract the plot consumes.

**Files:**
- Create: `scripts/plots/priority_consistency/sweep_priority_consistency.py`

- [ ] **Step 1: Create the directory**

Run: `mkdir -p scripts/plots/priority_consistency`

- [ ] **Step 2: Write the sweep driver**

Write the following to `scripts/plots/priority_consistency/sweep_priority_consistency.py`:

```python
#!/usr/bin/env python3
"""Sweep priority-ordering consistency vs the JP-SL^A reference.

For each EGR dataset: dump the JP-SL^A reference priority list with
cpu_SL, dump/synthesize each comparison framework's priority list, run
scripts/consistency_metric <egr> <ref.bin> <test.bin>, and collect the
consistency ratio. Writes a JSON consumed by
plot_priority_consistency.py.

Framework -> priority source:
  JP-SLL      pa_dumper -a JP_SLL
  JP-ADG      pa_dumper -a JP_ADG
  ECL-GC      synthesized largest-degree-first ordering (this script)
  CHROMA      pa_dumper -a cuSL_ELS        -e 0
  CHROMA+     pa_dumper -a cuSL_ELS_SDC    -e 0
  CHROMA*     CHROMA -a cuSL_ELS_SDC --predict --predict-model v0_paper
  CHROMA_v2   CHROMA -a cuSL_ELS_SDC --predict --predict-model v3

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
#   "chroma" : CHROMA -a cuSL_ELS_SDC --predict --predict-model <model>
#              [--no-dynamic-theta if spec["bump"] is False]
#   "cpuadg" : cpu_ADG <egr> <threads> --dump  (pa_dumper's JP_ADG
#              kernel emits no per-vertex order, so JP-ADG comes from
#              the CPU binary, mirroring JP-SL^A via cpu_SL)
#   "ldf"    : synthesized in Python (largest-degree-first)
# v3_raw = offline predictor only (online dynamic-theta bumping OFF);
# v3_bump = predictor + online bumping (CHROMA default in
# DYNAMIC_THETA=1 builds). v0_paper is the paper-era predictor, no bump.
FRAMEWORKS = {
    "JP-SLL":         ("pa",     {"algo": "JP_SLL"}),
    "JP-ADG":         ("cpuadg", {}),
    "ECL-GC":         ("ldf",    {}),
    "CHROMA":         ("pa",     {"algo": "cuSL_ELS"}),
    "CHROMA+":        ("pa",     {"algo": "cuSL_ELS_SDC"}),
    "CHROMA*":        ("chroma", {"model": "v0_paper", "bump": False}),
    "CHROMA_v2_raw":  ("chroma", {"model": "v3",       "bump": False}),
    "CHROMA_v2_bump": ("chroma", {"model": "v3",       "bump": True}),
}
DEFAULT_FRAMEWORKS = list(FRAMEWORKS.keys())


def read_egr_header_and_nindex(path: Path):
    """Return (nodes, edges, nindex list) from the .egr CSR header."""
    with open(path, "rb") as f:
        nodes, edges = struct.unpack("<ii", f.read(8))
        nindex = list(struct.unpack(f"<{nodes + 1}i", f.read(4 * (nodes + 1))))
    return nodes, edges, nindex


def write_ldf_priority(out_path: Path, nindex: list):
    """Synthesize ECL-GC's largest-degree-first ordering as a uint32
    priority dump: ascending priority value == processed earlier ==
    largest degree first (ties broken by ascending vertex id)."""
    nodes = len(nindex) - 1
    deg = [nindex[v + 1] - nindex[v] for v in range(nodes)]
    order = sorted(range(nodes), key=lambda v: (-deg[v], v))
    prio = [0] * nodes
    for rank, v in enumerate(order):
        prio[v] = rank
    with open(out_path, "wb") as f:
        f.write(struct.pack(f"<{nodes}I", *prio))


def run(cmd, timeout):
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


def make_test_dump(args, key, egr, nindex, out_bin):
    kind, spec = FRAMEWORKS[key]
    if kind == "ldf":
        write_ldf_priority(out_bin, nindex)
        return 0.0, None
    if kind == "pa":
        cmd = [args.pa_dumper, "-f", str(egr), "-a", spec["algo"],
               "-e", "0", "--dump-priority", str(out_bin)]
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
    ap.add_argument("--repo", default=str(repo))
    ap.add_argument("--binary",      default=str(repo / "CHROMA" / "CHROMA"))
    ap.add_argument("--pa-dumper",   default=str(repo / "tools" / "pa_dumper" / "pa_dumper"))
    ap.add_argument("--cpu-sl",      default=str(repo / "CPU" / "Parallel" / "cpu_SL"))
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

    for label, p in (("cpu_SL", args.cpu_sl),
                     ("consistency_metric", args.metric_bin)):
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
            nodes, edges, nindex = read_egr_header_and_nindex(egr)
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
            test_bin = tmp_root / f"{name}__{fw}.bin"
            dump_dt, dump_err = make_test_dump(args, fw, egr, nindex, test_bin)
            row = {"framework": fw, "dataset": name,
                   "nodes": nodes, "edges": edges,
                   "ref_wall_s": ref_dt, "dump_wall_s": dump_dt}
            if dump_err:
                row["error"] = f"dump failed: {dump_err}"
                print(f"# {name:28s} {fw:11s} DUMP FAIL ({dump_err})",
                      file=sys.stderr)
                rows.append(row)
                continue
            mr, m_dt, m_err = run(
                [args.metric_bin, str(egr), str(ref_bin), str(test_bin)],
                args.timeout)
            if m_err:
                row["error"] = f"metric failed: {m_err}"
                print(f"# {name:28s} {fw:11s} METRIC FAIL ({m_err})",
                      file=sys.stderr)
            else:
                metric = parse_metric(mr.stdout)
                if metric is None:
                    row["error"] = "metric: no JSON line"
                    print(f"# {name:28s} {fw:11s} METRIC PARSE FAIL",
                          file=sys.stderr)
                else:
                    for k in ("consistency_ratio", "tie_ratio_ref",
                              "tie_ratio_test", "distinct_ref",
                              "distinct_test", "concordant_unord_pairs"):
                        if k in metric:
                            row[k] = metric[k]
                    row["metric_wall_s"] = m_dt
                    print(f"# {name:28s} {fw:11s} "
                          f"cons={row.get('consistency_ratio', float('nan')):.4f} "
                          f"ref={ref_dt:6.1f}s", file=sys.stderr)
            rows.append(row)
            if not args.keep_dumps:
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
```

- [ ] **Step 3: Make it executable**

Run: `chmod +x scripts/plots/priority_consistency/sweep_priority_consistency.py`

- [ ] **Step 4: Commit**

```bash
git add scripts/plots/priority_consistency/sweep_priority_consistency.py
git commit -m "scripts/plots/priority_consistency: add consistency sweep driver"
```

---

## Task 4 — Smoke-test the sweep on 2 datasets

**Why:** Validate the JSON contract and that every framework produces a usable dump before the slow full sweep.

**Files:** none (verification only)

- [ ] **Step 1: Run the small sweep**

Run:

```bash
cd /home/chsieh45/PunchShadow/CHROMA
python3 scripts/plots/priority_consistency/sweep_priority_consistency.py \
    --only facebook le450_25d --threads 8 \
    --out /tmp/pc_smoke.json 2>&1 | tail -20
```

Expected stderr: a header line then 2 datasets × 8 frameworks = 16 `cons=...` lines, then `# wrote /tmp/pc_smoke.json`. ECL-GC and the pa/chroma rows should all show `cons=` in (0, 1].

- [ ] **Step 2: Validate JSON shape**

Run:

```bash
python3 - <<'EOF'
import json
d = json.load(open('/tmp/pc_smoke.json'))
assert d['baseline'] == 'JP-SL^A'
assert d['frameworks'] == ['JP-SLL','JP-ADG','ECL-GC','CHROMA','CHROMA+','CHROMA*','CHROMA_v2_raw','CHROMA_v2_bump']
names = {x['name'] for x in d['datasets']}
assert names == {'facebook', 'le450_25d'}, names
assert len(d['rows']) == 16, len(d['rows'])
for r in d['rows']:
    assert 'error' not in r, r
    cr = r['consistency_ratio']
    assert 0.0 < cr <= 1.0, (r['dataset'], r['framework'], cr)
print('JSON OK: 16 rows, all consistency_ratio in (0,1]')
EOF
```

Expected: `JSON OK: 16 rows, all consistency_ratio in (0,1]`. If any row has an `error`, STOP and report which framework/dataset and the error string.

- [ ] **Step 3: ECL-GC sanity (largest-degree-first)**

Run:

```bash
python3 - <<'EOF'
import struct
egr = '/home/chsieh45/PunchShadow/CHROMA/Datasets/test/facebook.egr'
import sys
sys.path.insert(0, '/home/chsieh45/PunchShadow/CHROMA/scripts/plots/priority_consistency')
from sweep_priority_consistency import read_egr_header_and_nindex, write_ldf_priority
nodes, edges, nindex = read_egr_header_and_nindex(egr)
write_ldf_priority('/tmp/ldf.bin', nindex)
prio = struct.unpack(f'<{nodes}I', open('/tmp/ldf.bin','rb').read())
deg = [nindex[v+1]-nindex[v] for v in range(nodes)]
vmax = max(range(nodes), key=lambda v: (deg[v], -v))
print('max-degree vertex', vmax, 'deg', deg[vmax], 'priority', prio[vmax],
      '-> should be 0 (processed first):', prio[vmax] == 0)
assert prio[vmax] == 0
EOF
```

Expected: `... -> should be 0 (processed first): True`.

- [ ] **Step 4: Cleanup**

Run: `rm -f /tmp/pc_smoke.json /tmp/ldf.bin`

- [ ] **Step 5: Commit nothing** (verification only)

---

## Task 5 — Create `scripts/plots/priority_consistency/plot_priority_consistency.py`

**Why:** Render the paper-Fig.5-style grouped bar chart from the sweep JSON.

**Files:**
- Create: `scripts/plots/priority_consistency/plot_priority_consistency.py`

- [ ] **Step 1: Write the plot script**

Write the following to `scripts/plots/priority_consistency/plot_priority_consistency.py`:

```python
#!/usr/bin/env python3
"""Render the priority-consistency figure (paper Fig. 5 redraw).

Single axes; one grouped cluster per dataset; one solid-colour bar per
framework; y-axis "Consistency" as percent with a configurable floor
(default 50%); 19 datasets on the x-axis sorted alphabetically by
displayed name (deterministic across servers); one horizontal legend on
top; no title.

Examples:
    python3 scripts/plots/priority_consistency/plot_priority_consistency.py
    python3 scripts/plots/priority_consistency/plot_priority_consistency.py \\
        --in scripts/plots/priority_consistency/consistency_results.json \\
        --ymin 0 --figsize 16 4
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
import matplotlib.ticker as mticker
import numpy as np

# Framework key -> (display label, fill colour). Order = bar order.
FRAMEWORK_STYLE = [
    ("JP-SLL",    "JP-SLL",     "#4C72B0"),
    ("JP-ADG",    "JP-ADG",     "#DD8452"),
    ("ECL-GC",    "ECL-GC",     "#55A467"),
    ("CHROMA",    "CHROMA",     "#C44E52"),
    ("CHROMA+",        r"CHROMA$^{+}$",            "#8172B3"),
    ("CHROMA*",        r"CHROMA$^{*}$",            "#937860"),
    ("CHROMA_v2_raw",  r"CHROMA$_{v2}$ (v3$_{raw}$)",  "#DA8BC3"),
    ("CHROMA_v2_bump", r"CHROMA$_{v2}$ (v3$_{bump}$)", "#CCB974"),
]

DATASET_LABELS = {
    "wiki-Vote.col":               "wiki-Vote",
    "Email-Enron.col":             "Email-Enron",
    "soc-Epinions1.col":           "soc-Epinions1",
    "wiki-Talk.col":               "wiki-Talk",
    "twitter_combined":            "twitter",
    "soc-pokec-relationships.col": "soc-Pokec",
}

TICK_FS, LABEL_FS, LEGEND_FS = 11, 13, 12


def main():
    repo = Path(__file__).resolve().parents[3]
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path",
                    default=str(here / "consistency_results.json"))
    ap.add_argument("--out-prefix",
                    default=str(here / "priority_consistency"))
    ap.add_argument("--ymin", type=float, default=50.0,
                    help="Y-axis floor in percent (default 50, paper style)")
    ap.add_argument("--figsize", nargs=2, type=float, default=[14.0, 4.0],
                    metavar=("W", "H"))
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run "
              f"sweep_priority_consistency.py first.", file=sys.stderr)
        sys.exit(1)

    d = json.loads(in_path.read_text())
    json_fw = set(d["frameworks"])
    fw_style = [(k, lbl, c) for (k, lbl, c) in FRAMEWORK_STYLE
                if k in json_fw]
    by_key = {(r["framework"], r["dataset"]): r for r in d["rows"]}

    datasets = sorted(
        d["datasets"],
        key=lambda x: DATASET_LABELS.get(x["name"], x["name"]).lower())
    names = [x["name"] for x in datasets]
    disp = [DATASET_LABELS.get(n, n) for n in names]
    n_ds = len(datasets)
    n_fw = len(fw_style)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    centers = np.arange(n_ds)
    bar_w = 0.85 / max(1, n_fw)
    offsets = (np.arange(n_fw) - (n_fw - 1) / 2) * bar_w

    for i, (key, _lbl, color) in enumerate(fw_style):
        # Percent; missing/errored cells render as a 0-height gap.
        ys = np.array([
            (by_key.get((key, nm), {}).get("consistency_ratio") or 0.0) * 100.0
            for nm in names])
        ax.bar(centers + offsets[i], ys, width=bar_w, color=color,
               edgecolor="black", linewidth=0.6)

    ax.set_xticks(centers)
    ax.set_xticklabels(disp, rotation=35, ha="right", fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylabel("Consistency", fontsize=LABEL_FS)
    ax.set_ylim(args.ymin, 100.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xlim(-0.5, n_ds - 0.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [mpatches.Patch(facecolor=c, edgecolor="black",
                              linewidth=0.6, label=lbl)
               for (_k, lbl, c) in fw_style]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.00), ncol=len(handles),
               frameon=False, fontsize=LEGEND_FS, columnspacing=1.8)

    fig.subplots_adjust(top=0.88, bottom=0.28, left=0.06, right=0.995)

    pdf = Path(args.out_prefix + ".pdf")
    png = Path(args.out_prefix + ".png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"# wrote {pdf}\n# wrote {png}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/plots/priority_consistency/plot_priority_consistency.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/plots/priority_consistency/plot_priority_consistency.py
git commit -m "scripts/plots/priority_consistency: add paper-Fig.5 plot script"
```

---

## Task 6 — Smoke-test the plot on synthetic data

**Why:** Confirm the plot renders a valid PDF+PNG without needing the full sweep.

**Files:** none (verification only)

- [ ] **Step 1: Synthetic JSON + render**

Run:

```bash
python3 - <<'EOF'
import json, pathlib
fws = ["JP-SLL","JP-ADG","ECL-GC","CHROMA","CHROMA+","CHROMA*","CHROMA_v2_raw","CHROMA_v2_bump"]
dss = [{"name":"facebook","nodes":4039,"edges":176468},
       {"name":"le450_25d","nodes":450,"edges":34850},
       {"name":"wiki-Talk.col","nodes":2394385,"edges":5021410}]
rows=[]
for k,ds in enumerate(dss):
    for j,fw in enumerate(fws):
        rows.append({"framework":fw,"dataset":ds["name"],
                     "nodes":ds["nodes"],"edges":ds["edges"],
                     "consistency_ratio":0.6+0.05*j-0.03*k})
pathlib.Path("/tmp/pc_plot_smoke.json").write_text(json.dumps(
    {"baseline":"JP-SL^A","frameworks":fws,"datasets":dss,"rows":rows}))
EOF
python3 scripts/plots/priority_consistency/plot_priority_consistency.py \
    --in /tmp/pc_plot_smoke.json --out-prefix /tmp/pc_plot_smoke 2>&1
ls -la /tmp/pc_plot_smoke.pdf /tmp/pc_plot_smoke.png
```

Expected: two `# wrote` lines and both files non-empty (> 5 KB).

- [ ] **Step 2: Cleanup**

Run: `rm -f /tmp/pc_plot_smoke.json /tmp/pc_plot_smoke.pdf /tmp/pc_plot_smoke.png`

- [ ] **Step 3: Commit nothing** (verification only)

---

## Task 7 — Create `scripts/plots/priority_consistency/README.md`

**Files:**
- Create: `scripts/plots/priority_consistency/README.md`

- [ ] **Step 1: Write the README**

Write the following to `scripts/plots/priority_consistency/README.md`:

```markdown
# Priority Consistency Ratio Plot

Redraws Fig. 5 of the CHROMA paper — vertex-priority-ordering
consistency of each framework with respect to the **JP-SL^A**
reference (Eq. 1–3: `C / T` concordant ordered pairs).

Reference (not drawn; it is the 100% anchor): **JP-SL^A** via
`CPU/Parallel/cpu_SL`. Comparison bars: `JP-SLL`, `JP-ADG`,
`ECL-GC` (largest-degree-first, synthesized), `CHROMA` (ELS),
`CHROMA+` (ELS+SDC), `CHROMA*` (ELS+SDC+EGC, `--predict-model
v0_paper`), `CHROMA_v2` (ELS+SDC+EGC, `--predict-model v3`).

## Prerequisites

- `cd CPU/Parallel && make`            (cpu_SL, with the `--dump` flag)
- `tools/pa_dumper/pa_dumper` built
- `CHROMA/CHROMA` built with `PRE_MODEL=1` (needed for `--predict`)
- `scripts/consistency_metric` built
  (`g++ -O3 -std=c++17 -Ilib/io scripts/consistency_metric.cpp -o scripts/consistency_metric`)

## Step 1 — Sweep

```
python3 scripts/plots/priority_consistency/sweep_priority_consistency.py
```

Key flags: `--only/--skip <stems>`, `--threads N` (cpu_SL OpenMP,
default 32), `--frameworks ...`, `--timeout SECS` (default 1800),
`--keep-dumps`, `--out PATH`. Writes
`scripts/plots/priority_consistency/consistency_results.json`
(gitignored under the project `*.json` rule; regenerable).

## Step 2 — Plot

```
python3 scripts/plots/priority_consistency/plot_priority_consistency.py
```

Flags: `--in`, `--out-prefix`, `--ymin` (percent floor, default 50 to
match the paper), `--figsize`. Writes `priority_consistency.{pdf,png}`.

## Notes

- `cpu_SL` is the JP-SL^A reference and JP-SL^A's priority allocation
  is the slow phase on large graphs (paper Fig. 1) — `europe_osm`,
  `as-skitter` references can take minutes. The per-cell `--timeout`
  absorbs this; a failed cell is recorded with an `error` field and
  drawn as a 0-height gap.
- `consistency_metric` dense-ranks each priority list independently,
  so only within-list ordering matters. JP-SL^A is a partial order
  (batch removal → tied priorities), so a ref-vs-ref check yields
  `1 − tie-fraction` (≈ 0.99), not exactly 1.0; the dump is still
  deterministic and correctly ordered (validated during development).
- ECL-GC has no SL-style priority list; its ordering is synthesized
  as largest-degree-first (ties broken by ascending vertex id),
  matching what ECL-GC actually uses for color allocation.
```

- [ ] **Step 2: Commit**

```bash
git add scripts/plots/priority_consistency/README.md
git commit -m "scripts/plots/priority_consistency: add usage README"
```

---

## Task 8 — Run the full 19×8 sweep

**Why:** Produce the real data for the figure.

**Files:** none (produces the gitignored JSON + a committed log)

- [ ] **Step 1: Run the sweep**

Run:

```bash
cd /home/chsieh45/PunchShadow/CHROMA
python3 scripts/plots/priority_consistency/sweep_priority_consistency.py \
    2>&1 | tee scripts/plots/priority_consistency/sweep.log
```

Expected: 19 × 8 = 152 result lines on stderr, then `# wrote .../consistency_results.json`. cpu_SL on `europe_osm` / `as-skitter` may take minutes each; total wall time can be 30+ min.

- [ ] **Step 2: Sanity check**

Run:

```bash
python3 -c "
import json
d=json.load(open('scripts/plots/priority_consistency/consistency_results.json'))
rows=d['rows']; errs=[r for r in rows if 'error' in r]
print(len(rows),'rows;',len(errs),'errors;',len(d['datasets']),'datasets')
for r in errs[:20]: print(' FAIL',r['framework'],r['dataset'],r['error'])
"
```

Expected: `152 rows; 0 errors; 19 datasets` (a few cpu_SL timeouts on the largest graphs are acceptable — note them; they render as gaps).

- [ ] **Step 3: Commit the log**

```bash
git add scripts/plots/priority_consistency/sweep.log
git commit -m "scripts/plots/priority_consistency: capture 19x8 consistency sweep log"
```

(The JSON is gitignored under the project's `*.json` rule, like the breakdown sweep results.)

---

## Task 9 — Render and commit the final figure

**Files:**
- Create (artifacts): `scripts/plots/priority_consistency/priority_consistency.{pdf,png}`

- [ ] **Step 1: Render**

Run: `python3 scripts/plots/priority_consistency/plot_priority_consistency.py 2>&1`

Expected: two `# wrote` lines.

- [ ] **Step 2: Visual check**

Open `scripts/plots/priority_consistency/priority_consistency.png`. Confirm:
- 19 dataset clusters, alphabetical x-order, rotated labels.
- 8 solid-colour bars per cluster in the order JP-SLL, JP-ADG,
  ECL-GC, CHROMA, CHROMA⁺, CHROMA*, CHROMA_v2.
- Y-axis "Consistency", 50%–100%, percent ticks.
- One horizontal legend on top, no title.
- Sanity vs paper Fig. 5: CHROMA⁺ / CHROMA* are high (≈ JP-SL^A);
  JP-ADG / ECL-GC are notably lower on skewed graphs (e.g. cit-Patents,
  delaunay_n24).

- [ ] **Step 3: Commit the artifacts**

```bash
git add scripts/plots/priority_consistency/priority_consistency.pdf \
        scripts/plots/priority_consistency/priority_consistency.png
git commit -m "scripts/plots/priority_consistency: render Fig.5 priority-consistency figure"
```

---

## Self-review checklist (run after writing, fix inline)

1. **Spec coverage:** cpu_SL `--dump` (T1, validated T2); sweep driver incl. ECL-GC LDF synth (T3, smoke T4); plot paper-Fig.5 style (T5, smoke T6); README (T7); full sweep (T8); figure (T9). All spec sections mapped.
2. **Placeholder scan:** every code/command step has concrete content; no TBD/TODO/"similar to".
3. **Type/name consistency:** framework keys `JP-SLL/JP-ADG/ECL-GC/CHROMA/CHROMA+/CHROMA*/CHROMA_v2` identical across sweep `FRAMEWORKS`, JSON, plot `FRAMEWORK_STYLE`, smoke asserts; JSON keys `consistency_ratio/datasets/rows/frameworks/baseline` consistent between writer (T3) and reader (T5); `read_egr_header_and_nindex` / `write_ldf_priority` names match between T3 definition and T4 import.
4. **Scope:** single figure + pipeline; one small C++ change + 3 new files. Single-plan sized.

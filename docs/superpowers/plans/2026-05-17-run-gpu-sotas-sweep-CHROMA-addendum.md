# CHROMA Integration (GPU Sweep Addendum) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the project's 6 single-GPU CHROMA configurations to the existing
registry-driven `scripts/run_sotas/sweep_gpu_sotas.py` (12 → 18 tools, 8 → 9
build units), keeping the deliverable contract (total exec ms excl. graph
load + color count) and graceful degradation.

**Architecture:** One file modified (`scripts/run_sotas/sweep_gpu_sotas.py`):
+1 time-regex constant, +6 declarative `REGISTRY` entries (one
`CHROMA/CHROMA` binary, `make PRE_MODEL=1` build unit `chroma`, distinct
argv, new `kind="chroma"`), +1 `build_unit_steps` branch, append `"chroma"`
to the build order, +1 `selftest_chroma` (3 checks) wired into
`run_selftest`, and 4 existing REGISTRY-size expected values updated. Plus
the `README.md` tool list. Selftest is the test mechanism (no pytest, parent
constraint): red/green via `--selftest`, target `61/61`.

**Tech Stack:** Python 3 stdlib only. Spec:
`docs/superpowers/specs/2026-05-17-run-gpu-sotas-sweep-CHROMA-addendum-design.md`.
All commands run from repo root `/home/chsieh45/PunchShadow/CHROMA`.

---

## File Structure

- **Modify** `scripts/run_sotas/sweep_gpu_sotas.py` — all logic changes.
- **Modify** `scripts/run_sotas/README.md` — tool list 12 → 18 + naming caveat.

No new files. Single-file engine per the parent design.

**Concurrency note (unchanged from parent):** a parallel user session also
commits to branch `feat/run-gpu-sotas-sweep`. NEVER `git add -A`/`.`/`-u`,
NEVER `git commit -a`/`-am`/`--amend`. Only `git add` the exact files and
commit with an explicit `-- <paths>` pathspec. Verify your commit with
`git show --stat --oneline <sha>`; HEAD may be a concurrent commit on top —
locate your own SHA via `git log --oneline -3`.

---

## Task 1: Add CHROMA to registry, parser, build, selftest, README

**Files:**
- Modify: `scripts/run_sotas/sweep_gpu_sotas.py`
- Modify: `scripts/run_sotas/README.md`

- [ ] **Step 1: Read the current file**

Run: `sed -n '108,116p;179,186p;318,326p;362,366p;617,632p;693,696p;707,710p' scripts/run_sotas/sweep_gpu_sotas.py`
Confirm the anchors below still match the current state (the file's last
change was the hermetic-selftest fix; `--selftest` currently prints
`SELFTEST: PASS (58/58 checks)`).

- [ ] **Step 2: Add the failing selftest first (RED)**

Add the `selftest_chroma` function. Insert it immediately AFTER the
`selftest_json` function and BEFORE `def selftest_arch(results: list)`
(use this exact anchor — the line `def selftest_arch(results: list) -> None:`
is unique):

Edit — old_string:
```python
def selftest_arch(results: list) -> None:
```
Edit — new_string:
```python
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
```

Then wire the call into `run_selftest`. Edit — old_string:
```python
    selftest_json(results)
    failed = [r for r in results if not r[0]]
```
Edit — new_string:
```python
    selftest_json(results)
    selftest_chroma(results)
    failed = [r for r in results if not r[0]]
```

- [ ] **Step 3: Run selftest to verify it fails (RED)**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest; echo "exit=$?"`
Expected: a traceback `KeyError: 'CHROMA'` (the registry has no `CHROMA`
entry yet) — confirms the new test is exercised and currently failing.

- [ ] **Step 4: Add the `_CHROMA_TOTAL_MS` constant**

Edit — old_string:
```python
_COLORS_USED = r"^\s*colors used:\s*(\d+)"

REGISTRY: list[dict] = [
```
Edit — new_string:
```python
_COLORS_USED = r"^\s*colors used:\s*(\d+)"
_CHROMA_TOTAL_MS = r"^\s*Total runtime:\s+([0-9]+(?:\.[0-9]+)?)\s*ms"

REGISTRY: list[dict] = [
```

- [ ] **Step 5: Append the 6 CHROMA registry entries**

Edit — old_string (the JP-SLL entry followed by the list close):
```python
    dict(name="JP-SLL", kind="jp", unit="jp-series",
         binrel="JP-Series/JP-Series",
         argv=["-f", "{G}", "-a", "JP-SLL"],
         colors=r"colors\s+used:\s*(\d+)",
         time=("ms", r"runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*ms"),
         algo="JP-SLL", usrc="ms"),
]
```
Edit — new_string:
```python
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
```

- [ ] **Step 6: Add the `chroma` build unit**

Edit — old_string (the jp-series branch then the final raise):
```python
    if unit == "jp-series":
        return [
            {"cmd": ["make", "-C", "JP-Series", "clean"], "cwd": ".",
             "ignore_fail": True, "retry": None},
            {"cmd": ["make", "-C", "JP-Series", f"ARCH={sm}"],
             "cwd": ".", "ignore_fail": False, "retry": None},
        ]
    raise ValueError(f"unknown build unit: {unit}")
```
Edit — new_string:
```python
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
```

- [ ] **Step 7: Append `chroma` to the build order**

Edit — old_string:
```python
    order = ["csrcolor", "csrcolor_data", "kokkos", "pgc", "picasso",
             "ecl-gc", "ecl-gc-r", "jp-series"]
```
Edit — new_string:
```python
    order = ["csrcolor", "csrcolor_data", "kokkos", "pgc", "picasso",
             "ecl-gc", "ecl-gc-r", "jp-series", "chroma"]
```

- [ ] **Step 8: Update the 4 REGISTRY-size-dependent expected values**

These are existing checks whose expected value changes because REGISTRY
grew 12 → 18 (they are NOT new checks; the total count is unchanged by
them). Apply all four exact edits:

Edit 1 — old: `    _check(results, "registry size", len(REGISTRY), 12)`
        new: `    _check(results, "registry size", len(REGISTRY), 18)`

Edit 2 — old:
```python
    _check(results, "registry names unique",
           len({t["name"] for t in REGISTRY}), 12)
```
new:
```python
    _check(results, "registry names unique",
           len({t["name"] for t in REGISTRY}), 18)
```

Edit 3 — old: `    _check(results, "no filter = all", len(sel3), 12)`
        new: `    _check(results, "no filter = all", len(sel3), 18)`

Edit 4 — old: `    _check(results, "exclude filter count", len(sel2), 10)`
        new: `    _check(results, "exclude filter count", len(sel2), 16)`
(`select_tools(exclude="data_wlc,data_pq")` now yields 18 − 2 = 16.)

- [ ] **Step 9: Run selftest to verify it passes (GREEN)**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest; echo "exit=$?"`
Expected EXACTLY: a line `SELFTEST: PASS (61/61 checks)` and `exit=0`.
(58 prior + 3 new CHROMA checks = 61; the 4 edited expected values do not
change the count.) If the count is not 61, do NOT change code to force it —
recount the `_check` calls, report the true number, and STOP
(`DONE_WITH_CONCERNS`) so the plan's annotation can be corrected (per the
parent plan's count-correction precedent).

- [ ] **Step 10: Verify the `chroma` build steps + argv via import**

Run:
```bash
python3 -c "import importlib.util as u; s=u.spec_from_file_location('m','scripts/run_sotas/sweep_gpu_sotas.py'); m=u.module_from_spec(s); s.loader.exec_module(m); \
print(m.build_unit_steps('chroma','89')); \
bn={t['name']:t for t in m.REGISTRY}; \
print(len(m.REGISTRY)); \
print(m.build_argv(bn['CHROMA*'],'/b/c','/g.egr')); \
print(m.needed_units(m.select_tools('CHROMA,CHROMA_v2','')) )"
```
Expected: `build_unit_steps('chroma','89')` = a 2-step list — step 0
`make -C CHROMA clean` (ignore_fail True), step 1
`make -C CHROMA PRE_MODEL=1 ARCH=sm_89` (ignore_fail False, retry None);
`len(m.REGISTRY)` = `18`; `build_argv` for `CHROMA*` =
`['/b/c', '-f', '/g.egr', '-a', '1', '-p', '--predict-model', 'v0_paper']`;
`needed_units(...)` = `{'chroma'}`.

- [ ] **Step 11: Update README**

Edit `scripts/run_sotas/README.md`.

Edit — old_string:
```markdown
## Tools (12)

SOTA (`External/`): `csrcolor`, `data_wlc`, `data_pq`, `kokkos_VB`,
`kokkos_VBBIT`, `pgc_parallel`, `Picasso`, `ECL-GC`, `ECL-GC-R`.
JP-Series: `cuSL`, `JP-ADG`, `JP-SLL`.
```
Edit — new_string:
```markdown
## Tools (18)

SOTA (`External/`): `csrcolor`, `data_wlc`, `data_pq`, `kokkos_VB`,
`kokkos_VBBIT`, `pgc_parallel`, `Picasso`, `ECL-GC`, `ECL-GC-R`.
JP-Series: `cuSL`, `JP-ADG`, `JP-SLL`.
CHROMA (`CHROMA/`, one `make PRE_MODEL=1` binary): `CHROMA` (-a 0),
`CHROMA+` (-a 1), `CHROMA*` (-a 1 -p --predict-model v0_paper),
`CHROMA_v2-b-adw` (-a 1 -p --predict-model v3 --no-dynamic-theta),
`CHROMA_v2-b` (-a 10 -p --predict-model v3 --no-dynamic-theta),
`CHROMA_v2` (-a 10 -p --predict-model v3).

Note: the `*` / `+` in CHROMA names are valid in JSON and in the
`--only`/`--exclude` CSV, but quote them in a shell to avoid globbing,
e.g. `--only 'CHROMA*'`.
```

- [ ] **Step 12: Commit (SCOPED — concurrency-safe pathspec)**

```bash
git add scripts/run_sotas/sweep_gpu_sotas.py scripts/run_sotas/README.md
git commit -m "scripts/run_sotas: integrate 6 CHROMA configs (12->18 tools)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" -- scripts/run_sotas/sweep_gpu_sotas.py scripts/run_sotas/README.md
```
Then verify: locate your commit (`git log --oneline -3`), run
`git show --stat --oneline <your-sha>` — it must contain ONLY those two
files. Report what `git diff --cached --name-status` showed pre-commit
(unrelated concurrent-session files staged is expected and excluded by the
pathspec). Branch must remain `feat/run-gpu-sotas-sweep`.

---

## Task 2: Acceptance — build `chroma` fresh + sweep the 6 CHROMA tools

**Files:** none (validation only; DIAGNOSE-ONLY — if a discrepancy is
found, report it precisely; do NOT edit code here — fixes route back through
the implement→review loop).

- [ ] **Step 1: Selftest regression (no GPU needed)**

Run: `python3 scripts/run_sotas/sweep_gpu_sotas.py --selftest; echo $?`
Expected: `SELFTEST: PASS (61/61 checks)`, exit 0.

- [ ] **Step 2: Build `chroma` fresh + sweep the 6 CHROMA tools on the smoke set**

Run (rebuilds only the `chroma` unit then runs the 6 CHROMA tools ×
facebook.egr + youtube.egr; CHROMA compile is a few minutes):
```bash
cd /home/chsieh45/PunchShadow/CHROMA
timeout 3000 python3 scripts/run_sotas/sweep_gpu_sotas.py \
  --only 'CHROMA,CHROMA+,CHROMA*,CHROMA_v2-b-adw,CHROMA_v2-b,CHROMA_v2' \
  --dataset-dir Datasets/test --out /tmp/gpu_sotas_chroma.json \
  > /tmp/chroma_sweep.log 2>&1; echo "EXIT=$?"
tail -30 /tmp/chroma_sweep.log
```
Expected: `[build] chroma: OK (Ns)`, then `[run] CHROMA :: facebook.egr`
… progress, a build+sweep summary, exit 0. If `chroma` build FAILs, capture
`builds[].error` from the JSON and report it (likely env: missing CUDA /
model files). Graceful degradation must still exit 0.

- [ ] **Step 3: Validate acceptance (spec §7)**

```bash
python3 -c "
import json
d=json.load(open('/tmp/gpu_sotas_chroma.json'))
print('builds:', [(b['unit'],b['ok'],(b['error'] or '')[-200:]) for b in d['builds']])
print('avail:', [(t['name'],t['available'],t['unavailable_reason']) for t in d['tools']])
bad=[]
for r in d['rows']:
    if not (r['ok'] and r['best_colors'] and r['best_colors']>0 and r['best_total_exec_ms'] and r['best_total_exec_ms']>0):
        bad.append((r['tool'],r['dataset'],r.get('error')))
print('FAILED CELLS:', bad if bad else 'NONE')
for r in d['rows']: print(r['tool'], r['dataset'], r['best_colors'], r['best_total_exec_ms'])
"
```
Acceptance: `chroma` build OK; all 6 CHROMA tools `available`; `FAILED
CELLS: NONE` (every CHROMA cell on both test graphs has `best_colors>0`
and `best_total_exec_ms>0`); script exited 0.

- [ ] **Step 4: Spot-check time source (Total runtime, not PA/CA)**

Run one predict config manually and compare to the recorded value:
```bash
cd /home/chsieh45/PunchShadow/CHROMA
CHROMA/CHROMA -f Datasets/test/facebook.egr -a 1 -p --predict-model v0_paper 2>&1 | grep -E 'EGC θ|PA runtime|CA runtime|Total runtime|colors used'
```
Confirm the recorded `best_total_exec_ms` for `CHROMA*`/facebook.egr in the
JSON corresponds to the `Total runtime:` ms (NOT the smaller `PA runtime:`
or `CA runtime:`), and `best_colors` == the `colors used:` value. Report the
raw lines vs the JSON values. (Diagnose-only: if it mismatches, report
exactly which line the parser picked vs expected; do not edit here.)

- [ ] **Step 5: Triage / report**

If `chroma` build failed: report `builds[].error` and whether it is an
environment issue (CUDA/model) vs a Makefile-invocation bug in the new
`chroma` build unit. If an available CHROMA tool produced a failed cell:
paste the tool's exact argv (from `tools[].binary` + registry) and the real
stdout of one manual run vs the recorded error. Do NOT modify any file.
`git log --oneline -3` + `git branch --show-current` (report, don't change).

---

## Self-Review (filled by plan author)

**Spec coverage (addendum spec §1–§7):**
- §2 6 configs / names / argv / kind=chroma / order-after-JP → Task 1 Step 5 ✔
- §3 colors reuse `_COLORS_USED`; new `_CHROMA_TOTAL_MS` time regex →
  Steps 4–5 (+ selftest Step 2 proves Total-not-PA/CA) ✔
- §4 `chroma` build unit (`make -C CHROMA clean` ; `make -C CHROMA
  PRE_MODEL=1 ARCH=sm_<NN>`) + appended to build order + graceful → Steps
  6–7 (graceful is the existing `compute_availability` path, unchanged) ✔
- §5 +3 selftest checks; 4 expected-value updates; `61/61` → Steps 2,8,9 ✔
- §6 README 12→18 + shell caveat; plan appended → Step 11 (this plan) ✔
- §7 acceptance (build chroma + 6-tool sweep + colors>0/time>0 + Total-time
  spot-check + selftest regression) → Task 2 ✔

**Placeholder scan:** no TBD/TODO; every edit gives exact old/new strings;
every command has an expected result. ✔

**Type consistency:** new entries use the established tool-dict keys
(`name/kind/unit/binrel/argv/colors/time/algo/usrc`); `time=("ms", regex)`
matches the existing `parse_time_ms` "ms" kind; `build_unit_steps` step dict
keys (`cmd/cwd/ignore_fail/retry`) match the existing 8 branches;
`selftest_chroma` uses the existing `_check`/`parse_colors`/`parse_time_ms`/
`build_argv` signatures. Count 58→61 stated and gated in Step 9. ✔

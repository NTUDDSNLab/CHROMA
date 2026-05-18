# Design Spec Addendum — Integrate CHROMA into the GPU Sweep

- **Date:** 2026-05-17 (addendum to `2026-05-17-run-gpu-sotas-sweep-design.md`)
- **Topic:** add 6 CHROMA configurations to `scripts/run_sotas/sweep_gpu_sotas.py`
- **Status:** Approved (brainstorming → spec)

## 1. Goal

Extend the existing registry-driven sweep with the project's own single-GPU
**CHROMA** framework, as **6 configurations** (one `CHROMA/CHROMA` binary,
distinct CLI flags). Same deliverable contract as the parent spec: per
`(tool, dataset)` record **total execution time (ms, excluding graph load)**
+ **color count**; no CA/PA columns; build-all-fresh with graceful
degradation; one aggregated JSON.

Registry grows **12 → 18 tools**; build units **8 → 9** (`+chroma`); a new
tool `kind` value `"chroma"`. Everything else in the parent spec is
unchanged.

## 2. The 6 CHROMA configurations

All run the single binary `CHROMA/CHROMA` (built once, see §4). `{G}` = the
absolute `.egr` path. Registry order: appended **after** the JP-Series block
(SOTA ×9 → JP ×3 → CHROMA ×6).

| name (verbatim, symbols kept per user) | argv (after binary) | `algo` metadata |
|---|---|---|
| `CHROMA` | `-f {G} -a 0` | `a=0` |
| `CHROMA+` | `-f {G} -a 1` | `a=1` |
| `CHROMA*` | `-f {G} -a 1 -p --predict-model v0_paper` | `a=1,predict=v0_paper` |
| `CHROMA_v2-b-adw` | `-f {G} -a 1 -p --predict-model v3 --no-dynamic-theta` | `a=1,predict=v3,no-dyn-theta` |
| `CHROMA_v2-b` | `-f {G} -a 10 -p --predict-model v3 --no-dynamic-theta` | `a=10,predict=v3,no-dyn-theta` |
| `CHROMA_v2` | `-f {G} -a 10 -p --predict-model v3` | `a=10,predict=v3` |

All 6 are distinct (verified against the user spec: `-b-adw` is `-a 1`,
`-b` is `-a 10`). `kind="chroma"`, `build_unit="chroma"`,
`binrel="CHROMA/CHROMA"`, `usrc="ms"` for all 6.

**Naming caveat (documented in README):** the `*` / `+` symbols are kept in
the tool names. They are valid JSON keys and the `--only`/`--exclude` CSV
split handles them, but in a shell the user must quote a glob-y name, e.g.
`--only 'CHROMA*'` (otherwise the shell expands it). No code change needed.

## 3. Parsers (verified against CHROMA source)

- **colors:** reuse the existing `_COLORS_USED` =
  `r"^\s*colors used:\s*(\d+)"`. CHROMA prints `colors used: %d`
  (`CHROMA/CHROMA.cu:716`, value = `colors_after`). Post-CA color reduction
  is **default-off** (commit `0ee9a6c` made `--no-reduce` the default; these
  configs do not pass `--reduce`), so `colors_after == colors_before` — the
  authoritative final count. `^\s*colors used:` does **not** match the
  `colors before reduction:` / `colors after reduction:` lines. Predict
  configs print the identical line.
- **time:** a **new** constant
  `_CHROMA_TOTAL_MS = r"^\s*Total runtime:\s+([0-9]+(?:\.[0-9]+)?)\s*ms"`,
  used as `time=("ms", _CHROMA_TOTAL_MS)`. CHROMA prints
  `Total runtime: %.6f ms` (`CHROMA/CHROMA.cu:710`,
  = `(runtime_PA + runtime_CA + reduction) * 1000`), with the timer started
  **after** graph load + H2D copy → **excludes graph loading** (matches the
  deliverable's metric). CHROMA also prints `PA runtime:`, `CA runtime:`,
  `Post reduction runtime:` — the existing `_MS`
  (`^\s*runtime:\s+...`) must **not** be reused (it would never match
  `Total runtime:` and is the wrong figure anyway). Predict configs print an
  extra `EGC θ: N (Predicted)` info line that does not affect either parsed
  line.

No new `parse_*` logic: the existing `parse_colors`/`parse_time_ms` +
`("ms", <regex>)` time-spec kind already cover this.

## 4. Build unit `chroma`

One unit serving all 6 configs (verified: a single PRE_MODEL=1 binary runs
the non-predict configs too; predict configs need `-DPRED_MODEL` which this
build provides; `--predict-model v3|v0_paper` is selected at **runtime**;
`--no-dynamic-theta` is a runtime flag and works with the default
`DYNAMIC_THETA=1` build):

```
make -C CHROMA clean                       # cwd=".", ignore_fail=True, retry=None
make -C CHROMA PRE_MODEL=1 ARCH=sm_<NN>    # cwd=".", ignore_fail=False, retry=None
```
Output binary: `CHROMA/CHROMA`. `<NN>` = the existing resolved numeric arch
(`resolve_arch`); CHROMA's Makefile default is `sm_86` so the explicit
`ARCH=sm_<NN>` override is required. Add `chroma` to `build_unit_steps` and
append `"chroma"` to `run_build_phase`'s stable `order` list (now 9 units).
A `chroma` build failure marks all 6 CHROMA tools `unavailable`
(reason `build failed (chroma)`) and the sweep continues — existing
`compute_availability` / graceful-degradation mechanism, no special-casing.

## 5. Selftest delta

- **Add 3 new checks** (a CHROMA golden case): given a realistic
  single-run CHROMA stdout that includes `PA runtime:`, `CA runtime:`,
  `colors before reduction:`, `colors after reduction:`, `Total runtime:`,
  `colors used:`, and an `EGC θ: N (Predicted)` line, assert
  (1) `parse_colors` → the `colors used:` integer,
  (2) `parse_time_ms` → the **`Total runtime`** ms (proving it does NOT pick
  `PA runtime:` / `CA runtime:` / a reduction line),
  (3) `build_argv` for one CHROMA config (e.g. `CHROMA_v2-b-adw`) → the
  exact expected argv list.
- **Update 4 existing REGISTRY-size-dependent expected values** (these are
  modified in place, not new checks — count unchanged by them):
  `selftest_parsers` "registry size" `12→18` and "registry names unique"
  `12→18`; `selftest_engine` "no filter = all" `12→18` and
  "exclude filter count" (excludes `data_wlc,data_pq`) `10→16`.
- **New selftest total: `SELFTEST: PASS (61/61 checks)`** (58 + 3). The
  implementer must re-verify this exact literal and the plan's count
  annotation must match (recompute if off, per the parent plan's
  count-correction precedent).

## 6. Spec / plan / README / memory updates

- This addendum is the spec record (committed; self-reviewed; user-reviewed
  before plan).
- `scripts/run_sotas/README.md`: tool list 12 → 18 (add the 6 CHROMA names);
  add the `--only 'CHROMA*'` shell-quote note.
- Implementation plan: append one new task ("Plan T10: integrate CHROMA")
  with the registry/build/parser/selftest steps and the corrected `61/61`
  expectation; keep the parent plan's prior tasks intact.
- The parent design spec stays as-is; this addendum is additive and
  authoritative for the CHROMA scope.

## 7. Testing / acceptance

No pytest (parent constraint) — `--selftest` is the mechanism. Acceptance
(targeted, after implementation):

1. Build the `chroma` unit fresh (`make -C CHROMA PRE_MODEL=1 ARCH=sm_<NN>`)
   and run `python3 scripts/run_sotas/sweep_gpu_sotas.py --only
   'CHROMA,CHROMA+,CHROMA*,CHROMA_v2-b-adw,CHROMA_v2-b,CHROMA_v2'
   --dataset-dir Datasets/test`.
2. Confirm: `chroma` build OK; all 6 CHROMA tools `available`; every
   CHROMA cell on `facebook.egr` + `youtube.egr` has `best_colors > 0` and
   `best_total_exec_ms > 0`; script exits 0.
3. Spot-check (not hard gates): the predict configs (`CHROMA*`,
   `CHROMA_v2*`) still produce a valid color/time (the `EGC θ: N (Predicted)`
   line doesn't break parsing); `best_total_exec_ms` corresponds to
   `Total runtime:` (not the smaller `PA`/`CA runtime:` values) by
   eyeballing one raw run vs the recorded value.
4. Regression: full `--selftest` → `SELFTEST: PASS (61/61 checks)`, exit 0;
   the other 12 tools + kokkos behavior unchanged.

Implementation proceeds via the established subagent-driven loop
(implementer → spec-compliance review → code-quality review) and scoped
commits (concurrent-session-safe pathspec discipline) — same as the parent.

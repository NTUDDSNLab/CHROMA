# BB-cuSL: Bucket-Based Speculative Locking with Elastic Coloring

- **Date**: 2026-05-06
- **Status**: Approved design, ready for implementation
- **Module**: `CHROMA/` (single-GPU CUDA backend)
- **Variant ID**: `5` / `cuSL_ELS_BB`

---

## 1. Motivation

### 1.1 Observed bottleneck

Profiling the existing single-GPU pipeline shows
**`PA runtime : CA runtime` = 5x to 20x** across representative graphs. The
`P_SL_ELS` priority-assignment kernel (`CHROMA/PA.cu:91-167`) is therefore the
dominant cost.

The structural cause is that every outer iteration of `P_SL_ELS` performs a
full **O(N) scan** to find peelable vertices:

```c
for (int v = tid; v < nodes; v += threads) {
  if (prio == 0) {
    if (degree <= theta + FuzzyNumber) {
      // ── peel: ~O(|peel set|) of the iteration's work
    } else {
      // ── pure bookkeeping: iteration_v += FuzzyNumber + 1, no real work
    }
  }
}
```

The `else` branch only updates the unpeeled vertex's `iteration_v` accumulator
and contributes nothing to peeling progress. Outer-iteration count is
≈ `Δ / (FuzzyNumber + 1)`, so the total wasted scan is

```
O(N · Δ / (FuzzyNumber + 1))
```

For `N = 10⁷`, `Δ = 10³`, `FuzzyNumber = 10` this is ~10⁹ unnecessary
`degree_list` reads per coloring run.

### 1.2 Goal

Replace the O(N) scan with **bucket-driven peel**:
each outer iteration touches only the vertices currently in the peel range
`[θ, θ + FuzzyNumber]`. Target speedup factor ≈ `Δ / (FuzzyNumber + 1)²`,
which translates to **5x–10x PA latency reduction** on typical large social
and synthetic graphs in `Datasets/EGR/`.

End-to-end (given PA:CA = 5x–20x), this maps to **~4x–7x total runtime
improvement** — the primary deliverable.

---

## 2. Goals & Non-goals

### Goals

1. New algorithm variant `cuSL_ELS_BB` (numeric id `5`) selectable via `-a`,
   coexisting with the existing 5 `cuSL_ELS*` variants.
2. **PA latency reduction ≥ 5x** on at least one of `youtube.egr`,
   `as-skitter.egr`, `wiki-Talk.col.egr` from `Datasets/EGR/`.
3. **Color count parity** (±1) with `cuSL_ELS` on the same dataset / same
   `FuzzyNumber`.
4. Compatible with the existing `--predict` (auto-θ) path; no flag-surface
   changes for users.
5. Implementable inside a 1-hour time budget by following the existing
   `P_SL_ELS` cooperative-kernel pattern.

### Non-goals (explicit, deferred to future work)

- PA + CA-init fused variant (`cuSL_ELS_BB_FUSED`).
- SDC variant (`cuSL_ELS_BB_SDC`) — argued unnecessary in §3.5.
- Per-bucket capacity tuning (dynamic prefix-sum to reduce memory).
- Multi-GPU integration (`CHROMA_RGP`).
- Low-Δ fallback (`if Δ < threshold use P_SL_ELS`).
- Templating `bb_window` as kernel template parameter.

---

## 3. Design overview

### 3.1 Why bucket-based

A bucket structure `B[d] = {v : degree(v) = d}` lets the peel kernel iterate
**only over candidates** rather than scanning all `N` vertices. The naive CPU
implementation is textbook degeneracy ordering.

### 3.2 Why sliding window (not full buckets)

The naive layout allocates `B[0..Δ_max]`, but `Δ_max` reaches 1M+ for
power-law graphs and the cumulative push count is `O(2E)` (every degree
decrement triggers a bucket move). For `r4-2e23.sym.egr` (E ≈ 7×10⁷) this is
4.4 GB of bucket churn.

A **sliding window** materialises only `B[θ..θ + window − 1]`. A vertex with
degree above the window is left unbucketed and joins later when its degree
falls into the window via a neighbour-driven decrement. Window size
`window = FuzzyNumber + 1` exactly covers one peel iteration's range.

Memory becomes `O(N · window)` — independent of `Δ_max`.

### 3.3 Why push-on-decrement

Each Phase-2 decrement that lands inside the window pushes the affected
vertex into `B[new_d]`. Vertices outside the window are not pushed; they
enter on their first in-window decrement. Total pushes per vertex is bounded
by `window`, so total push count ≤ `N · window`.

### 3.4 Why lazy validation (Δ-stepping insight)

Atomic remove-from-bucket on GPU is impractical, so a vertex may have **stale
duplicate entries** across multiple buckets after decrements. Phase 1 reads
each entry and validates `degree[v] == bucket_idx`; mismatched entries are
skipped. This trick is borrowed directly from GPU Δ-stepping SSSP and is
particularly safe here because **degrees only decrease** (no priority
inversions).

### 3.5 Why no SDC variant

The existing `_SDC` variants compute `g_minDegree` more accurately via
`warpReduceMin` of post-decrement values. In BB-cuSL, finding new θ is
already an `O(window)` scan over `bucket_count[]` — this is the **ground
truth minimum within the window**, not a sample. SDC is therefore subsumed
for free; no separate `cuSL_ELS_BB_SDC` variant is needed.

### 3.6 Plan A: unbounded θ jumps via warpReduceMin (2026-05-06)

Profiling revealed that BB-cuSL performed **160 outer iterations** on
`youtube.egr -e 10` while `cuSL_ELS_SDC` performed only **66**. Per-iteration
cost was similar (BB ~144 μs vs SDC ~180 μs), so the 2.4× iter-count gap
explained the full 1.94× total slowdown.

Root cause: Phase 3's window-scan finds the minimum only within
`[curr_theta, curr_theta + window)`. When the window is empty, the old code
set `bb_overflow_needed = 1` (triggering the O(N) fallback scan). Phase 2's
constant refills kept the lowest in-window slot non-empty most of the time,
so Phase 3 advanced θ by 1–2 slots per iteration rather than jumping ahead.
SDC, by contrast, tracks the true post-decrement global minimum via
`warpReduceMin` + `atomicMin(&g_minDegree, ...)`, allowing θ to jump by
hundreds of slots at once.

**Fix:** BB now mirrors SDC's mechanism. Phase 2 computes a per-warp minimum
of `new_d` using `warpReduceMin` and publishes it to `g_minDegree` via
`atomicMin`. Phase 3 reads this `captured_min` after the window scan. When
the window is empty and `captured_min < INT_MAX`, Phase 3 jumps θ directly
to `captured_min` and signals `bb_overflow_needed = 2` (refill-only). Phase 4
gains a refill-only mode (mode 2) that skips the O(N) scan since θ is already
correct, directly zeroing the bucket counts and refilling from unpeeled
vertices. Mode 1 (full fallback) is retained for the edge case where no
decrements fired during Phase 2 (`captured_min == INT_MAX`).

Both the cooperative kernel (`BB.cu`) and the split-kernel diagnostic variant
(`BB_split.cu`) are updated to maintain semantic equivalence.

---

## 4. Architecture

### 4.1 High-level flow

```
init_degree<<<>>>            ← unchanged, fills degree_list
       │
       ▼
P_SL_ELS_BB (cooperative)    ← NEW kernel, single launch, internal phases:
       │
       │  Phase 0: compute θ_init + initial bucket fill (once)
       │  Phase 1: peel — read window buckets, lazy validate, CAS-claim
       │  Phase 2: decrement neighbours, push to new bucket if in window
       │  Phase 3: advance θ, reset emptied slot counts
       │  Phase 4: overflow scan (rare fallback)
       │
       ▼
ECL_GC_run (init + runLarge + runSmall)    ← unchanged
       │
       ▼
run_post_color_reduction
```

The outer pipeline (`init_degree → PA → CA → reduction`) is identical to
`P_SL_ELS`. Only the PA kernel symbol differs.

### 4.2 Integration with existing CHROMA

| File | Change |
|---|---|
| `globals.cuh` | Add 5 device-global declarations, 1 kernel prototype |
| `globals.cu` | Add 5 device-global definitions |
| `BB.cu` (new) | Define `P_SL_ELS_BB` cooperative kernel (~200–250 LOC) |
| `chroma_utils.cu` | Allocate bucket arrays in `allocAndInit`; `cudaMemcpyToSymbol` for pointers and `bb_window` |
| `CHROMA.cu` | Add case `5` / `"cuSL_ELS_BB"` in `select_algorithm`; update `print_help` |
| `Makefile` | Add `BB.cu` to `SRCS` |
| `scripts/batch_test.py` | Add regex + dataclass fields for `PA runtime`, `CA runtime`, and `EGC θ` so per-dataset records contain PA/CA times alongside total runtime and color counts |

The `cudaLaunchCooperativeKernel` calling convention does **not** change;
extra device pointers live in `__device__` globals (same pattern as
`remove_list`).

---

## 5. Data structures

### 5.1 New device globals

```cpp
// globals.cuh
extern __device__ int   bb_window;            // = FuzzyNumber + 1
extern __device__ int*  bb_bucket_data;       // size = N * bb_window
extern __device__ int*  bb_bucket_count;      // size = bb_window
extern __device__ int   bb_bucket_capacity;   // = N (per-bucket)
extern __device__ int   bb_overflow_needed;   // 0 or 1
extern __device__ int   bb_init_done;         // Phase 0 latch

__global__ void P_SL_ELS_BB(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);
```

The kernel signature **matches `P_SL_ELS` exactly** so the existing
`cudaLaunchCooperativeKernel` call site needs no signature change.

**Host-side initialisation** (in `allocAndInit`, before kernel launch):

```cpp
int bw_host = std::min(fuzzy_number + 1, 31);   // clamp per §10 R6
cudaMemcpyToSymbol(bb_window, &bw_host, sizeof(int));
cudaMalloc(&bucket_data_host, sizeof(int) * g.nodes * bw_host);
cudaMemcpyToSymbol(bb_bucket_data, &bucket_data_host, sizeof(int*));
cudaMalloc(&bucket_count_host, sizeof(int) * bw_host);
cudaMemset(bucket_count_host, 0, sizeof(int) * bw_host);
cudaMemcpyToSymbol(bb_bucket_count, &bucket_count_host, sizeof(int*));
int cap = g.nodes; cudaMemcpyToSymbol(bb_bucket_capacity, &cap, sizeof(int));
int zero = 0;     cudaMemcpyToSymbol(bb_init_done, &zero, sizeof(int));
                  cudaMemcpyToSymbol(bb_overflow_needed, &zero, sizeof(int));
```

`allocAndInit`'s signature gains an `int fuzzy_number` parameter so the
host-side `fuzzy_number` (already known after `setParameters` in `CHROMA.cu`)
flows in.

### 5.2 `iteration_list[v]` bit layout (unchanged from `P_SL_ELS`)

| Bits | Field | Purpose |
|---|---|---|
| 31 | `large_deg` | set if `nidx[v+1] − nidx[v] ≥ WS=32` (consumed by `init()`) |
| 30 | `peeled` | set when v is peeled (consumed by Phase 1 stale-check and `init()`) |
| 29-0 | `priority_v` | computed lazily at peel time (see §5.4) |

**No `in_bucket` flag**. Stale duplicate entries are tolerated by Phase 1's
lazy-validate; deduplication would require `atomicCAS` per push and would not
eliminate re-pushes (degree continues to fall, requiring fresh entries).

### 5.3 Bucket array layout

Single contiguous data array, one shared physical-slot mod-`window`
indexing:

```
logical bucket B[d]   →   physical_slot = d % bb_window
                          start_index   = physical_slot * N
                          count         = bucket_count[physical_slot]
```

Per-bucket capacity is `N` (worst-case upper bound; typically used <<10%).
Total `bucket_data` size is `4 · N · bb_window` bytes.

Wraparound is safe because:
- Window size is exactly `bb_window`; logical bucket `d` and `d + bb_window`
  cannot both be active at the same iteration.
- When θ advances past a slot, Phase 3 resets that slot's count to 0 before
  it is reused for a higher logical d.

### 5.4 Priority encoding (lazy at peel time)

```cpp
unsigned int priority_v = peel_iter * (FuzzyNumber + 1)
                        + (d_at_peel - theta + 1);
iteration_list[v] = (large_deg << 31) | (1u << 30) | priority_v;
```

Where `peel_iter` is a 0-indexed counter incremented in Phase 3.

**Equivalence with `P_SL_ELS`**: a vertex peeled at outer iteration `k`
(0-indexed) with degree `d` and current θ in the existing kernel ends with

```
iteration_v = k · (FuzzyNumber + 1) + (d − θ + 1)
```

— identical to the BB-cuSL formula. The downstream `init()` /
`runLarge()` / `runSmall()` kernels therefore observe identical priority
relationships and produce the same color count.

This equivalence is the most critical correctness invariant; §10 R5
schedules a direct dump-and-diff of `iteration_list` on `facebook.egr` as the
first validation step.

---

## 6. Algorithm

### 6.1 Phase 0 — Init θ and initial fill (once per run)

```c
// Inside P_SL_ELS_BB, gated by bb_init_done latch
if (bb_init_done == 0) {
  // 6.1.a Reduce: find theta_init = min(degree_list[v])
  unsigned int local_min = UINT_MAX;
  for (v = tid; v < N; v += threads) {
    iteration_list[v] = 0;
    local_min = min(local_min, degree_list[v]);
  }
  warp/block reduce → atomicMin(&theta, local_min);
  grid.sync();

  // 6.1.b Fill in-window buckets
  for (v = tid; v < N; v += threads) {
    unsigned int d = degree_list[v];
    if (d < (unsigned int)(theta + bb_window)) {
      int slot = d % bb_window;
      int idx  = atomicAdd(&bb_bucket_count[slot], 1);
      bb_bucket_data[slot * N + idx] = v;
    }
  }
  grid.sync();
  if (grid.thread_rank() == 0) bb_init_done = 1;
  grid.sync();
}
```

Cost: O(N) reads + O(|in-window vertices|) atomic pushes. Single grid pass.

### 6.2 Phase 1 — Peel (per outer iteration)

```c
int curr_theta = theta;
int window     = bb_window;

for (int slot_off = 0; slot_off < window; slot_off++) {
  int d        = curr_theta + slot_off;
  int physical = d % window;
  int cnt      = bb_bucket_count[physical];

  for (int i = tid; i < cnt; i += threads) {
    int v = bb_bucket_data[physical * N + i];
    unsigned int it_v = iteration_list[v];

    if (it_v & 0x40000000u) continue;                      // already peeled
    if (degree_list[v] != (unsigned int)d) continue;       // stale degree

    unsigned int prio_v = peel_iter_local * window
                        + (d - curr_theta + 1);
    bool large = (nidx[v+1] - nidx[v]) >= WS;
    unsigned int newval = ((large ? 1u : 0u) << 31)
                        | (1u << 30) | prio_v;

    if (atomicCAS(&iteration_list[v], it_v, newval) == it_v) {
      remove_list[atomicAdd(&remove_size, 1)] = v;
    }
  }
}
grid.sync();
```

The `atomicCAS` ensures a single peel-claim per vertex even if multiple stale
duplicates pass lazy validation in the same iteration.

After the outer `for slot_off` loop completes and the grid syncs, a single
thread resets `bb_bucket_count[physical] = 0` for all `window` slots. This is
required because Phase 1 does not decrement the count when it skips stale
entries or peels valid ones — without an explicit reset, the count carries
stale-positive values into Phase 3 and theta never advances. Phase 2 then
atomicAdds into freshly-zeroed counts, so `bb_bucket_count` correctly reflects
only this iteration's pushes. A second `grid.sync()` follows the reset so
Phase 2 sees count = 0 before any atomicAdd.

### 6.3 Phase 2 — Decrement + push

```c
int rs = remove_size;
for (int k = warpId; k < rs; k += numWarp) {
  int v   = remove_list[k];
  int beg = nidx[v];          // each lane reads itself — coalesced,
  int end = nidx[v + 1];      // no shfl_sync broadcast needed

  unsigned int warpMin = UINT_MAX;

  for (int i = beg + lane; i < end; i += WS) {
    int u = nlist[i];
    unsigned int it_u = __ldg(&iteration_list[u]);
    if (it_u & 0x40000000u) continue;        // u already peeled

    unsigned int new_d = atomicSub(&degree_list[u], 1) - 1;
    warpMin = min(warpMin, new_d);           // Plan A: track post-decrement min

    if (new_d >= (unsigned int)curr_theta && new_d < (unsigned int)(curr_theta + window)) {
      int physical = new_d % window;
      int idx = atomicAdd(&bb_bucket_count[physical], 1);
      if (idx < bb_bucket_capacity) {
        bb_bucket_data[physical * N + idx] = u;
      } else {
        atomicSub(&bb_bucket_count[physical], 1);   // overflow safety
      }
    }
  }

  // Plan A: warp-reduce + publish to g_minDegree (mirrors P_SL_ELS_SDC)
  warpMin = warpReduceMin(warpMin);
  if (lane == 0 && warpMin < UINT_MAX) atomicMin(&g_minDegree, (int)warpMin);
}
grid.sync();
```

Note: `nidx[v]` / `nidx[v+1]` are read independently by each lane (they all
target the same address — the load coalesces and broadcasts in cache). The
shfl-from-lane-0 broadcast pattern used in `P_SL_ELS` is unnecessary here.

The push condition guards both ends of the window. The lower-bound
`new_d >= curr_theta` is required: a vertex `u` with multiple peeled
neighbours can have its degree drop below `curr_theta` during Phase 2. Without
the guard, `new_d % window` maps to the same physical slot as a **future**
logical bucket (`new_d + window`), corrupting its count with a stale entry
that lazy-validation permanently rejects. That phantom count prevents Phase 4
overflow recovery from ever seeing the slot as empty, stalling `worker` at
`< N` and causing an infinite loop on high-degree-hub graphs. Vertices whose
degree falls below `curr_theta` are correctly recovered by Phase 4 once the
window legitimately empties.

### 6.4 Phase 3 — Advance θ + reset emptied slots

```c
if (grid.thread_rank() == 0) {
  worker      += rs;
  remove_size  = 0;

  int new_theta = curr_theta;
  while (new_theta < curr_theta + window) {
    if (bb_bucket_count[new_theta % window] > 0) break;
    new_theta++;
  }

  int captured_min = g_minDegree;
  atomicExch(&g_minDegree, INT_MAX);        // reset for next iter

  if (new_theta >= curr_theta + window) {
    // Window empty — unbounded jump via captured_min (Plan A)
    if (captured_min < INT_MAX) {
      theta              = captured_min;
      bb_overflow_needed = 2;              // refill-only Phase 4
    } else {
      bb_overflow_needed = 1;             // full Phase 4 fallback
    }
  } else {
    // Window has entries — normal advance (window scan is authoritative)
    for (int d = curr_theta; d < new_theta; d++) {
      bb_bucket_count[d % window] = 0;
    }
    theta              = new_theta;
    bb_overflow_needed = 0;
  }
  peel_iter_local++;
}
grid.sync();
```

Because Phase 1 resets `bb_bucket_count` to 0 at the end of each iteration
(before Phase 2 runs), the count entering Phase 3 reflects only Phase 2's
fresh pushes. `bb_bucket_count[s] > 0` therefore means there are genuine
live entries in slot `s` — not stale leftovers. Phase 3's scan is exact.

`captured_min` is the warp-reduced post-decrement minimum accumulated by
Phase 2's `warpReduceMin + atomicMin` calls. When the window scan finds
entries (`new_theta < curr_theta + window`), `captured_min` is irrelevant —
the window-scan answer is authoritative (it is the exact in-window minimum).
When the window is empty, `captured_min` gives the true global minimum of
remaining degree values, allowing Phase 3 to skip directly to it without an
O(N) scan.

### 6.5 Phase 4 — Overflow / refill (fallback)

`bb_overflow_needed` is now a tri-state latch:
- `0` = no Phase 4 needed
- `1` = full Phase 4: O(N) scan + set theta + refill
- `2` = refill-only: theta already set by Phase 3's Plan A jump; skip O(N) scan

```c
if (bb_overflow_needed != 0) {
  int mode = bb_overflow_needed;

  if (mode == 1) {
    // Full fallback: scan unpeeled to find min degree, then set theta
    unsigned int local_min = UINT_MAX;
    for (v = tid; v < N; v += threads) {
      if (!(iteration_list[v] & 0x40000000u)) {
        local_min = min(local_min, degree_list[v]);
      }
    }
    // warp/block reduce → atomicMin(&g_minDegree, local_min)
    grid.sync();

    if (grid.thread_rank() == 0) {
      for (int s = 0; s < window; s++) bb_bucket_count[s] = 0;
      theta = g_minDegree;
      atomicExch(&g_minDegree, INT_MAX);
    }
    grid.sync();
  } else {
    // mode == 2: refill-only (Plan A) — theta already correct, just reset buckets
    if (grid.thread_rank() == 0) {
      for (int s = 0; s < window; s++) bb_bucket_count[s] = 0;
    }
    grid.sync();
  }

  // Common: refill buckets for [new_theta, new_theta + window)
  int new_theta = theta;
  for (v = tid; v < N; v += threads) {
    if (iteration_list[v] & 0x40000000u) continue;
    unsigned int d = degree_list[v];
    if (d < (unsigned int)(new_theta + window)) {
      int physical = d % window;
      int idx = atomicAdd(&bb_bucket_count[physical], 1);
      bb_bucket_data[physical * N + idx] = v;
    }
  }
  grid.sync();

  if (grid.thread_rank() == 0) bb_overflow_needed = 0;
  grid.sync();
}
```

Mode 1 is structurally a re-execution of Phase 0 over the unpeeled subset.
Triggered when `captured_min == INT_MAX` (no decrements in Phase 2) — rare.

Mode 2 (Plan A fast path) fires when the window empties and Phase 2 did
observe at least one decrement. This eliminates the O(N) scan on every
window-empty iteration; only the O(N) refill remains (unavoidable).

The outer `do { … } while (worker != N)` loop wraps Phases 1–4.

---

## 7. Memory budget

Extra memory per run = `4 · N · bb_window + ~32 bytes metadata`.

With `bb_window = 11` (FuzzyNumber = 10):

| Dataset (`Datasets/EGR/`) | Size on disk | Est. N | BB extra mem |
|---|---:|---:|---:|
| school1 | 154 KB | < 1 K | < 50 KB |
| le450_25d | 141 KB | < 1 K | < 50 KB |
| facebook | 722 KB | ~ 4 K | ~ 176 KB |
| wiki-Vote | 835 KB | ~ 7 K | ~ 310 KB |
| Email-Enron | 1.6 MB | ~ 36 K | ~ 1.6 MB |
| soc-Epinions1 | 3.5 MB | ~ 75 K | ~ 3.3 MB |
| Slashdot 0811 / 0902 | ~ 4 MB | ~ 80 K | ~ 3.5 MB |
| twitter_combined | 11 MB | ~ 81 K | ~ 3.6 MB |
| Stanford | 17 MB | ~ 280 K | ~ 12 MB |
| youtube | 28 MB | ~ 1.1 M | ~ 48 MB |
| wiki-Talk | 47 MB | ~ 2.4 M | ~ 106 MB |
| as-skitter | 96 MB | ~ 1.7 M | ~ 75 MB |
| cit-Patents | 147 MB | ~ 3.7 M | ~ 163 MB |
| soc-pokec | 185 MB | ~ 1.6 M | ~ 70 MB |
| delaunay_n24 | 470 MB | ~ 16 M | ~ 700 MB |
| rmat22.sym | 542 MB | ~ 4 M | ~ 176 MB |
| r4-2e23.sym | 570 MB | ~ 8 M | ~ 350 MB |
| europe_osm | 636 MB | ~ 50 M | ~ 2.2 GB |

All datasets fit on A100 24 GB; europe_osm is tight, comfortable on H100 80 GB.

---

## 8. Performance estimates

| Phase | `P_SL_ELS` cost / iter | BB-cuSL cost / iter |
|---|---|---|
| Phase 1 | **N reads + N writes** | `O(|peel set|)` reads + lazy-validate |
| Phase 2 | `|peel set| · avg_deg` decrements | same + ≤ `window · |peel set|` pushes |
| Phase 3 | N reduction (g_minDegree) | `O(window)` count scan |
| Outer iters | ~ `Δ / (FuzzyNumber + 1)` | same |
| Total Phase-1 work | **`N · Δ / (FuzzyNumber + 1)` reads** | `O(N)` reads + `O(N)` CAS |
| Total push count | 0 | ≤ `N · bb_window` atomicAdds |

### 8.1 Per-dataset PA speedup (theoretical, atomicAdd ~5x slower than read)

| Dataset | Est. Δ | Est. PA speedup |
|---|---:|---:|
| facebook | ~ 1k | **~ 1.8x** |
| youtube | ~ 28k | **~ 50x** |
| as-skitter | ~ 35k | **~ 62x** |
| wiki-Talk | ~ 100k | **~ 180x** |
| soc-pokec | ~ 14k | **~ 25x** |
| rmat22 / r4-2e23 | ~ 100k+ | **~ 180x+** |
| europe_osm | ~ 13 | **~ 0.02x** (BB-cuSL slower than `P_SL_ELS`) |

Low-Δ road / mesh networks lose to BB-cuSL because the per-push overhead
exceeds the saved scan cost. This is documented and accepted; future work
includes a Δ-threshold fallback (out of scope for this prototype).

### 8.2 End-to-end estimates (given `PA:CA = 5x ~ 20x`)

Assuming a 10x PA speedup:

| `PA:CA` | Old PA fraction | New PA fraction | Total speedup |
|---|---:|---:|---:|
| 5 : 1 | 83 % | 28 % | **~ 4.0x** |
| 10 : 1 | 91 % | 48 % | **~ 5.6x** |
| 20 : 1 | 95 % | 64 % | **~ 7.0x** |

These are estimates only; the prototype validation (§9) will replace them
with measured numbers.

---

## 9. Validation plan

All builds use `PRE_MODEL=1` so that `--predict` is exercised:

```bash
cd CHROMA && make ARCH=sm_86 PRE_MODEL=1
```

### 9.0 Required metrics (every run, every stage)

Every CHROMA invocation in this validation must record **all three** of the
following from stdout:

| Metric | Source line in CHROMA stdout |
|---|---|
| **PA runtime** (ms) | `PA runtime: %.6f ms` |
| **CA runtime** (ms) | `CA runtime: %.6f ms` |
| **Colors used** | `colors used: %d` (also `colors before/after reduction`) |

`scripts/batch_test.py` is extended in this prototype to parse `PA runtime`,
`CA runtime`, and `EGC θ` (the `--predict`-chosen value) in addition to the
existing `runtime` / `colors_used` / `colors_before/after_reduction`.

### 9.1 Stage 1 — `facebook.egr` correctness gate

```bash
./CHROMA -f ../Datasets/EGR/facebook.egr -a cuSL_ELS    --predict > ref.log
./CHROMA -f ../Datasets/EGR/facebook.egr -a cuSL_ELS_BB --predict > bb.log
```

Tabulate both runs side-by-side as:

| run | EGC θ | PA runtime (ms) | CA runtime (ms) | colors used |
|---|---:|---:|---:|---:|
| `cuSL_ELS` (ref) | … | … | … | … |
| `cuSL_ELS_BB` | … | … | … | … |

**Pass criteria** (all four required):

- [ ] `bb.log` contains `result verification passed`
- [ ] `bb.log`'s `colors used:` is within ±1 of `ref.log`'s
- [ ] `bb.log`'s `PA runtime:` ≤ 1.2× `ref.log`'s (small graph; absolute
      speedup not required)
- [ ] `bb.log`'s `CA runtime:` within ±10 % of `ref.log`'s (CA must not
      regress — BB-cuSL only changes PA)
- [ ] No `bucket overflow` warning printed

**Encoding equivalence subcheck** (DEBUG build only): dump
`iteration_list[0..99]` after PA from both runs, byte-equal compare. This
directly verifies the §5.4 invariant.

### 9.2 Stage 2 — Sweep across `Datasets/EGR/`

```bash
python3 scripts/batch_test.py \
  --dataset-dir Datasets/EGR \
  --binary CHROMA/CHROMA \
  --algorithm cuSL_ELS_BB \
  --predict --runs 5 \
  --out bb_sweep.json

python3 scripts/batch_test.py \
  --dataset-dir Datasets/EGR \
  --binary CHROMA/CHROMA \
  --algorithm cuSL_ELS \
  --predict --runs 5 \
  --out cusl_sweep.json
```

Each `*_sweep.json` record must contain at minimum:
`{ dataset, fuzzy_number, pa_runtime_ms, ca_runtime_ms, total_runtime_ms,
colors_used, colors_before_reduction, colors_after_reduction }`.

A small diff script reads both JSONs and reports **per dataset**:

| Field | Definition |
|---|---|
| `colors_diff` | `bb.colors_used − cusl.colors_used` |
| `PA_speedup` | `cusl.pa_runtime_ms / bb.pa_runtime_ms` |
| `CA_delta_pct` | `(bb.ca_runtime_ms − cusl.ca_runtime_ms) / cusl.ca_runtime_ms` |
| `total_speedup` | `cusl.total_runtime_ms / bb.total_runtime_ms` |

Expected:
- `|colors_diff|` ≤ 1 on all datasets
- `PA_speedup` ≥ 5x on at least one of `youtube`, `as-skitter`, `wiki-Talk`
- `|CA_delta_pct|` ≤ 10 % on all datasets (CA must not regress)
- `PA_speedup` < 1 acceptable on `europe_osm`, `delaunay_n24`,
  `le450_25d`, `school1`

### 9.3 Stage 3 — Edge cases

- `europe_osm.egr` — confirm BB-cuSL produces correct color count even when
  slower than reference; no crash; no overflow-scan loop.
- `school1.egr`, `le450_25d.egr` — small DIMACS, confirm BB-cuSL doesn't
  break on tiny `N`.
- `rmat22.sym.egr`, `r4-2e23.sym.egr` — high-Δ stress, confirm large
  speedup materialises.

---

## 10. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|:---:|:---:|---|
| R1 | `atomicCAS` contention in Phase 1 (multiple lanes claim same v) | Med | Med | Profile retry rate; if > 30% add in-warp dedup |
| R2 | Phase 1 lazy-validate skip rate too high (sparse buckets) | Med | Med | Profile bucket density; future: periodic compaction |
| R3 | Per-bucket capacity (= N) overflows | Low | High | Phase 2 has overflow check + atomicSub rollback; warning printed |
| R4 | Phase 4 overflow triggered every iter (degenerates to scan) | Low | High | facebook expected to never trigger; profile per dataset |
| R5 | `iteration_list` encoding diverges from `P_SL_ELS` → wrong colors | **Med** | **High** | **Stage 1 dump-and-diff is the gate** |
| R6 | `bb_window > 30` from `--predict` → mod cost / memory | Low | Low | Assert + clamp `fuzzy_number ≤ 30` in `allocAndInit` |
| R7 | Cooperative kernel register pressure reduces `blkPerSM` | Med | Med | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` already adjusts; observe |
| R8 | BB-cuSL slower than `P_SL_ELS` on `europe_osm` | **High** | Low (known) | Documented; no fallback in prototype |
| R9 | Implementation overruns 1-hour budget | Med | Med | Three-phase cut: BB.cu writeup → facebook gate → sweep |
| R10 | Grid-sync deadlock from conditional early-return inside a phase | Low | Critical | All conditional returns sit outside `grid.sync()` boundaries |
| R11 | `--predict` returns very large `fuzzy_number` → out-of-memory | Low | High | R6 clamp covers this |
| R12 | `--predict` baseline differs from prior `-e 10` results, breaking comparison with old logs | Med | Low | New comparison runs are self-contained; old logs not used as baseline |

R5 is the highest-priority correctness risk; R8 is the highest-likelihood
performance risk but explicitly accepted.

---

## 11. Time budget

| Step | Estimate |
|---|---:|
| `BB.cu` + `globals.{cuh,cu}` + `chroma_utils.cu` + `CHROMA.cu` edits + Makefile | 30 min |
| `scripts/batch_test.py` extension (PA / CA / EGC θ regex + fields) | 5 min |
| Stage 1 facebook validation + encoding diff | 10 min |
| Stage 2 sweep + diff script | 10 min |
| Stage 3 edge cases + report | 5 min |
| **Total** | **~ 1 hr** |

---

## 12. Future work

- `cuSL_ELS_BB_FUSED` — fuse PA + CA-init like the existing `_FUSED` variant
  to skip the `init_degree`-then-`init` round trip.
- Per-bucket capacity from dynamic prefix-sum, cutting memory 5–10x.
- Δ-threshold fallback to `P_SL_ELS` for low-Δ graphs (`europe_osm`,
  `delaunay_n24`).
- Multi-GPU port: BB-cuSL kernel as the per-partition PA inside `CHROMA_RGP`.
- Periodic in-warp compaction to bound stale duplicate accumulation on
  pathological graphs.
- Make `bb_window` a kernel template parameter for register-allocator
  optimisation.

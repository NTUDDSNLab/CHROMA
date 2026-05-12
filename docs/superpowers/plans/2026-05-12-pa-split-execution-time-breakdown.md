# PA-Split Execution-Time Breakdown Plot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-phase `PA scan` / `PA decrement` timing for three SPLIT-mode frameworks (`cuSL_ELS_SDC_SPLIT`, new `cuSL_ELS_SDC_CTA_SPLIT`, new `cuSL_ELS_SDC_CTA_S_SPLIT`), sweep them over 19 EGR datasets, and render a stacked grouped-bar figure showing the CA / PA-scan / PA-decrement breakdown.

**Architecture:**
- Reuse the existing `P_SL_ELS_SDC_split_scan` for Phase 1 across all three frameworks (Phase 1 is byte-identical between `P_SL_ELS_SDC`, `_CTA`, and `_CTA_S` in `PA.cu`).
- Add two new Phase-2 split kernels (`_CTA_split_decrement`, `_CTA_S_split_decrement`) and three host launchers in `chroma_utils.cu` that wrap every scan/decrement launch with a `cudaEvent` pair and return a `PaSplitStats { scan_ms, decrement_ms }`.
- Wire two new algo ids (11, 12) into `CHROMA.cu` and gate per-phase printing on `*_SPLIT` algo names.
- A Python sweep (`scripts/batch_profile.py`) then drives 3 frameworks × 19 EGR datasets × 5 runs and writes a JSON contract that the plot script (`scripts/plots/plot_execution_breakdown.py`) renders into PDF + PNG.

**Tech Stack:** CUDA 12+, C++17, Python 3 (matplotlib + standard library).

**Spec:** `docs/superpowers/specs/2026-05-12-pa-split-execution-time-breakdown-design.md`

---

## File map

| File                                     | Action  | Responsibility                                                              |
|------------------------------------------|---------|-----------------------------------------------------------------------------|
| `CHROMA/PA_split.cu`                     | modify  | Extend `_split_advance`; add two new `_split_decrement` kernels             |
| `CHROMA/chroma_utils.cuh`                | modify  | `PaSplitStats` struct; declarations for three launchers                     |
| `CHROMA/chroma_utils.cu`                 | modify  | Refactor `run_sdc_split`; add `run_sdc_cta_split` and `run_sdc_cta_s_split` |
| `CHROMA/CHROMA.cu`                       | modify  | Help banner, `select_algorithm` cases 11/12, dispatch, per-phase printing   |
| `scripts/batch_profile.py`               | create  | 3-framework × 19-dataset × 5-run sweep → JSON                              |
| `scripts/plots/plot_execution_breakdown.py` | create | Render the stacked grouped-bar figure                                       |
| `scripts/plots/`                         | mkdir   | Directory for plot scripts and emitted figures                              |

---

## Task 1 — Extend `_split_advance` to reset `cursor_remove`

**Why:** The two new CTA split-decrement kernels consume `cursor_remove` via `atomicAdd` during Phase 2 (`P_SL_ELS_SDC_CTA` and `P_SL_ELS_SDC_CTA_S` reset it to 0 at the end of every outer iteration). The basic SDC SPLIT doesn't use the cursor, so adding the reset is a no-op there.

**Files:**
- Modify: `CHROMA/PA_split.cu:78-88`

- [ ] **Step 1: Edit `P_SL_ELS_SDC_split_advance` to also zero `cursor_remove`**

Replace the existing body with:

```cuda
__global__ void P_SL_ELS_SDC_split_advance()
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        worker        += remove_size;
        remove_size    = 0;
        theta          = g_minDegree;
        atomicExch(&g_minDegree, 0x7FFFFFFF);
        iteration      = iteration + 1 + FuzzyNumber;
        cursor_remove  = 0;
        iter_count++;
    }
}
```

- [ ] **Step 2: Verify `cursor_remove` is declared in `globals.cuh`**

```bash
grep -n "cursor_remove" CHROMA/globals.cuh
```

Expected: a line like `__device__ int cursor_remove;` (or `extern __device__` plus a definition in `globals.cu`). If it's not declared, the file already references it from `PA.cu` so the symbol must exist — confirm in `globals.cu`.

- [ ] **Step 3: Sanity build (no commit yet)**

Run: `cd CHROMA && make ARCH=sm_89 -j4 2>&1 | tail -20 && cd ..`

Expected: clean build (last line `Built target` or no error). If `globals.cuh` requires forward-declaration, add `extern __device__ int cursor_remove;` near the top of `PA_split.cu` (after the existing `#include "globals.cuh"`).

- [ ] **Step 4: Commit**

```bash
git add CHROMA/PA_split.cu
git commit -m "$(cat <<'EOF'
CHROMA/PA_split.cu: reset cursor_remove in _split_advance

Prepares the advance kernel to support new CTA / CTA_S split-decrement
kernels that consume cursor_remove during Phase 2. No-op for the existing
SDC SPLIT (it doesn't use the cursor).
EOF
)"
```

---

## Task 2 — Add `P_SL_ELS_SDC_CTA_split_decrement` kernel

**Why:** Mirrors the Phase 2 of `P_SL_ELS_SDC_CTA` (block-scan + CTA-balanced removal + `atomicSub`) as a standalone kernel so it can be timed externally with `cudaEvent`. Removes the `do { ... grid.sync(); } while(...)` outer loop (the host loop replaces it).

**Files:**
- Modify: `CHROMA/PA_split.cu` (append after the existing `_split_advance`)

- [ ] **Step 1: Append the new kernel at the bottom of `PA_split.cu`**

```cuda
// ──────────────────────────────────────────────────────────────────────────
// P_SL_ELS_SDC_CTA_split_decrement — Phase 2 of P_SL_ELS_SDC_CTA, isolated
// for per-phase profiling. Block-scan + CTA-balanced work distribution +
// atomicSub on neighbour degrees.
// ──────────────────────────────────────────────────────────────────────────
__global__ void P_SL_ELS_SDC_CTA_split_decrement(
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list)
{
    const int lane = threadIdx.x & 31;

    __shared__ node_buf_t node_buf[BLOCK_SIZE];
    __shared__ int buf_size;
    __shared__ int start_idx;
    __shared__ int task_size;
    __shared__ int total_degree;

    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    while (true) {
        unsigned int warpMin = 0x7FFFFFFF;

        if (threadIdx.x == 0) {
            start_idx = atomicAdd(&cursor_remove, BLOCK_SIZE);
            if (start_idx >= remove_size) {
                task_size = 0;
            } else {
                int remaining = remove_size - start_idx;
                task_size = (remaining < BLOCK_SIZE) ? remaining : BLOCK_SIZE;
            }
        }
        __syncthreads();
        if (task_size <= 0) break;

        node_buf_t val;
        {
            int idx = start_idx + threadIdx.x;
            val.node_id    = (idx < remove_size && idx >= start_idx)
                           ? remove_list[idx] : -1;
            val.degree     = (val.node_id == -1) ? 0
                           : nidx[val.node_id + 1] - nidx[val.node_id];
            val.prefix_sum = 0;
        }
        __syncthreads();

        int deg[1]  = { val.degree };
        int pref[1] = { 0 };
        BlockScan(temp_storage).ExclusiveSum(deg, pref);
        __syncthreads();

        val.prefix_sum = pref[0];
        node_buf[threadIdx.x] = val;
        __syncthreads();

        if (threadIdx.x == 0) {
            buf_size     = task_size;
            total_degree = node_buf[task_size - 1].prefix_sum
                         + node_buf[task_size - 1].degree;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_degree; i += BLOCK_SIZE) {
            int low = 0, high = buf_size - 1;
            while (low <= high) {
                int mid   = (low + high) >> 1;
                int start = node_buf[mid].prefix_sum;
                int end   = start + node_buf[mid].degree;
                if      (i >= end)   low  = mid + 1;
                else if (i <  start) high = mid - 1;
                else {
                    int source   = node_buf[mid].node_id;
                    int neighbor = __ldg(nlist + nidx[source] + (i - start));
                    unsigned int it = __ldg(iteration_list + neighbor);
                    if (!(it & 0x40000000u)) {
                        warpMin = min(warpMin,
                                      (unsigned int)(atomicSub(&degree_list[neighbor], 1) - 1));
                    }
                    break;
                }
            }
        }
        warpMin = warpReduceMin(warpMin);
        if (lane == 0 && warpMin < 0x7FFFFFFF) atomicMin(&g_minDegree, (int)warpMin);
        __syncthreads();
    }
}
```

- [ ] **Step 2: Add the `node_buf_t` struct forward-decl if needed**

Look at the top of `PA_split.cu`. `node_buf_t` is declared in `PA.cu` (around line 96) — `PA_split.cu` does not include `PA.cu`. If the build fails with "incomplete type 'node_buf_t'", add this near the top of `PA_split.cu` (right after `#include <cstdio>`):

```cuda
struct node_buf_t {
    int node_id;
    int degree;
    int prefix_sum;
};
```

Also confirm `cub::BlockScan` is available — `PA.cu` uses `<cub/cub.cuh>`; if `PA_split.cu` doesn't already pull it in, add at the top:

```cuda
#include <cub/cub.cuh>
```

And confirm `warpReduceMin` is visible. Search for its definition:

```bash
grep -n "warpReduceMin" CHROMA/*.cu CHROMA/*.cuh CHROMA/globals.cuh
```

If it's only in `PA.cu`, you need to either move it to `globals.cuh` (preferred for reuse) or copy it into `PA_split.cu`. Move-to-globals is cleaner; the inline body is:

```cuda
__device__ __forceinline__ unsigned int warpReduceMin(unsigned int v) {
    for (int o = 16; o > 0; o >>= 1) {
        unsigned int t = __shfl_xor_sync(0xffffffff, v, o);
        if (t < v) v = t;
    }
    return v;
}
```

If `globals.cuh` is the right home, move it there and delete the `static` (if any) copy from `PA.cu`.

- [ ] **Step 3: Build**

Run: `cd CHROMA && make ARCH=sm_89 -j4 2>&1 | tail -20 && cd ..`

Expected: clean build. If link errors mention undefined symbols, double-check the includes/forward-decls in step 2.

- [ ] **Step 4: Commit**

```bash
git add CHROMA/PA_split.cu CHROMA/globals.cuh
git commit -m "$(cat <<'EOF'
CHROMA/PA_split.cu: add P_SL_ELS_SDC_CTA_split_decrement kernel

Mirrors Phase 2 of P_SL_ELS_SDC_CTA (cub::BlockScan + CTA-balanced
removal + atomicSub) as a standalone kernel so the decrement phase can
be timed in isolation via cudaEvent. The do/grid.sync outer loop from
the cooperative kernel is dropped — the host loop in chroma_utils.cu
will drive iteration.
EOF
)"
```

---

## Task 3 — Add `P_SL_ELS_SDC_CTA_S_split_decrement` kernel

**Why:** Mirrors Phase 2 of `P_SL_ELS_SDC_CTA_S` — the dispatched SDC-warp / CTA path that switches behaviour based on `remove_size < CTA_S_THRESHOLD`. This is the framework the user wants to highlight ("dynamic workload balancing").

**Files:**
- Modify: `CHROMA/PA_split.cu` (append after the previous task's kernel)

- [ ] **Step 1: Append the new kernel at the bottom of `PA_split.cu`**

```cuda
// ──────────────────────────────────────────────────────────────────────────
// P_SL_ELS_SDC_CTA_S_split_decrement — Phase 2 of P_SL_ELS_SDC_CTA_S,
// isolated for per-phase profiling. Per-block dispatch:
//   remove_size <  CTA_S_THRESHOLD : SDC warp-per-vertex path
//   remove_size >= CTA_S_THRESHOLD : CTA-balanced removal
// ──────────────────────────────────────────────────────────────────────────
#ifndef CTA_S_THRESHOLD
#define CTA_S_THRESHOLD (BLOCK_SIZE * 4)
#endif

__global__ void P_SL_ELS_SDC_CTA_S_split_decrement(
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threads = gridDim.x * blockDim.x;
    const int lane    = threadIdx.x & 31;
    const int warpId  = tid >> 5;
    const int numWarp = threads >> 5;

    __shared__ node_buf_t node_buf[BLOCK_SIZE];
    __shared__ int buf_size;
    __shared__ int start_idx;
    __shared__ int task_size;
    __shared__ int total_degree;
    __shared__ int use_sdc_path;

    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    if (threadIdx.x == 0) {
        use_sdc_path = (remove_size < CTA_S_THRESHOLD) ? 1 : 0;
    }
    __syncthreads();

    if (use_sdc_path) {
        // SDC warp-per-vertex path (lifted from P_SL_ELS_SDC's inner loop)
        for (int k = warpId; k < remove_size; k += numWarp) {
            int v = remove_list[k];
            int beg, end;
            if (lane == 0) {
                beg = nidx[v];
                end = nidx[v + 1];
            }
            beg = __shfl_sync(0xffffffff, beg, 0);
            end = __shfl_sync(0xffffffff, end, 0);

            unsigned int warpMin = 0x7FFFFFFF;
            for (int i = beg + lane; i < end; i += 32) {
                int nei = __ldg(nlist + i);
                unsigned int it = __ldg(iteration_list + nei);
                if (!(it & 0x40000000u)) {
                    warpMin = min(warpMin,
                                  (unsigned int)(atomicSub(&degree_list[nei], 1) - 1));
                }
            }
            warpMin = warpReduceMin(warpMin);
            if (lane == 0 && warpMin < 0x7FFFFFFF) atomicMin(&g_minDegree, (int)warpMin);
        }
    } else {
        // CTA-balanced removal (same as P_SL_ELS_SDC_CTA_split_decrement above)
        while (true) {
            unsigned int warpMin = 0x7FFFFFFF;

            if (threadIdx.x == 0) {
                start_idx = atomicAdd(&cursor_remove, BLOCK_SIZE);
                if (start_idx >= remove_size) {
                    task_size = 0;
                } else {
                    int remaining = remove_size - start_idx;
                    task_size = (remaining < BLOCK_SIZE) ? remaining : BLOCK_SIZE;
                }
            }
            __syncthreads();
            if (task_size <= 0) break;

            node_buf_t val;
            {
                int idx = start_idx + threadIdx.x;
                val.node_id    = (idx < remove_size && idx >= start_idx)
                               ? remove_list[idx] : -1;
                val.degree     = (val.node_id == -1) ? 0
                               : nidx[val.node_id + 1] - nidx[val.node_id];
                val.prefix_sum = 0;
            }
            __syncthreads();

            int deg[1]  = { val.degree };
            int pref[1] = { 0 };
            BlockScan(temp_storage).ExclusiveSum(deg, pref);
            __syncthreads();

            val.prefix_sum = pref[0];
            node_buf[threadIdx.x] = val;
            __syncthreads();

            if (threadIdx.x == 0) {
                buf_size     = task_size;
                total_degree = node_buf[task_size - 1].prefix_sum
                             + node_buf[task_size - 1].degree;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < total_degree; i += BLOCK_SIZE) {
                int low = 0, high = buf_size - 1;
                while (low <= high) {
                    int mid   = (low + high) >> 1;
                    int start = node_buf[mid].prefix_sum;
                    int end   = start + node_buf[mid].degree;
                    if      (i >= end)   low  = mid + 1;
                    else if (i <  start) high = mid - 1;
                    else {
                        int source   = node_buf[mid].node_id;
                        int neighbor = __ldg(nlist + nidx[source] + (i - start));
                        unsigned int it = __ldg(iteration_list + neighbor);
                        if (!(it & 0x40000000u)) {
                            warpMin = min(warpMin,
                                          (unsigned int)(atomicSub(&degree_list[neighbor], 1) - 1));
                        }
                        break;
                    }
                }
            }
            warpMin = warpReduceMin(warpMin);
            if (lane == 0 && warpMin < 0x7FFFFFFF) atomicMin(&g_minDegree, (int)warpMin);
            __syncthreads();
        }
    }
}
```

- [ ] **Step 2: Build**

Run: `cd CHROMA && make ARCH=sm_89 -j4 2>&1 | tail -20 && cd ..`

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add CHROMA/PA_split.cu
git commit -m "$(cat <<'EOF'
CHROMA/PA_split.cu: add P_SL_ELS_SDC_CTA_S_split_decrement kernel

Mirrors Phase 2 of P_SL_ELS_SDC_CTA_S with the same per-block dispatch
(SDC warp-per-vertex when remove_size < CTA_S_THRESHOLD, CTA-balanced
otherwise). Allows the decrement phase of the dispatched workload-
balancing path to be timed in isolation.
EOF
)"
```

---

## Task 4 — Add `PaSplitStats` struct + refactor `run_sdc_split` to return per-phase ms

**Why:** The existing `run_sdc_split` returns `void`. The host launcher is the only place that can measure per-phase wall time (the kernels themselves don't accept a stats output). Wrap every scan/decrement launch with a `cudaEvent` pair and return the accumulated milliseconds.

**Files:**
- Modify: `CHROMA/chroma_utils.cuh:60-61` (declarations)
- Modify: `CHROMA/chroma_utils.cu:283-306` (definition)

- [ ] **Step 1: Add `PaSplitStats` struct and update the `run_sdc_split` declaration in `chroma_utils.cuh`**

Edit `CHROMA/chroma_utils.cuh`. Above the existing line `void run_sdc_split(int blocks, const ECLgraph& g, DevPtr& d);`, add:

```cpp
/* Per-phase timing returned by SDC split-mode launchers (milliseconds,
 * summed across all outer iterations).  scan_ms covers every
 * P_SL_ELS_SDC_split_scan kernel launch; decrement_ms covers every
 * Phase-2 split-decrement kernel launch.  Excludes the advance kernel
 * and the cudaMemcpyFromSymbol(worker) round-trip. */
struct PaSplitStats {
    float scan_ms{};
    float decrement_ms{};
};
```

Then replace the existing line

```cpp
void run_sdc_split(int blocks, const ECLgraph& g, DevPtr& d);
```

with

```cpp
PaSplitStats run_sdc_split(int blocks, const ECLgraph& g, DevPtr& d);
```

- [ ] **Step 2: Rewrite the `run_sdc_split` body in `chroma_utils.cu`**

Replace lines 283-306 (the existing function block) with:

```cpp
/* --------------- run_sdc_split ------------------- */
PaSplitStats run_sdc_split(int blocks, const ECLgraph& g, DevPtr& d)
{
    int N = g.nodes;
    int worker_h = 0;
    PaSplitStats stats{};

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    auto record = [&](float& acc) {
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, beg, end);
        acc += ms;
    };

    while (worker_h != N) {
        // Phase 1: scan all N vertices, peel those with degree <= theta+FuzzyNumber
        cudaEventRecord(beg, 0);
        P_SL_ELS_SDC_split_scan<<<blocks, ThreadsPerBlock>>>(
            N, d.nidx_d, d.degree_list, d.iteration_list_d);
        record(stats.scan_ms);

        // Phase 2: decrement neighbours of peeled vertices, track new min degree
        cudaEventRecord(beg, 0);
        P_SL_ELS_SDC_split_decrement<<<blocks, ThreadsPerBlock>>>(
            d.nidx_d, d.nlist_d, d.degree_list, d.iteration_list_d);
        record(stats.decrement_ms);

        // Phase 3: advance worker, reset remove_size, update theta, reset cursor_remove
        P_SL_ELS_SDC_split_advance<<<1, 32>>>();
        cudaDeviceSynchronize();

        cudaMemcpyFromSymbol(&worker_h, worker, sizeof(int));
    }

    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return stats;
}
```

- [ ] **Step 3: Build (will fail at the existing caller — that's expected)**

Run: `cd CHROMA && make ARCH=sm_89 -j4 2>&1 | tail -20 && cd ..`

Expected: build may succeed if no caller uses the old return type, or fail in `CHROMA.cu` with "value computed is not used" / "void returned from function with non-void return type". Task 8 wires the caller. If it succeeds, that's also fine (the void→struct return change is binary-compatible in the trivial case).

- [ ] **Step 4: Commit**

```bash
git add CHROMA/chroma_utils.cuh CHROMA/chroma_utils.cu
git commit -m "$(cat <<'EOF'
CHROMA: PaSplitStats + run_sdc_split returns per-phase ms

Wraps every split_scan / split_decrement launch with a single reusable
cudaEvent pair and accumulates milliseconds into scan_ms / decrement_ms.
Caller in CHROMA.cu still ignores the return value; the per-phase
printout will be wired in Task 8.
EOF
)"
```

---

## Task 5 — Add `run_sdc_cta_split` host launcher

**Why:** Driver for the CTA-balanced split-decrement kernel. Same host-loop shape as `run_sdc_split`, swapping Phase 2.

**Files:**
- Modify: `CHROMA/chroma_utils.cuh` (declaration)
- Modify: `CHROMA/chroma_utils.cu` (definition right after `run_sdc_split`)

- [ ] **Step 1: Add the declaration in `chroma_utils.cuh`**

After the existing `PaSplitStats run_sdc_split(...)` declaration, append:

```cpp
/* CTA-balanced SDC split-phase host driver. Same shape as run_sdc_split
 * but uses P_SL_ELS_SDC_CTA_split_decrement for Phase 2. */
PaSplitStats run_sdc_cta_split(int blocks, const ECLgraph& g, DevPtr& d);
```

- [ ] **Step 2: Add the definition in `chroma_utils.cu`**

Append right after the new `run_sdc_split` body (i.e. before `bb_setup_sorted_S`):

```cpp
/* --------------- run_sdc_cta_split ------------------- */
PaSplitStats run_sdc_cta_split(int blocks, const ECLgraph& g, DevPtr& d)
{
    int N = g.nodes;
    int worker_h = 0;
    PaSplitStats stats{};

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    auto record = [&](float& acc) {
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, beg, end);
        acc += ms;
    };

    while (worker_h != N) {
        cudaEventRecord(beg, 0);
        P_SL_ELS_SDC_split_scan<<<blocks, ThreadsPerBlock>>>(
            N, d.nidx_d, d.degree_list, d.iteration_list_d);
        record(stats.scan_ms);

        cudaEventRecord(beg, 0);
        P_SL_ELS_SDC_CTA_split_decrement<<<blocks, ThreadsPerBlock>>>(
            d.nidx_d, d.nlist_d, d.degree_list, d.iteration_list_d);
        record(stats.decrement_ms);

        P_SL_ELS_SDC_split_advance<<<1, 32>>>();
        cudaDeviceSynchronize();

        cudaMemcpyFromSymbol(&worker_h, worker, sizeof(int));
    }

    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return stats;
}
```

- [ ] **Step 3: Build**

Run: `cd CHROMA && make ARCH=sm_89 -j4 2>&1 | tail -20 && cd ..`

Expected: clean build. If `P_SL_ELS_SDC_CTA_split_decrement` is "undeclared", add a forward declaration near the top of `chroma_utils.cu` (alongside the existing forward decl for `P_SL_ELS_SDC_split_*`):

```cuda
__global__ void P_SL_ELS_SDC_CTA_split_decrement(
    const int*, const int*, unsigned int*, const unsigned int*);
```

- [ ] **Step 4: Commit**

```bash
git add CHROMA/chroma_utils.cuh CHROMA/chroma_utils.cu
git commit -m "CHROMA: add run_sdc_cta_split host driver (per-phase timing)"
```

---

## Task 6 — Add `run_sdc_cta_s_split` host launcher

**Why:** Same as Task 5 but for the dispatched CTA_S path.

**Files:**
- Modify: `CHROMA/chroma_utils.cuh`
- Modify: `CHROMA/chroma_utils.cu`

- [ ] **Step 1: Add the declaration in `chroma_utils.cuh`**

After the `run_sdc_cta_split` declaration, append:

```cpp
/* CTA_S (dispatched warp/CTA) SDC split-phase host driver. Phase 2 uses
 * P_SL_ELS_SDC_CTA_S_split_decrement which itself dispatches per-block
 * between SDC-warp and CTA-balanced based on remove_size. */
PaSplitStats run_sdc_cta_s_split(int blocks, const ECLgraph& g, DevPtr& d);
```

- [ ] **Step 2: Add the definition in `chroma_utils.cu`**

Append right after `run_sdc_cta_split`:

```cpp
/* --------------- run_sdc_cta_s_split ------------------- */
PaSplitStats run_sdc_cta_s_split(int blocks, const ECLgraph& g, DevPtr& d)
{
    int N = g.nodes;
    int worker_h = 0;
    PaSplitStats stats{};

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    auto record = [&](float& acc) {
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, beg, end);
        acc += ms;
    };

    while (worker_h != N) {
        cudaEventRecord(beg, 0);
        P_SL_ELS_SDC_split_scan<<<blocks, ThreadsPerBlock>>>(
            N, d.nidx_d, d.degree_list, d.iteration_list_d);
        record(stats.scan_ms);

        cudaEventRecord(beg, 0);
        P_SL_ELS_SDC_CTA_S_split_decrement<<<blocks, ThreadsPerBlock>>>(
            d.nidx_d, d.nlist_d, d.degree_list, d.iteration_list_d);
        record(stats.decrement_ms);

        P_SL_ELS_SDC_split_advance<<<1, 32>>>();
        cudaDeviceSynchronize();

        cudaMemcpyFromSymbol(&worker_h, worker, sizeof(int));
    }

    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return stats;
}
```

- [ ] **Step 3: Build**

Run: `cd CHROMA && make ARCH=sm_89 -j4 2>&1 | tail -20 && cd ..`

Expected: clean build. If `P_SL_ELS_SDC_CTA_S_split_decrement` is "undeclared", add a forward decl next to the one added in Task 5.

- [ ] **Step 4: Commit**

```bash
git add CHROMA/chroma_utils.cuh CHROMA/chroma_utils.cu
git commit -m "CHROMA: add run_sdc_cta_s_split host driver (per-phase timing)"
```

---

## Task 7 — Wire two new algo strings in `select_algorithm` and update the help banner

**Why:** Make ids `11` and `12` selectable from the CLI. Existing ids 0–10 are already taken (0–4 cuSL_ELS variants, 5 BB, 6 BB_SPLIT, 7 SDC_SPLIT, 8 CTA, 9 CTA_W, 10 CTA_S). The new ids are `11 = cuSL_ELS_SDC_CTA_SPLIT` and `12 = cuSL_ELS_SDC_CTA_S_SPLIT`.

**Files:**
- Modify: `CHROMA/CHROMA.cu:56` (help banner — append two lines)
- Modify: `CHROMA/CHROMA.cu:119-122` (`select_algorithm` — add two cases before the `else` fallback)

- [ ] **Step 1: Append the two new help lines after the existing `cuSL_ELS_SDC_CTA_S` line (line 56)**

Insert directly after the line that ends with `[recommended]\n";`:

```cpp
    std::cout << "                           11 or cuSL_ELS_SDC_CTA_SPLIT   : CTA-balanced SDC, per-phase split-kernel diagnostic variant\n";
    std::cout << "                           12 or cuSL_ELS_SDC_CTA_S_SPLIT : Dispatched CTA_S SDC, per-phase split-kernel diagnostic variant\n";
```

- [ ] **Step 2: Add two new `else if` branches in `select_algorithm` before the trailing `else` fallback (line 122)**

Locate the existing block:

```cpp
    } else if (algo_str == "10" || algo_str == "cuSL_ELS_SDC_CTA_S") {
        algo_name = "cuSL_ELS_SDC_CTA_S";
        return (void*)P_SL_ELS_SDC_CTA_S;
    } else {
```

Insert these branches between the `10` block and the `else`:

```cpp
    } else if (algo_str == "11" || algo_str == "cuSL_ELS_SDC_CTA_SPLIT") {
        algo_name = "cuSL_ELS_SDC_CTA_SPLIT";
        return (void*)P_SL_ELS_SDC_CTA;   // placeholder; main flow detects by name
    } else if (algo_str == "12" || algo_str == "cuSL_ELS_SDC_CTA_S_SPLIT") {
        algo_name = "cuSL_ELS_SDC_CTA_S_SPLIT";
        return (void*)P_SL_ELS_SDC_CTA_S; // placeholder; main flow detects by name
    } else {
```

- [ ] **Step 3: Build**

Run: `cd CHROMA && make ARCH=sm_89 -j4 2>&1 | tail -20 && cd ..`

Expected: clean build.

- [ ] **Step 4: Smoke test (still uses the old kernel because dispatch isn't wired yet)**

Run: `CHROMA/CHROMA --help | grep SDC_CTA_SPLIT`

Expected: two lines including `11 or cuSL_ELS_SDC_CTA_SPLIT` and `12 or cuSL_ELS_SDC_CTA_S_SPLIT`. Do NOT run the algo yet — dispatch isn't there.

- [ ] **Step 5: Commit**

```bash
git add CHROMA/CHROMA.cu
git commit -m "CHROMA.cu: register algo ids 11/12 for SDC_CTA_SPLIT and SDC_CTA_S_SPLIT"
```

---

## Task 8 — Dispatch the new SPLIT variants and print per-phase ms

**Why:** Tie everything together. For algo names ending in `_SPLIT`, use the per-phase launchers and emit `PA scan` / `PA decrement` printouts in the per-run line and the multi-run summary. Non-SPLIT outputs unchanged.

**Files:**
- Modify: `CHROMA/CHROMA.cu` around lines 502-695

- [ ] **Step 1: Extend the per-run stat arrays (around line 502)**

Replace the existing line:

```cpp
    std::vector<float> pa_ms_arr, ca_ms_arr, reduce_ms_arr, total_ms_arr;
```

with:

```cpp
    std::vector<float> pa_ms_arr, ca_ms_arr, reduce_ms_arr, total_ms_arr;
    std::vector<float> pa_scan_ms_arr, pa_decrement_ms_arr;
    const bool is_split = (algo_name == "cuSL_ELS_SDC_SPLIT"     ||
                           algo_name == "cuSL_ELS_SDC_CTA_SPLIT" ||
                           algo_name == "cuSL_ELS_SDC_CTA_S_SPLIT");
```

- [ ] **Step 2: Replace the SPLIT dispatch block (around lines 544-549) so it captures `PaSplitStats`**

Find:

```cpp
        } else if (algo_name == "cuSL_ELS_SDC_SPLIT") {
            int blkPerSM_sdc_split;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM_sdc_split,
                P_SL_ELS_SDC_split_scan, ThreadsPerBlock, 0);
            int gridDim_sdc_split = blkPerSM_sdc_split * SMs;
            run_sdc_split(gridDim_sdc_split, g, d);
        } else {
```

Replace with:

```cpp
        } else if (is_split) {
            int blkPerSM_sdc_split;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM_sdc_split,
                P_SL_ELS_SDC_split_scan, ThreadsPerBlock, 0);
            int gridDim_sdc_split = blkPerSM_sdc_split * SMs;
            PaSplitStats sp{};
            if (algo_name == "cuSL_ELS_SDC_SPLIT") {
                sp = run_sdc_split(gridDim_sdc_split, g, d);
            } else if (algo_name == "cuSL_ELS_SDC_CTA_SPLIT") {
                sp = run_sdc_cta_split(gridDim_sdc_split, g, d);
            } else /* cuSL_ELS_SDC_CTA_S_SPLIT */ {
                sp = run_sdc_cta_s_split(gridDim_sdc_split, g, d);
            }
            // Stash per-phase ms in run-local variables; they'll be
            // recorded into the arrays after timer_PA stops.
            pa_scan_ms_arr.push_back(sp.scan_ms);
            pa_decrement_ms_arr.push_back(sp.decrement_ms);
        } else {
```

- [ ] **Step 3: Pad `pa_scan_ms_arr` / `pa_decrement_ms_arr` for non-SPLIT runs so indices line up across all algos**

Find the existing block (around lines 662-666):

```cpp
        pa_ms_arr.push_back(runtime_PA * 1000);
        ca_ms_arr.push_back(runtime_CA * 1000);
        reduce_ms_arr.push_back(reduction_stats.runtime_sec * 1000);
        total_ms_arr.push_back(total_runtime * 1000);
        colors_arr.push_back(colors_after);
        iter_count_arr.push_back(host_iter_count);
```

Replace with:

```cpp
        pa_ms_arr.push_back(runtime_PA * 1000);
        ca_ms_arr.push_back(runtime_CA * 1000);
        reduce_ms_arr.push_back(reduction_stats.runtime_sec * 1000);
        total_ms_arr.push_back(total_runtime * 1000);
        colors_arr.push_back(colors_after);
        iter_count_arr.push_back(host_iter_count);
        if (!is_split) {
            // Keep arrays in lockstep so the multi-run summary doesn't need
            // separate length tracking. Non-SPLIT runs report 0/0.
            pa_scan_ms_arr.push_back(0.0f);
            pa_decrement_ms_arr.push_back(0.0f);
        }
```

- [ ] **Step 4: Extend the per-run printout for SPLIT runs (around lines 636-660)**

Find:

```cpp
        if (num_runs > 1) {
            printf("[Run %d/%d] PA: %8.3f ms  CA: %7.3f ms  Reduce: %6.3f ms  "
                   "Total: %8.3f ms  colors: %d  iters: %d\n",
                   run_idx + 1, num_runs,
                   runtime_PA * 1000, runtime_CA * 1000,
                   reduction_stats.runtime_sec * 1000, total_runtime * 1000,
                   colors_after, host_iter_count);
        } else {
            // Backward-compatible single-run output (verbose).
            std::cout << "Finish PA" << (is_fused ? "+Init" : "") << std::endl;
            std::cout << "Finish CA" << (is_fused ? " (coloring only)" : "")
                      << std::endl;
            printf("PA runtime: %.6f ms\n", runtime_PA * 1000);
            printf("CA runtime: %.6f ms\n", runtime_CA * 1000);
            printf("Post reduction runtime: %.6f ms\n",
                   reduction_stats.runtime_sec * 1000);
            printf("Total runtime: %.6f ms\n", total_runtime * 1000);
            ...
        }
```

Replace with:

```cpp
        if (num_runs > 1) {
            if (is_split) {
                printf("[Run %d/%d] PA: %8.3f ms (scan: %7.3f dec: %7.3f)  "
                       "CA: %7.3f ms  Reduce: %6.3f ms  Total: %8.3f ms  "
                       "colors: %d  iters: %d\n",
                       run_idx + 1, num_runs,
                       runtime_PA * 1000,
                       pa_scan_ms_arr.back(),
                       pa_decrement_ms_arr.back(),
                       runtime_CA * 1000,
                       reduction_stats.runtime_sec * 1000, total_runtime * 1000,
                       colors_after, host_iter_count);
            } else {
                printf("[Run %d/%d] PA: %8.3f ms  CA: %7.3f ms  Reduce: %6.3f ms  "
                       "Total: %8.3f ms  colors: %d  iters: %d\n",
                       run_idx + 1, num_runs,
                       runtime_PA * 1000, runtime_CA * 1000,
                       reduction_stats.runtime_sec * 1000, total_runtime * 1000,
                       colors_after, host_iter_count);
            }
        } else {
            // Backward-compatible single-run output (verbose).
            std::cout << "Finish PA" << (is_fused ? "+Init" : "") << std::endl;
            std::cout << "Finish CA" << (is_fused ? " (coloring only)" : "")
                      << std::endl;
            printf("PA runtime: %.6f ms\n", runtime_PA * 1000);
            if (is_split) {
                printf("PA scan runtime: %.6f ms\n",      pa_scan_ms_arr.back());
                printf("PA decrement runtime: %.6f ms\n", pa_decrement_ms_arr.back());
            }
            printf("CA runtime: %.6f ms\n", runtime_CA * 1000);
            printf("Post reduction runtime: %.6f ms\n",
                   reduction_stats.runtime_sec * 1000);
            printf("Total runtime: %.6f ms\n", total_runtime * 1000);
            printf("colors before reduction: %d\n", reduction_stats.colors_before);
            printf("colors after reduction: %d\n", reduction_stats.colors_after);
            printf("color reduction delta: %d\n",
                   reduction_stats.colors_before - reduction_stats.colors_after);
            printf("result verification passed\n");
            printf("colors used: %d\n", colors_after);
            printf("Iter count: %d\n", host_iter_count);
        }
```

(Note: keep the existing lines after the `Iter count` printout unchanged; only the block above changes.)

- [ ] **Step 5: Extend the multi-run summary (around lines 688-694)**

Find:

```cpp
        printf("\n=== Statistics over %d runs (ms) ===\n", num_runs);
        printf("PA time     : %s\n", stats_f(pa_ms_arr).c_str());
        printf("CA time     : %s\n", stats_f(ca_ms_arr).c_str());
        printf("Reduce time : %s\n", stats_f(reduce_ms_arr).c_str());
        printf("Total time  : %s\n", stats_f(total_ms_arr).c_str());
        printf("colors used : %s\n", stats_i(colors_arr).c_str());
        printf("iter count  : %s\n", stats_i(iter_count_arr).c_str());
```

Replace with:

```cpp
        printf("\n=== Statistics over %d runs (ms) ===\n", num_runs);
        printf("PA time     : %s\n", stats_f(pa_ms_arr).c_str());
        if (is_split) {
            printf("PA scan     : %s\n", stats_f(pa_scan_ms_arr).c_str());
            printf("PA decrement: %s\n", stats_f(pa_decrement_ms_arr).c_str());
        }
        printf("CA time     : %s\n", stats_f(ca_ms_arr).c_str());
        printf("Reduce time : %s\n", stats_f(reduce_ms_arr).c_str());
        printf("Total time  : %s\n", stats_f(total_ms_arr).c_str());
        printf("colors used : %s\n", stats_i(colors_arr).c_str());
        printf("iter count  : %s\n", stats_i(iter_count_arr).c_str());
```

- [ ] **Step 6: Build**

Run: `cd CHROMA && make ARCH=sm_89 -j4 2>&1 | tail -20 && cd ..`

Expected: clean build.

- [ ] **Step 7: Commit**

```bash
git add CHROMA/CHROMA.cu
git commit -m "$(cat <<'EOF'
CHROMA.cu: dispatch + per-phase printing for SDC_*_SPLIT variants

For algo names ending in _SPLIT (7, 11, 12) the run loop now calls the
new per-phase launchers, records scan_ms / decrement_ms per run, and
prints them in the per-run line, the verbose single-run block, and the
multi-run summary. Non-SPLIT output is byte-identical to before.
EOF
)"
```

---

## Task 9 — Smoke-test the three SPLIT variants on facebook.egr

**Why:** Verify the new kernels run, produce a valid colouring, and emit `PA scan` / `PA decrement` numbers that are positive and consistent.

**Files:**
- (none — this is verification only)

- [ ] **Step 1: Run all three SPLIT variants single-run on facebook**

Run:

```bash
cd /home/chsieh45/PunchShadow/CHROMA
for algo in cuSL_ELS_SDC_SPLIT cuSL_ELS_SDC_CTA_SPLIT cuSL_ELS_SDC_CTA_S_SPLIT; do
    echo "=== $algo ==="
    ./CHROMA/CHROMA -f Datasets/test/facebook.egr -a "$algo" -e 0
done
```

Expected for each:
- `result verification passed`
- `colors used: <n>` (around 70-75 for facebook)
- A line `PA scan runtime: X.XXXXXX ms` (X > 0)
- A line `PA decrement runtime: Y.YYYYYY ms` (Y > 0)
- `PA runtime`, `CA runtime`, `Total runtime` all positive.

- [ ] **Step 2: Multi-run sanity check (5 runs)**

Run:

```bash
./CHROMA/CHROMA -f Datasets/test/facebook.egr -a cuSL_ELS_SDC_CTA_S_SPLIT -e 0 --runs 5
```

Expected output includes:
- Five `[Run i/5] PA: ... (scan: ... dec: ...) CA: ... Reduce: ... Total: ...` lines
- A `=== Statistics over 5 runs (ms) ===` block containing both `PA scan` and `PA decrement` rows.

- [ ] **Step 3: Cross-check: scan + decrement ≈ PA**

Eyeball: `PA scan + PA decrement` should be roughly equal to or slightly less than `PA time` (the outer wrap includes `init_degree` + the `cudaMemcpyFromSymbol` per iter). If they're off by orders of magnitude, the per-phase timers are broken — revisit Tasks 4-6.

- [ ] **Step 4: Diff non-SPLIT output to confirm zero regression**

Run:

```bash
./CHROMA/CHROMA -f Datasets/test/facebook.egr -a cuSL_ELS_SDC_CTA_S -e 0 --runs 3
```

Expected: NO `PA scan`/`PA decrement` lines anywhere. Output should be byte-identical to before this PR. If the new lines leak into non-SPLIT runs, the `is_split` gating in Task 8 is wrong.

- [ ] **Step 5: Commit nothing** (this task is verification only)

---

## Task 10 — Create `scripts/batch_profile.py`

**Why:** Drive the 3-framework × 19-EGR-dataset × 5-run sweep and write the JSON the plot script consumes.

**Files:**
- Create: `scripts/batch_profile.py`

- [ ] **Step 1: Create the file with the full sweep driver**

Write the following to `scripts/batch_profile.py`:

```python
#!/usr/bin/env python3
"""Sweep three CHROMA SPLIT-mode frameworks over EGR datasets and capture
per-phase (CA / PA scan / PA decrement) timings.

For each (framework, dataset) pair the script invokes the CHROMA binary
with --runs N (default 5), parses the `=== Statistics over N runs (ms) ===`
block from stdout, and writes a JSON suitable for
`scripts/plots/plot_execution_breakdown.py`.

Examples:
    python3 scripts/batch_profile.py
    python3 scripts/batch_profile.py --only facebook le450_25d --runs 3
    python3 scripts/batch_profile.py --frameworks cuSL_ELS_SDC_CTA_SPLIT \\
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
    "cuSL_ELS_SDC_SPLIT",
    "cuSL_ELS_SDC_CTA_SPLIT",
    "cuSL_ELS_SDC_CTA_S_SPLIT",
]

# stats_f produces  "avg=%9.3f  min=%9.3f  max=%9.3f" — \s* covers padding.
STAT_RE = {
    "ca_ms":           re.compile(r"^CA time\s*:\s*avg=\s*([0-9.]+)",          re.MULTILINE),
    "pa_ms":           re.compile(r"^PA time\s*:\s*avg=\s*([0-9.]+)",          re.MULTILINE),
    "pa_scan_ms":      re.compile(r"^PA scan\s*:\s*avg=\s*([0-9.]+)",          re.MULTILINE),
    "pa_decrement_ms": re.compile(r"^PA decrement\s*:\s*avg=\s*([0-9.]+)",     re.MULTILINE),
    "total_ms":        re.compile(r"^Total time\s*:\s*avg=\s*([0-9.]+)",       re.MULTILINE),
    "colors_used":     re.compile(r"^colors used\s*:\s*avg=\s*([0-9.]+)",      re.MULTILINE),
}

EGR_HEADER_RE = re.compile(rb"^")  # placeholder — header parsed binary-style below


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
    for key, rx in STAT_RE.items():
        m = rx.search(stdout)
        if m is None:
            return None
        out[key] = float(m.group(1))
    return out


def run_one(binary: Path, egr: Path, framework: str, runs: int, timeout: int):
    cmd = [str(binary), "-f", str(egr), "-a", framework, "-e", "0", "--runs", str(runs)]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",      default=str(repo / "CHROMA" / "CHROMA"))
    ap.add_argument("--dataset-dir", default=str(repo / "Datasets" / "EGR"))
    ap.add_argument("--runs",        type=int, default=5)
    ap.add_argument("--timeout",     type=int, default=1200,
                    help="Per-cell timeout (seconds). SPLIT mode is slow.")
    ap.add_argument("--frameworks",  nargs="+", default=DEFAULT_FRAMEWORKS)
    ap.add_argument("--only",        nargs="*", default=None,
                    help="Restrict to dataset stems (matched by .egr basename)")
    ap.add_argument("--skip",        nargs="*", default=[])
    ap.add_argument("--out",         default=str(repo / "scripts" / "batch_profile_results.json"))
    args = ap.parse_args()

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

    print(f"# {len(egrs)} datasets x {len(args.frameworks)} frameworks x "
          f"{args.runs} runs", file=sys.stderr)

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
            stats, wall, err = run_one(bin_path, egr, fw, args.runs, args.timeout)
            row = {
                "framework": fw,
                "dataset":   name,
                "nodes":     nodes,
                "edges":     edges,
                "runs":      args.runs,
                "wall_s":    wall,
            }
            if err:
                row["error"] = err
                print(f"# {name:32s} {fw:30s} FAIL  ({err})", file=sys.stderr)
            else:
                row.update(stats)
                print(f"# {name:32s} {fw:30s} "
                      f"ca={stats['ca_ms']:7.2f}ms "
                      f"scan={stats['pa_scan_ms']:7.2f}ms "
                      f"dec={stats['pa_decrement_ms']:7.2f}ms "
                      f"colors={stats['colors_used']:5.1f} "
                      f"wall={wall:6.1f}s", file=sys.stderr)
            rows.append(row)

    summary = {
        "frameworks": list(args.frameworks),
        "datasets":   datasets_meta,
        "rows":       rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"# wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/batch_profile.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/batch_profile.py
git commit -m "scripts: add batch_profile.py — 3-framework x 19-EGR sweep with per-phase ms"
```

---

## Task 11 — Smoke-test batch_profile.py on 2 datasets

**Why:** Make sure the JSON contract is well-formed before kicking off the full ~60-cell sweep.

**Files:**
- (none — this is verification only)

- [ ] **Step 1: Run a tiny sweep**

Run:

```bash
cd /home/chsieh45/PunchShadow/CHROMA
python3 scripts/batch_profile.py \
    --only facebook le450_25d \
    --runs 2 \
    --out /tmp/batch_profile_smoke.json
```

Expected stderr: 2 datasets × 3 frameworks × 2 runs = 6 lines, all showing non-zero `ca=`, `scan=`, `dec=` numbers.

- [ ] **Step 2: Verify JSON shape**

Run:

```bash
python3 -c "
import json
d = json.load(open('/tmp/batch_profile_smoke.json'))
assert d['frameworks'] == ['cuSL_ELS_SDC_SPLIT', 'cuSL_ELS_SDC_CTA_SPLIT', 'cuSL_ELS_SDC_CTA_S_SPLIT']
assert {ds['name'] for ds in d['datasets']} == {'facebook', 'le450_25d'}
assert len(d['rows']) == 6
for r in d['rows']:
    assert 'error' not in r, r
    for k in ('ca_ms','pa_scan_ms','pa_decrement_ms','pa_ms','total_ms','colors_used'):
        assert r[k] > 0, (r['dataset'], r['framework'], k, r.get(k))
print('JSON OK:', len(d['rows']), 'rows')
"
```

Expected: prints `JSON OK: 6 rows`.

- [ ] **Step 3: Cleanup**

Run: `rm -f /tmp/batch_profile_smoke.json`

- [ ] **Step 4: Commit nothing** (verification only)

---

## Task 12 — Create `scripts/plots/plot_execution_breakdown.py`

**Why:** Read the JSON and render the stacked grouped-bar figure per the spec.

**Files:**
- Create: `scripts/plots/plot_execution_breakdown.py`

- [ ] **Step 1: Ensure the directory exists**

Run: `mkdir -p scripts/plots`

- [ ] **Step 2: Write the plot script**

```python
#!/usr/bin/env python3
"""Render the execution-time breakdown figure from batch_profile_results.json.

Layout: 19 dataset groups on the x-axis (sorted by edge count ascending),
three grouped bars per group (one per framework, hatched), each bar a
3-segment stack of CA (bottom) / PA scan / PA decrement (top). Two
horizontal legends at the top of the figure: stack colours, framework
hatches. No title.

Examples:
    python3 scripts/plots/plot_execution_breakdown.py
    python3 scripts/plots/plot_execution_breakdown.py --log
    python3 scripts/plots/plot_execution_breakdown.py \\
        --in scripts/batch_profile_results.json \\
        --out-prefix /tmp/figure --figsize 16 5
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
import numpy as np


SEGMENTS = [
    ("ca_ms",            "CA",           "#4C72B0"),  # blue
    ("pa_scan_ms",       "PA scan",      "#DD8452"),  # orange
    ("pa_decrement_ms",  "PA decrement", "#55A467"),  # green
]
HATCHES = ["",  "//", "xx"]  # one per framework, in input order

FRAMEWORK_LABELS = {
    "cuSL_ELS_SDC_SPLIT":        "cuSL_ELS_SDC",
    "cuSL_ELS_SDC_CTA_SPLIT":    "cuSL_ELS_SDC_CTA",
    "cuSL_ELS_SDC_CTA_S_SPLIT":  "cuSL_ELS_SDC_CTA_S",
}


def load(path: Path):
    d = json.loads(path.read_text())
    by_key = {(r["framework"], r["dataset"]): r for r in d["rows"]}
    return d["frameworks"], d["datasets"], by_key


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(repo / "scripts" / "batch_profile_results.json"))
    ap.add_argument("--out-prefix",
                    default=str(repo / "scripts" / "plots" / "execution_time_breakdown"))
    ap.add_argument("--log", action="store_true",
                    help="Use log y-axis (symlog with small linthresh).")
    ap.add_argument("--figsize", nargs=2, type=float, default=[14.0, 5.0])
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run batch_profile.py first.", file=sys.stderr)
        sys.exit(1)

    frameworks, datasets, by_key = load(in_path)
    datasets = sorted(datasets, key=lambda d: d["edges"])
    dataset_names = [d["name"] for d in datasets]
    n_ds = len(datasets)
    n_fw = len(frameworks)

    fig, ax = plt.subplots(figsize=tuple(args.figsize), constrained_layout=False)

    bar_w = 0.27
    group_centers = np.arange(n_ds)
    offsets = (np.arange(n_fw) - (n_fw - 1) / 2) * bar_w  # symmetric around center

    for fw_idx, fw in enumerate(frameworks):
        xs = group_centers + offsets[fw_idx]
        bottoms = np.zeros(n_ds)
        for seg_key, _seg_label, seg_color in SEGMENTS:
            heights = np.array([by_key.get((fw, name), {}).get(seg_key, 0.0)
                                for name in dataset_names])
            ax.bar(xs, heights, width=bar_w, bottom=bottoms,
                   color=seg_color,
                   edgecolor="black", linewidth=0.8,
                   hatch=HATCHES[fw_idx])
            bottoms += heights

    ax.set_xticks(group_centers)
    ax.set_xticklabels(dataset_names, rotation=35, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_xlim(-0.5, n_ds - 0.5)
    if args.log:
        ax.set_yscale("symlog", linthresh=0.1)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Two horizontal legends above the axes.
    seg_handles = [mpatches.Patch(facecolor=c, edgecolor="black",
                                   linewidth=0.8, label=lbl)
                   for _k, lbl, c in SEGMENTS]
    fw_handles = [mpatches.Patch(facecolor="lightgrey", edgecolor="black",
                                  linewidth=0.8, hatch=HATCHES[i],
                                  label=FRAMEWORK_LABELS.get(fw, fw))
                  for i, fw in enumerate(frameworks)]

    leg1 = fig.legend(handles=seg_handles, loc="upper center",
                      bbox_to_anchor=(0.5, 1.00), ncol=len(SEGMENTS),
                      frameon=False, handlelength=2.0, columnspacing=2.0)
    fig.add_artist(leg1)
    fig.legend(handles=fw_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.95), ncol=len(frameworks),
               frameon=False, handlelength=2.0, columnspacing=2.0)

    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.06, right=0.99)

    pdf_path = Path(args.out_prefix + ".pdf")
    png_path = Path(args.out_prefix + ".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    print(f"# wrote {pdf_path}\n# wrote {png_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make it executable**

Run: `chmod +x scripts/plots/plot_execution_breakdown.py`

- [ ] **Step 4: Smoke-test against synthetic data (no full sweep yet)**

Build a tiny synthetic JSON, render it, and confirm the figure opens.

Run:

```bash
python3 - <<'EOF'
import json, pathlib
rows = []
fwts = ["cuSL_ELS_SDC_SPLIT", "cuSL_ELS_SDC_CTA_SPLIT", "cuSL_ELS_SDC_CTA_S_SPLIT"]
dss  = [
  {"name":"le450_25d","nodes":450,"edges":17415},
  {"name":"school1","nodes":385,"edges":19095},
  {"name":"facebook","nodes":4039,"edges":88234},
]
for ds in dss:
    for k, fw in enumerate(fwts):
        rows.append({
          "framework": fw, "dataset": ds["name"], "nodes": ds["nodes"],
          "edges": ds["edges"], "runs": 3, "wall_s": 0.3,
          "ca_ms": 5.0+k, "pa_scan_ms": 1.0+0.5*k,
          "pa_decrement_ms": 2.0-0.5*k, "pa_ms": 3.0,
          "total_ms": 8.0, "colors_used": 30+k,
        })
pathlib.Path("/tmp/plot_smoke.json").write_text(json.dumps(
    {"frameworks":fwts,"datasets":dss,"rows":rows}))
EOF

python3 scripts/plots/plot_execution_breakdown.py \
    --in /tmp/plot_smoke.json \
    --out-prefix /tmp/plot_smoke
ls -la /tmp/plot_smoke.pdf /tmp/plot_smoke.png
```

Expected:
- `/tmp/plot_smoke.pdf` and `/tmp/plot_smoke.png` exist, both non-empty.
- Visual sanity: open `/tmp/plot_smoke.png` and confirm 3 dataset groups, each with 3 hatched bars, each bar stacked 3 colours; two horizontal legends at top; no title.

- [ ] **Step 5: Cleanup**

Run: `rm -f /tmp/plot_smoke.json /tmp/plot_smoke.pdf /tmp/plot_smoke.png`

- [ ] **Step 6: Commit**

```bash
git add scripts/plots/plot_execution_breakdown.py
git commit -m "scripts/plots: add plot_execution_breakdown.py — stacked grouped-bar figure"
```

---

## Task 13 — Run the full 19×3×5 sweep

**Why:** Produce the real data for the figure. SPLIT mode is slow; expect ~30 minutes of wall time.

**Files:**
- (none — this is execution only; writes `scripts/batch_profile_results.json`)

- [ ] **Step 1: Kick off the full sweep**

Run:

```bash
cd /home/chsieh45/PunchShadow/CHROMA
python3 scripts/batch_profile.py 2>&1 | tee scripts/batch_profile_full.log
```

Expected stderr (mirrored to the log file): 19 × 3 = 57 result lines, one per (dataset, framework) cell.

- [ ] **Step 2: Sanity check that all cells succeeded**

Run:

```bash
python3 -c "
import json
d = json.load(open('scripts/batch_profile_results.json'))
n = len(d['rows'])
errs = [r for r in d['rows'] if 'error' in r]
print(f'{n} rows, {len(errs)} errors')
for r in errs:
    print('  FAIL:', r['framework'], r['dataset'], r['error'])
"
```

Expected: `57 rows, 0 errors`. If any cell errored, decide whether to retry it (e.g. add `--only <failing dataset>`) or proceed with a missing bar group.

- [ ] **Step 3: Commit the sweep output**

```bash
git add scripts/batch_profile_results.json scripts/batch_profile_full.log
git commit -m "scripts: capture batch_profile sweep over 19 EGR datasets x 3 SPLIT frameworks"
```

---

## Task 14 — Render the final figure

**Why:** Produce the deliverable that motivated the whole change.

**Files:**
- Create (as artifact): `scripts/plots/execution_time_breakdown.pdf`
- Create (as artifact): `scripts/plots/execution_time_breakdown.png`

- [ ] **Step 1: Render**

Run:

```bash
python3 scripts/plots/plot_execution_breakdown.py
```

Expected: stderr shows two `# wrote ...` lines for the PDF and PNG paths.

- [ ] **Step 2: Visual verification**

Open `scripts/plots/execution_time_breakdown.png` (or copy to a local machine if working over SSH). Confirm:
- 19 dataset groups along x, sorted from smallest edge count (`le450_25d`) on the left to largest on the right.
- Each group has 3 hatched bars side-by-side; bars are visually distinct.
- Each bar is a 3-segment stack with distinct colours for CA / PA scan / PA decrement.
- Two horizontal legends sit above the axes (colour legend on top, hatch legend below it).
- No graph title; y-axis labelled "Time (ms)"; dataset labels rotated, not overlapping.

- [ ] **Step 3 (optional): Render a log-scale version for dynamic range comparison**

Run:

```bash
python3 scripts/plots/plot_execution_breakdown.py --log \
    --out-prefix scripts/plots/execution_time_breakdown_log
```

- [ ] **Step 4: Commit the artifacts**

```bash
git add scripts/plots/execution_time_breakdown.pdf scripts/plots/execution_time_breakdown.png
if [ -f scripts/plots/execution_time_breakdown_log.pdf ]; then
    git add scripts/plots/execution_time_breakdown_log.pdf scripts/plots/execution_time_breakdown_log.png
fi
git commit -m "scripts/plots: render execution-time breakdown figure"
```

---

## Self-review checklist (run after the plan is written)

1. **Spec coverage** — every spec section maps to at least one task:
   - SPLIT kernels for CTA/CTA_S → Tasks 2, 3
   - Per-phase timing in launchers → Tasks 4, 5, 6
   - CHROMA.cu dispatch + printouts → Tasks 7, 8
   - Verification → Task 9
   - `batch_profile.py` → Tasks 10, 11
   - `plot_execution_breakdown.py` → Task 12
   - Full sweep + figure → Tasks 13, 14
2. **Placeholders** — no `TODO`/`TBD`/"similar to" placeholders; every step includes the exact code or command.
3. **Type / name consistency:**
   - `PaSplitStats { float scan_ms; float decrement_ms; }` — used in Tasks 4-6 + 8; spelt the same everywhere.
   - Launcher names `run_sdc_split` / `run_sdc_cta_split` / `run_sdc_cta_s_split` — consistent across `chroma_utils.cuh`, `chroma_utils.cu`, `CHROMA.cu`.
   - Algo strings `cuSL_ELS_SDC_SPLIT` / `cuSL_ELS_SDC_CTA_SPLIT` / `cuSL_ELS_SDC_CTA_S_SPLIT` — consistent in select_algorithm, dispatch, batch_profile.py, and plot script.
   - JSON keys `ca_ms` / `pa_scan_ms` / `pa_decrement_ms` / `pa_ms` / `total_ms` — used in both sweep and plot scripts.
4. **Scope** — single PR-sized: source + sweep + plot. No unrelated refactor.

If any of these fail on re-read, fix inline and proceed.

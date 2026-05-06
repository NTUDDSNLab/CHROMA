// CHROMA/BB_split.cu — Per-phase split-kernel diagnostic variant of BB-cuSL
//
// cuSL_ELS_BB_SPLIT (id 6) replicates BB-cuSL phase logic exactly but uses
// a separate kernel launch at each grid.sync() boundary so that NCU can
// report per-phase timing. This destroys total performance but allows
// profiling each phase independently.
//
// Each kernel corresponds to one "segment" between consecutive grid.sync()
// calls in BB.cu.  The host launcher (run_bb_split in chroma_utils.cu) does
// cudaDeviceSynchronize() between every pair, which replaces the grid.sync().
//
// Phase mapping vs BB.cu grid.sync() calls:
//   phase0a  : find min degree, zero iteration_list      [before line-43 sync]
//   phase0b  : set theta = g_minDegree, reset g_minDegree [before line-49 sync]
//   phase0c  : fill in-window buckets, set bb_init_done  [before line-67/71 sync]
//   phase1   : peel window buckets                        [before line-103 sync]
//   phase1r  : reset bucket_counts for current window     [before line-115 sync]
//   phase2   : decrement neighbours + push                [before line-143 sync]
//   phase3   : advance theta, set overflow latch           [before line-170 sync]
//   phase4a  : scan unpeeled, atomicMin                   [before line-183 sync]
//   phase4b  : set theta = g_minDegree, reset counts      [before line-191 sync]
//   phase4c  : refill buckets from unpeeled               [before line-211 sync]

#include "globals.cuh"
#include <climits>
#include <cstdio>

// ────────────────────────────────────────────────────────────────────────────
// Phase 0a: zero iteration_list and find min degree via atomicMin
// Corresponds to BB.cu lines 37-43 (before first grid.sync() in Phase 0)
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase0a_find_theta(
    int N,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threads = gridDim.x * blockDim.x;

    unsigned int local_min = UINT_MAX;
    for (int v = tid; v < N; v += threads) {
        iteration_list[v] = 0;
        unsigned int d = degree_list[v];
        if (d < local_min) local_min = d;
    }
    if (local_min < UINT_MAX) atomicMin(&g_minDegree, (int)local_min);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 0b: thread 0 sets theta = g_minDegree, resets g_minDegree.
// Single-thread kernel — safe to read/write device globals without grid sync.
// Corresponds to BB.cu lines 45-49 (between first and second grid.sync())
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase0b_set_theta()
{
    // Only thread 0 in this single-thread launch does work
    theta = g_minDegree;
    atomicExch(&g_minDegree, INT_MAX);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 0c: fill in-window buckets; mark bb_init_done.
// Reads theta and bb_window from device globals (set by phase0b above, visible
// after the host cudaDeviceSynchronize() that separates the launches).
// Corresponds to BB.cu lines 52-71 (between second and fourth grid.sync())
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase0c_fill_buckets(
    int N,
    const unsigned int* __restrict__ degree_list)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threads = gridDim.x * blockDim.x;
    const int lane    = threadIdx.x & 31;

    int curr_theta = theta;    // visible: host sync after phase0b
    int window     = bb_window;

    for (int v = tid; v < N; v += threads) {
        unsigned int d = degree_list[v];
        if (d < (unsigned int)(curr_theta + window)) {
            int slot = (int)d % window;
            int idx  = atomicAdd(&bb_bucket_count[slot], 1);
            if (idx < bb_bucket_capacity) {
                bb_bucket_data[(size_t)slot * (size_t)N + idx] = v;
            } else {
                atomicSub(&bb_bucket_count[slot], 1);
                if (lane == 0)
                    printf("BB_split: bucket overflow in Phase 0c (slot %d)\n", slot);
            }
        }
    }

    // Mark init done (thread 0 only, but in a non-cooperative kernel this
    // write is visible to subsequent kernels after host sync)
    if (tid == 0) bb_init_done = 1;
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 1: peel window buckets (lazy-validate + atomicCAS-claim)
// peel_iter is host-maintained and passed as argument.
// Corresponds to BB.cu lines 80-102 (before line-103 grid.sync())
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase1_peel(
    int N,
    const int* __restrict__ nidx,
    const unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list,
    int peel_iter)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threads = gridDim.x * blockDim.x;

    int curr_theta = theta;
    int window     = bb_window;

    for (int slot_off = 0; slot_off < window; slot_off++) {
        int d        = curr_theta + slot_off;
        int physical = d % window;
        int cnt      = bb_bucket_count[physical];

        for (int i = tid; i < cnt; i += threads) {
            int v = bb_bucket_data[(size_t)physical * (size_t)N + i];
            unsigned int it_v = iteration_list[v];

            if (it_v & 0x40000000u) continue;                 // already peeled
            if (degree_list[v] != (unsigned int)d) continue;  // stale degree

            unsigned int prio_v = (unsigned int)peel_iter * (unsigned int)window
                                + (unsigned int)(d - curr_theta + 1);
            int  degv  = nidx[v + 1] - nidx[v];
            unsigned int large = (degv >= WS) ? 1u : 0u;
            unsigned int newval = (large << 31) | (1u << 30) | prio_v;

            if (atomicCAS(&iteration_list[v], it_v, newval) == it_v) {
                remove_list[atomicAdd(&remove_size, 1)] = v;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 1 reset: zero bb_bucket_count for all slots in the current window.
// Single-block, single-thread kernel.
// Corresponds to BB.cu lines 109-114 (between line-103 and line-115 sync)
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase1_reset()
{
    if (threadIdx.x != 0) return;

    int curr_theta = theta;
    int window     = bb_window;
    for (int slot_off = 0; slot_off < window; slot_off++) {
        int physical = (curr_theta + slot_off) % window;
        bb_bucket_count[physical] = 0;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 2: warp-per-vertex decrement neighbours + push into new bucket slot
// Corresponds to BB.cu lines 118-142 (before line-143 grid.sync())
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase2_decrement(
    int N,
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

    int curr_theta = theta;
    int window     = bb_window;
    int rs         = remove_size;   // remove_size finalized by phase1_peel; visible after host sync

    for (int k = warpId; k < rs; k += numWarp) {
        int v   = remove_list[k];
        int beg = nidx[v];
        int end = nidx[v + 1];

        unsigned int warpMin = UINT_MAX;

        for (int i = beg + lane; i < end; i += WS) {
            int u = nlist[i];
            unsigned int it_u = __ldg(&iteration_list[u]);
            if (it_u & 0x40000000u) continue;       // u already peeled

            unsigned int new_d = atomicSub(&degree_list[u], 1) - 1;
            warpMin = min(warpMin, new_d);          // Plan A: track post-decrement min

            if (new_d >= (unsigned int)curr_theta &&
                new_d < (unsigned int)(curr_theta + window)) {
                int physical = (int)new_d % window;
                int idx = atomicAdd(&bb_bucket_count[physical], 1);
                if (idx < bb_bucket_capacity) {
                    bb_bucket_data[(size_t)physical * (size_t)N + idx] = u;
                } else {
                    atomicSub(&bb_bucket_count[physical], 1);
                    if (lane == 0)
                        printf("BB_split: bucket overflow in Phase 2 (slot %d)\n", physical);
                }
            }
        }

        // Plan A: warp-reduce + atomicMin to g_minDegree (mirrors P_SL_ELS_SDC)
        warpMin = warpReduceMin(warpMin);
        if (lane == 0 && warpMin < UINT_MAX) {
            atomicMin(&g_minDegree, (int)warpMin);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 3: single-thread kernel — advance theta, update worker/remove_size,
//   set bb_overflow_needed latch.
// Corresponds to BB.cu lines 146-169 (before line-170 grid.sync())
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase3_advance()
{
    if (threadIdx.x != 0) return;

    int curr_theta = theta;
    int window     = bb_window;
    int rs         = remove_size;

    worker      += rs;
    remove_size  = 0;

    int new_theta = curr_theta;
    while (new_theta < curr_theta + window) {
        if (bb_bucket_count[new_theta % window] > 0) break;
        new_theta++;
    }

    int captured_min = g_minDegree;
    atomicExch(&g_minDegree, INT_MAX);

    if (new_theta >= curr_theta + window) {
        // Window empty — unbounded jump via captured_min (Plan A)
        if (captured_min < INT_MAX) {
            theta              = captured_min;
            bb_overflow_needed = 2;             // refill-only Phase 4
        } else {
            bb_overflow_needed = 1;             // full Phase 4 fallback
        }
    } else {
        // Window has entries — normal advance (window scan is authoritative)
        for (int dd = curr_theta; dd < new_theta; dd++) {
            bb_bucket_count[dd % window] = 0;
        }
        theta              = new_theta;
        bb_overflow_needed = 0;
    }
    bb_peel_iter += 1;
#ifdef PROFILE
    iter_count += 1;
#endif
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 4 reset-buckets: mode==2 helper — zero all bucket_count slots.
// Theta was already set by Phase 3 (Plan A captured_min path); skip O(N) scan.
// Single-block kernel (same pattern as phase4b, minus theta assignment).
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase4_reset_buckets()
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int win = bb_window;
        for (int s = 0; s < win; s++) bb_bucket_count[s] = 0;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 4a: scan unpeeled vertices, atomicMin to g_minDegree
// Corresponds to BB.cu lines 174-182 (before line-183 grid.sync())
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase4a_scan(
    int N,
    const unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threads = gridDim.x * blockDim.x;

    unsigned int local_min = UINT_MAX;
    for (int v = tid; v < N; v += threads) {
        unsigned int it = iteration_list[v];
        if (!(it & 0x40000000u)) {
            unsigned int d = degree_list[v];
            if (d < local_min) local_min = d;
        }
    }
    if (local_min < UINT_MAX) atomicMin(&g_minDegree, (int)local_min);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 4b: thread 0 resets all bucket_count slots, sets theta = g_minDegree,
//   resets g_minDegree, clears bb_overflow_needed.
// Single-thread kernel.
// Corresponds to BB.cu lines 185-191 (before line-191 grid.sync())
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase4b_set_theta()
{
    if (threadIdx.x != 0) return;

    int win = bb_window;
    for (int s = 0; s < win; s++) bb_bucket_count[s] = 0;
    theta = g_minDegree;
    atomicExch(&g_minDegree, INT_MAX);
    bb_overflow_needed = 0;
}

// ────────────────────────────────────────────────────────────────────────────
// Phase 4c: scan unpeeled and push degree-in-window into buckets
// Corresponds to BB.cu lines 194-211 (before line-211 grid.sync())
// ────────────────────────────────────────────────────────────────────────────
__global__ void bb_split_phase4c_refill(
    int N,
    const unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threads = gridDim.x * blockDim.x;
    const int lane    = threadIdx.x & 31;

    int new_theta = theta;    // visible: host sync after phase4b
    int win       = bb_window;

    for (int v = tid; v < N; v += threads) {
        unsigned int it = iteration_list[v];
        if (it & 0x40000000u) continue;
        unsigned int d = degree_list[v];
        if (d < (unsigned int)(new_theta + win)) {
            int physical = (int)d % win;
            int idx = atomicAdd(&bb_bucket_count[physical], 1);
            if (idx < bb_bucket_capacity) {
                bb_bucket_data[(size_t)physical * (size_t)N + idx] = v;
            } else {
                atomicSub(&bb_bucket_count[physical], 1);
                if (lane == 0)
                    printf("BB_split: bucket overflow in Phase 4c (slot %d)\n", physical);
            }
        }
    }
}

// CHROMA/BB.cu — Bucket-Based cuSL (BB-cuSL) PA kernel
// Spec: docs/superpowers/specs/2026-05-06-bb-cusl-design.md
//
// One cooperative kernel (P_SL_ELS_BB) runs the entire PA loop:
//   Phase 0: compute theta_init + initial bucket fill (gated by bb_init_done)
//   Phase 1: peel — read window buckets, lazy-validate, atomicCAS-claim
//   Phase 2: decrement neighbours, push to new bucket if in window
//   Phase 3: advance theta, reset emptied slots (overflow latch on emptied window)
//   Phase 4: overflow scan (rare fallback)

#include "globals.cuh"
#include <cuda.h>
#include <cooperative_groups.h>
#include <climits>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void P_SL_ELS_BB(
    const int  N,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threads = gridDim.x * blockDim.x;
    const int lane    = threadIdx.x & 31;
    const int warpId  = tid >> 5;
    const int numWarp = threads >> 5;
    cg::grid_group grid = cg::this_grid();

    // ───── Phase 0: Init theta + initial bucket fill (once per run) ─────
    if (bb_init_done == 0) {
        // 0.a Find theta_init = min(degree_list[v]) and zero iteration_list
        unsigned int local_min = UINT_MAX;
        for (int v = tid; v < N; v += threads) {
            iteration_list[v] = 0;
            unsigned int d = degree_list[v];
            if (d < local_min) local_min = d;
        }
        if (local_min < UINT_MAX) atomicMin(&g_minDegree, (int)local_min);
        grid.sync();

        if (grid.thread_rank() == 0) {
            theta = g_minDegree;
            atomicExch(&g_minDegree, INT_MAX);
        }
        grid.sync();

        // 0.b Fill in-window buckets
        int curr_theta = theta;
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
                    if (lane == 0) printf("BB: bucket overflow in Phase 0 (slot %d)\n", slot);
                }
            }
        }
        grid.sync();

        if (grid.thread_rank() == 0) bb_init_done = 1;
        grid.sync();
    }

    // ───── Main loop: Phases 1–4 ─────
    do {
        int curr_theta = theta;
        int window     = bb_window;
        int peel_iter  = bb_peel_iter;

        // ───── Phase 1: Peel window buckets ─────
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
        grid.sync();

        // ───── Phase 2: Decrement neighbours + push (warp-per-vertex) ─────
        int rs = remove_size;
        for (int k = warpId; k < rs; k += numWarp) {
            int v   = remove_list[k];
            int beg = nidx[v];          // each lane reads itself — coalesced
            int end = nidx[v + 1];

            for (int i = beg + lane; i < end; i += WS) {
                int u = nlist[i];
                unsigned int it_u = __ldg(&iteration_list[u]);
                if (it_u & 0x40000000u) continue;       // u already peeled

                unsigned int new_d = atomicSub(&degree_list[u], 1) - 1;

                if (new_d >= (unsigned int)curr_theta && new_d < (unsigned int)(curr_theta + window)) {
                    int physical = (int)new_d % window;
                    int idx = atomicAdd(&bb_bucket_count[physical], 1);
                    if (idx < bb_bucket_capacity) {
                        bb_bucket_data[(size_t)physical * (size_t)N + idx] = u;
                    } else {
                        atomicSub(&bb_bucket_count[physical], 1);
                        if (lane == 0) printf("BB: bucket overflow in Phase 2 (slot %d)\n", physical);
                    }
                }
            }
        }
        grid.sync();

        // ───── Phase 3: Advance theta + reset emptied slots ─────
        if (grid.thread_rank() == 0) {
            worker      += rs;
            remove_size  = 0;

            int new_theta = curr_theta;
            while (new_theta < curr_theta + window) {
                if (bb_bucket_count[new_theta % window] > 0) break;
                new_theta++;
            }

            if (new_theta >= curr_theta + window) {
                bb_overflow_needed = 1;
            } else {
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
        grid.sync();

        // ───── Phase 4: Overflow scan (fallback) ─────
        if (bb_overflow_needed) {
            unsigned int local_min = UINT_MAX;
            for (int v = tid; v < N; v += threads) {
                unsigned int it = iteration_list[v];
                if (!(it & 0x40000000u)) {
                    unsigned int d = degree_list[v];
                    if (d < local_min) local_min = d;
                }
            }
            if (local_min < UINT_MAX) atomicMin(&g_minDegree, (int)local_min);
            grid.sync();

            if (grid.thread_rank() == 0) {
                int win = bb_window;
                for (int s = 0; s < win; s++) bb_bucket_count[s] = 0;
                theta = g_minDegree;
                atomicExch(&g_minDegree, INT_MAX);
                bb_overflow_needed = 0;
            }
            grid.sync();

            int new_theta = theta;
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
                        if (lane == 0) printf("BB: bucket overflow in Phase 4 (slot %d)\n", physical);
                    }
                }
            }
            grid.sync();
        }

    } while (worker != N);
}

// CHROMA/PA_split.cu — Split-kernel diagnostic version of P_SL_ELS_SDC
// Allows per-phase profiling via nsys/ncu. Expected to be slower than the
// cooperative version due to per-iter launch + cudaMemcpyFromSymbol overhead.

#include "globals.cuh"
#include <cuda.h>
#include <cstdio>
#include <cub/cub.cuh>

// CTA block size (mirrors PA.cu — must match ThreadsPerBlock so cub::BlockScan
// is templated on the same compile-time constant as the launch configuration).
#define BLOCK_SIZE ThreadsPerBlock

struct node_buf_t {
    int node_id;
    int degree;
    int prefix_sum;
};

__global__ void P_SL_ELS_SDC_split_scan(
    const int N,
    const int* __restrict__ nidx,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threads = gridDim.x * blockDim.x;

    unsigned int localMin = 0x7FFFFFFFu;
    for (int v = tid; v < N; v += threads) {
        int iteration_list_v = iteration_list[v];
        unsigned int prio = (iteration_list_v >> 30) & 0x1u;
        unsigned int iteration_v = (unsigned int)iteration_list_v & 0x3FFFFFFFu;
        unsigned int large_deg = 0u;
        unsigned int degree = degree_list[v];
        if (prio == 0u) {
            if (degree <= (unsigned int)(theta + FuzzyNumber)) {
                prio = 1u;
                int beg = nidx[v];
                int end = nidx[v + 1];
                if ((end - beg) >= WS) large_deg = 1u;
                iteration_list[v] =
                    (large_deg << 31) | (prio << 30) |
                    (iteration_v + (degree - (unsigned int)theta) + 1u);
                remove_list[atomicAdd(&remove_size, 1)] = v;
            } else {
                if (degree < localMin) localMin = degree;
                iteration_list[v] =
                    (large_deg << 31) | (prio << 30) |
                    (iteration_v + (unsigned int)FuzzyNumber + 1u);
            }
        }
    }
    if (localMin < 0x7FFFFFFFu) atomicMin(&g_minDegree, (int)localMin);
}

__global__ void P_SL_ELS_SDC_split_decrement(
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list)
{
    const int lane    = threadIdx.x & 31;
    const int warpId  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int numWarp = (gridDim.x * blockDim.x) >> 5;

    const int rs = remove_size;

    for (int k = warpId; k < rs; k += numWarp) {
        int v = remove_list[k];
        int beg = nidx[v];          // each lane reads itself; same addr -> cached
        int end = nidx[v + 1];

        unsigned int warpMin = 0x7FFFFFFFu;
        for (int i = beg + lane; i < end; i += 32) {
            int nei = __ldg(nlist + i);
            unsigned int iter = __ldg(iteration_list + nei);
            if (!(iter & 0x40000000u)) {
                warpMin = min(warpMin, (unsigned int)(atomicSub(&degree_list[nei], 1) - 1));
            }
        }
        warpMin = warpReduceMin(warpMin);
        if (lane == 0 && warpMin < 0x7FFFFFFFu) {
            atomicMin(&g_minDegree, (int)warpMin);
        }
    }
}

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

// ──────────────────────────────────────────────────────────────────────────
// P_SL_ELS_SDC_CTA_S_split_decrement — Phase 2 of P_SL_ELS_SDC_CTA_S,
// isolated for per-phase profiling. Per-block dispatch:
//   remove_size <  cta_s_threshold : SDC warp-per-vertex path
//   remove_size >= cta_s_threshold : CTA-balanced removal
// η lives in the device global `cta_s_threshold` (host-set via --eta).
// ──────────────────────────────────────────────────────────────────────────

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
        use_sdc_path = (remove_size < cta_s_threshold) ? 1 : 0;
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

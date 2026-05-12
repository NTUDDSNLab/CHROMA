// CHROMA/PA_split.cu — Split-kernel diagnostic version of P_SL_ELS_SDC
// Allows per-phase profiling via nsys/ncu. Expected to be slower than the
// cooperative version due to per-iter launch + cudaMemcpyFromSymbol overhead.

#include "globals.cuh"
#include <cuda.h>
#include <cstdio>

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

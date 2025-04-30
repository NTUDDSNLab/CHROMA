// globals.cuh
#pragma once
#include <climits>   // for INT_MAX
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ── device 全域變數：只宣告，不初始化 ───────────────────
extern __device__ int  wlsize;
extern __device__ int* remove_list;
extern __device__ unsigned int* random_vals_d;
extern __device__ int  g_minDegree;
extern __device__ int  FuzzyNumber;
extern __device__ int  remove_size;
extern __device__ int  worker;
extern __device__ int  theta;
extern __device__ int  iteration;

// ── host-side 常數可以留在這裡（因為它們不是 __device__ 變數） ──
static const int Device          = 0;
static const int ThreadsPerBlock = 512;
static const unsigned int Warp   = 0xffffffff;
static const int WS              = 32;               // warp size
static const int MSB             = 1 << (WS - 1);
static const int Mask            = (1 << (WS / 2)) - 1;

// ── kernel prototype ───────────────────────────────────────
__global__ void  init(const int nodes, const int edges, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nlist2, int* const __restrict__ posscol, int* const __restrict__ posscol2, int* const __restrict__ color, int* const __restrict__ wl,unsigned int* const __restrict__ priority);

__global__ void  runLarge(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ posscol, int* const __restrict__ posscol2, volatile int* const __restrict__ color, const int* const __restrict__ wl);

__global__ void runSmall(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, volatile int* const __restrict__ posscol, int* const __restrict__ color);
__device__ __forceinline__
unsigned int warpReduceMin(unsigned int val)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__global__ void P_SL_WBR(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void P_SL_WBR_SDC(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);
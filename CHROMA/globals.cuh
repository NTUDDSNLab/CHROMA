// globals.cuh
#pragma once
#include <climits>   // for INT_MAX
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ── device global variables: declare only, do not initialize ───────────────────
extern __device__ int  wlsize;
extern __device__ int* remove_list;
extern __device__ unsigned int* random_vals_d;
extern __device__ int  g_minDegree;
extern __device__ int  FuzzyNumber;
extern __device__ int  remove_size;
extern __device__ int  worker;
extern __device__ int  theta;
extern __device__ int  iteration;
extern __device__ int  iter_count;

// ── CTA-balanced removal cursor (used by P_SL_ELS_SDC_CTA) ────────────────
extern __device__ int  cursor_remove;

// ── JP-Series PA globals (used by JP_ADG; declared here so the unified
//    pa_dumper binary can link CHROMA + JP-Series PA against shared globals) ─
extern __device__ int  avg_deg;
extern __device__ int  total_deg;
extern __device__ int  total_worker;

// ── BB-cuSL device globals (declared here, defined in globals.cu) ─────────
extern __device__ int   bb_window;            // = FuzzyNumber + 1, host-set
extern __device__ int   bb_bucket_capacity;   // = N, host-set
extern __device__ int*  bb_bucket_data;       // size = N * bb_window
extern __device__ int*  bb_bucket_count;      // size = bb_window
extern __device__ int   bb_init_done;         // 0 = need Phase 0; 1 = done
extern __device__ int   bb_overflow_needed;   // 0=none, 1=full Phase4, 2=refill-only Phase4
extern __device__ int   bb_peel_iter;         // 0-indexed outer iter counter

// Path A: sorted-S hint for global-min visibility
extern __device__ int*  bb_sorted_S;        // size N: vertex IDs sorted by initial degree (ascending)
extern __device__ int*  bb_sorted_degree;   // size N: initial degrees in sorted order (same order as bb_sorted_S)
extern __device__ int*  bb_initial_degree;  // size N: initial degree indexed by vertex id
extern __device__ int   bb_S_ptr;           // current scan position into bb_sorted_S

// ── host-side constants can stay here (since they are not __device__ variables) ──
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

__global__ void P_SL_ELS(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void P_SL_ELS_SDC(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void P_SL_ELS_SDC_CTA(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void P_SL_ELS_SDC_CTA_W(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void P_SL_ELS_SDC_CTA_S(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void P_SL_ELS_FUSED(
    const int  nodes,
    const int  edges,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    int* __restrict__ nlist2,
    int* __restrict__ posscol,
    int* __restrict__ posscol2,
    int* __restrict__ color,
    int* __restrict__ wl,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void P_SL_ELS_SDC_FUSED(
    const int  nodes,
    const int  edges,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    int* __restrict__ nlist2,
    int* __restrict__ posscol,
    int* __restrict__ posscol2,
    int* __restrict__ color,
    int* __restrict__ wl,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void P_SL_ELS_SDC_FUSED_BATCH(
    const int  nodes,
    const int  edges,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    int* __restrict__ nlist2,
    int* __restrict__ posscol,
    int* __restrict__ posscol2,
    int* __restrict__ color,
    int* __restrict__ wl,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void P_SL_ELS_BB(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

// ── BB-cuSL split-phase kernel prototypes (BB_split.cu) ───────────────────
__global__ void bb_split_phase0a_find_theta(
    int N,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void bb_split_phase0b_set_theta();

__global__ void bb_split_phase0c_fill_buckets(
    int N,
    const unsigned int* __restrict__ degree_list);

__global__ void bb_split_phase1_peel(
    int N,
    const int* __restrict__ nidx,
    const unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list,
    int peel_iter);

__global__ void bb_split_phase1_reset();

__global__ void bb_split_phase2_decrement(
    int N,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list);

__global__ void bb_split_phase3_advance(
    const unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list);

__global__ void bb_split_phase4a_scan(
    int N,
    const unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list);

__global__ void bb_split_phase4b_set_theta();

__global__ void bb_split_phase4c_refill(
    int N,
    const unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list);

__global__ void bb_split_phase4_reset_buckets();

// ── SDC-cuSL split-phase kernel prototypes (PA_split.cu) ──────────────────
__global__ void P_SL_ELS_SDC_split_scan(
    const int N,
    const int* __restrict__ nidx,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list);

__global__ void P_SL_ELS_SDC_split_decrement(
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list);

__global__ void P_SL_ELS_SDC_split_advance();

__global__ void P_SL_ELS_SDC_CTA_split_decrement(
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    const unsigned int* __restrict__ iteration_list);

// ── hash function (shared by PA fused + ECLGC init) ──────
static __device__ __forceinline__ unsigned int hash(unsigned int val)
{
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    val = ((val >> 16) ^ val) * 0x45d9f3b;
    return (val >> 16) ^ val;
}

#ifdef DYNAMIC_THETA
extern __device__ int   g_nodes;
extern __device__ int   last_remove_size;
extern __device__ int   CTRL_K;
extern __device__ float CTRL_RATE_THRESHOLD;
extern __device__ int   CTRL_STEP;
extern __device__ int   CTRL_CAP;

#define BUMP_LOG_MAX 32
extern __device__ int   bump_count;
extern __device__ int   bump_iter [BUMP_LOG_MAX];
extern __device__ int   bump_theta[BUMP_LOG_MAX];
#endif
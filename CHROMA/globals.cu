// globals.cu
#include "globals.cuh"

// This is where real allocation & initialization happens
__device__ int  wlsize      = 0;
__device__ int* remove_list = nullptr;
__device__ unsigned int* random_vals_d = nullptr;

__device__ int  g_minDegree = INT_MAX;
__device__ int  FuzzyNumber = 0;
__device__ int  remove_size = 0;
__device__ int  worker      = 0;
__device__ int  theta       = 1;
__device__ int  iteration   = 0;
__device__ int  iter_count  = 0;

// ── BB-cuSL device globals ────────────────────────────────────────────────
__device__ int   bb_window           = 0;
__device__ int   bb_bucket_capacity  = 0;
__device__ int*  bb_bucket_data      = nullptr;
__device__ int*  bb_bucket_count     = nullptr;
__device__ int   bb_init_done        = 0;
__device__ int   bb_overflow_needed  = 0;
__device__ int   bb_peel_iter        = 0;

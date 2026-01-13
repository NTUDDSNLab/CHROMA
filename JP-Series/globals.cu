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
__device__ int avg_deg      = 0;
__device__ int total_deg    = 0;
__device__ int total_worker = 0;

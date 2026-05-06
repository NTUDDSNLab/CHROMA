#include "chroma_utils.cuh"
#include "globals.cuh"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <stdio.h>

/* ----------------- allocAndInit ----------------- */
void allocAndInit(const ECLgraph& g, DevPtr& d, int fuzzy_number)
{
    cudaMalloc(&d.nidx_d,      (g.nodes + 1) * sizeof(int));
    cudaMalloc(&d.nlist_d,      g.edges        * sizeof(int));
    cudaMalloc(&d.nlist2_d,     g.edges        * sizeof(int));
    cudaMalloc(&d.posscol_d,    g.nodes        * sizeof(int));
    cudaMalloc(&d.posscol2_d,  (g.edges / WS + 1) * sizeof(int));
    cudaMalloc(&d.color_d,      g.nodes        * sizeof(int));
    cudaMalloc(&d.wl_d,         g.nodes        * sizeof(int));
    cudaMalloc(&d.degree_list,  g.nodes        * sizeof(unsigned int));
    cudaMalloc(&d.iteration_list_d, g.nodes    * sizeof(unsigned int));

    cudaMemcpy(d.nidx_d,  g.nindex, (g.nodes + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d.nlist_d, g.nlist,   g.edges * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemset(d.iteration_list_d, 0,
               g.nodes * sizeof(unsigned int));

    int* remove_list_ptr;
    cudaMalloc(&remove_list_ptr, g.nodes * sizeof(int));
    cudaMemcpyToSymbol(remove_list,
                       &remove_list_ptr, sizeof(int*));

    // Initialize nlist2 to -1
    cudaMemset(d.nlist2_d, -1, g.edges * sizeof(int));

    // ── BB-cuSL setup (memory + device-global init) ──────────────────────
    int bw_host = fuzzy_number + 1;
    if (bw_host < 1)  bw_host = 1;
    if (bw_host > 31) bw_host = 31;            // §10 R6 clamp
    cudaMemcpyToSymbol(bb_window, &bw_host, sizeof(int));

    int cap = g.nodes;
    cudaMemcpyToSymbol(bb_bucket_capacity, &cap, sizeof(int));

    int* bb_data_ptr = nullptr;
    int* bb_count_ptr = nullptr;
    cudaMalloc(&bb_data_ptr,  sizeof(int) * (size_t)g.nodes * (size_t)bw_host);
    cudaMalloc(&bb_count_ptr, sizeof(int) * (size_t)bw_host);
    cudaMemset(bb_count_ptr, 0, sizeof(int) * (size_t)bw_host);
    cudaMemcpyToSymbol(bb_bucket_data,  &bb_data_ptr,  sizeof(int*));
    cudaMemcpyToSymbol(bb_bucket_count, &bb_count_ptr, sizeof(int*));

    int zero = 0;
    cudaMemcpyToSymbol(bb_init_done,       &zero, sizeof(int));
    cudaMemcpyToSymbol(bb_overflow_needed, &zero, sizeof(int));
    cudaMemcpyToSymbol(bb_peel_iter,       &zero, sizeof(int));
}




/* ----------------- ECL_GC_run ------------------- */
void ECL_GC_run(int blocks, const ECLgraph& g, DevPtr& d)
{
    cudaFuncSetCacheConfig(init, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(runLarge, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(runSmall, cudaFuncCachePreferL1);
    
    cudaSetDevice(Device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);
    const int SMs = deviceProp.multiProcessorCount;
    int blkPerSM_GC;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM_GC, runLarge, ThreadsPerBlock, 0);
    int gridDim_GC = blkPerSM_GC * SMs;

    init<<<blocks, ThreadsPerBlock>>>(g.nodes, g.edges,
              d.nidx_d, d.nlist_d, d.nlist2_d,
              d.posscol_d, d.posscol2_d,
              d.color_d, d.wl_d,
              d.iteration_list_d);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error synchronizing device after init: %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // NOTE: use too many blocks may cause OCCUPANCY DEADLOCK in runLarge
    runLarge<<<gridDim_GC, ThreadsPerBlock>>>(g.nodes,
              d.nidx_d, d.nlist2_d,
              d.posscol_d, d.posscol2_d,
              d.color_d, d.wl_d);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error synchronizing device after runLarge: %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    runSmall<<<blocks, ThreadsPerBlock>>>(g.nodes,
              d.nidx_d, d.nlist_d,
              d.posscol_d,
              d.color_d);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error synchronizing device after runSmall: %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* ----------- ECL_GC_coloring_only --------------- */
void ECL_GC_coloring_only(int blocks, const ECLgraph& g, DevPtr& d)
{
    cudaFuncSetCacheConfig(runLarge, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(runSmall, cudaFuncCachePreferL1);

    cudaSetDevice(Device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);
    const int SMs = deviceProp.multiProcessorCount;
    int blkPerSM_GC;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM_GC, runLarge, ThreadsPerBlock, 0);
    int gridDim_GC = blkPerSM_GC * SMs;

    // NOTE: use too many blocks may cause OCCUPANCY DEADLOCK in runLarge
    runLarge<<<gridDim_GC, ThreadsPerBlock>>>(g.nodes,
              d.nidx_d, d.nlist2_d,
              d.posscol_d, d.posscol2_d,
              d.color_d, d.wl_d);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after runLarge: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    runSmall<<<blocks, ThreadsPerBlock>>>(g.nodes,
              d.nidx_d, d.nlist_d,
              d.posscol_d,
              d.color_d);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after runSmall: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* --------------- run_bb_split ------------------- */
void run_bb_split(int blocks, const ECLgraph& g, DevPtr& d)
{
    int N = g.nodes;

    // ── Phase 0: Init theta + initial bucket fill (once) ──────────────────
    bb_split_phase0a_find_theta<<<blocks, ThreadsPerBlock>>>(N, d.degree_list, d.iteration_list_d);
    cudaDeviceSynchronize();

    bb_split_phase0b_set_theta<<<1, 1>>>();
    cudaDeviceSynchronize();

    bb_split_phase0c_fill_buckets<<<blocks, ThreadsPerBlock>>>(N, d.degree_list);
    cudaDeviceSynchronize();

    // ── Main loop: Phases 1–4 ─────────────────────────────────────────────
    int worker_h = 0;
    int peel_iter = 0;

    while (worker_h != N) {
        // Phase 1: peel
        bb_split_phase1_peel<<<blocks, ThreadsPerBlock>>>(
            N, d.nidx_d, d.degree_list, d.iteration_list_d, peel_iter);
        cudaDeviceSynchronize();

        // Phase 1 reset: zero bucket counts for current window
        bb_split_phase1_reset<<<1, 32>>>();
        cudaDeviceSynchronize();

        // Phase 2: decrement neighbours + push
        bb_split_phase2_decrement<<<blocks, ThreadsPerBlock>>>(
            N, d.nidx_d, d.nlist_d, d.degree_list, d.iteration_list_d);
        cudaDeviceSynchronize();

        // Phase 3: advance theta, update worker/remove_size, set overflow latch
        bb_split_phase3_advance<<<1, 32>>>();
        cudaDeviceSynchronize();

        // Check overflow latch on host
        int overflow_h = 0;
        cudaMemcpyFromSymbol(&overflow_h, bb_overflow_needed, sizeof(int));
        if (overflow_h) {
            // Phase 4a: scan unpeeled, find new min degree
            bb_split_phase4a_scan<<<blocks, ThreadsPerBlock>>>(
                N, d.degree_list, d.iteration_list_d);
            cudaDeviceSynchronize();

            // Phase 4b: set theta = g_minDegree, reset counts
            bb_split_phase4b_set_theta<<<1, 32>>>();
            cudaDeviceSynchronize();

            // Phase 4c: refill buckets from unpeeled
            bb_split_phase4c_refill<<<blocks, ThreadsPerBlock>>>(
                N, d.degree_list, d.iteration_list_d);
            cudaDeviceSynchronize();
        }

        cudaMemcpyFromSymbol(&worker_h, worker, sizeof(int));
        peel_iter++;
    }
}

/* --------------- verify & stats ----------------- */
void verifyAndPrintStats(const ECLgraph& g,
                         const int* color,
                         float runtime)
{
    printf("runtime:    %.6f ms\n", runtime * 1000);

    for (int v = 0; v < g.nodes; v++) {
        if (color[v] < 0) {
            printf("ERROR: found unprocessed node (v=%d)\n", v);
            exit(-1);
        }
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
            if (color[g.nlist[i]] == color[v] && g.nlist[i] != v) {
                printf("ERROR: adjacent nodes share color %d (%d %d)\n",
                       color[v], v, g.nlist[i]);
                exit(-1);
            }
        }
    }
    printf("result verification passed\n");

    const int vals = 16;
    int c[vals] = {0};
    int cols = -1;
    for (int v = 0; v < g.nodes; v++) {
        cols = std::max(cols, color[v]);
        if (color[v] < vals) c[color[v]]++;
    }
    cols++;
    printf("colors used: %d\n", cols);
}

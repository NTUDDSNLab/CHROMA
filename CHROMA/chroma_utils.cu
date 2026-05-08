#include "chroma_utils.cuh"
#include "globals.cuh"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>


/* ----------------- resetForRun ----------------- *
 * Reset all per-run device state in place without re-allocating.
 * Used by `-r/--runs N` to dispatch N back-to-back PA+CA passes on the
 * same graph after a single allocAndInit, with each run starting from a
 * clean state.
 */
void resetForRun(const ECLgraph& g, DevPtr& d)
{
    // Reset two device-side arrays that PA / ECLGC-init initialise to a
    // sentinel before reading.
    cudaMemset(d.iteration_list_d, 0, g.nodes * sizeof(unsigned int));
    cudaMemset(d.nlist2_d, -1, g.edges * sizeof(int));

    // Reset every __device__ global the PA/CA path mutates back to its
    // start-of-binary value.
    int  zero = 0;
    int  one  = 1;
    int  imax = INT_MAX;
    cudaMemcpyToSymbol(g_minDegree,         &imax, sizeof(int));
    cudaMemcpyToSymbol(remove_size,         &zero, sizeof(int));
    cudaMemcpyToSymbol(worker,              &zero, sizeof(int));
    cudaMemcpyToSymbol(theta,               &one,  sizeof(int));
    cudaMemcpyToSymbol(iteration,           &zero, sizeof(int));
    cudaMemcpyToSymbol(iter_count,          &zero, sizeof(int));
    cudaMemcpyToSymbol(cursor_remove,       &zero, sizeof(int));
    cudaMemcpyToSymbol(wlsize,              &zero, sizeof(int));
    cudaMemcpyToSymbol(avg_deg,             &zero, sizeof(int));
    cudaMemcpyToSymbol(total_deg,           &zero, sizeof(int));
    cudaMemcpyToSymbol(total_worker,        &zero, sizeof(int));
    cudaMemcpyToSymbol(bb_init_done,        &zero, sizeof(int));
    cudaMemcpyToSymbol(bb_overflow_needed,  &zero, sizeof(int));
    cudaMemcpyToSymbol(bb_peel_iter,        &zero, sizeof(int));

    // Clear bb_bucket_count if BB was set up by allocAndInit. Capacity is
    // bb_window (= FuzzyNumber + 1, clamped to [1, 31]).
    int bw = 0;
    cudaMemcpyFromSymbol(&bw, bb_window, sizeof(int));
    if (bw > 0) {
        int* bb_count_ptr = nullptr;
        cudaMemcpyFromSymbol(&bb_count_ptr, bb_bucket_count, sizeof(int*));
        if (bb_count_ptr != nullptr) {
            cudaMemset(bb_count_ptr, 0, (size_t)bw * sizeof(int));
        }
    }
}

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
        bb_split_phase3_advance<<<1, 32>>>(d.degree_list, d.iteration_list_d);
        cudaDeviceSynchronize();

        // Check overflow latch on host
        int overflow_h = 0;
        cudaMemcpyFromSymbol(&overflow_h, bb_overflow_needed, sizeof(int));
        if (overflow_h == 1) {
            // Full Phase 4: scan unpeeled, find new min degree
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
        } else if (overflow_h == 2) {
            // Refill-only Phase 4 (Plan A): theta already set in Phase 3, skip O(N) scan
            bb_split_phase4_reset_buckets<<<1, 32>>>();
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

/* --------------- run_sdc_split ------------------- */
void run_sdc_split(int blocks, const ECLgraph& g, DevPtr& d)
{
    int N = g.nodes;
    int worker_h = 0;

    while (worker_h != N) {
        // Phase 1: scan all N vertices, peel those with degree <= theta+FuzzyNumber
        P_SL_ELS_SDC_split_scan<<<blocks, ThreadsPerBlock>>>(
            N, d.nidx_d, d.degree_list, d.iteration_list_d);
        cudaDeviceSynchronize();

        // Phase 2: decrement neighbours of peeled vertices, track new min degree
        P_SL_ELS_SDC_split_decrement<<<blocks, ThreadsPerBlock>>>(
            d.nidx_d, d.nlist_d, d.degree_list, d.iteration_list_d);
        cudaDeviceSynchronize();

        // Phase 3: advance worker, reset remove_size, update theta
        P_SL_ELS_SDC_split_advance<<<1, 32>>>();
        cudaDeviceSynchronize();

        cudaMemcpyFromSymbol(&worker_h, worker, sizeof(int));
    }
}

/* --------------- bb_setup_sorted_S -------------- */
void bb_setup_sorted_S(const ECLgraph& g, DevPtr& d)
{
    int N = g.nodes;

    int* sorted_S_ptr      = nullptr;
    int* sorted_degree_ptr = nullptr;
    int* initial_degree_ptr = nullptr;
    cudaMalloc(&sorted_S_ptr,        sizeof(int) * (size_t)N);
    cudaMalloc(&sorted_degree_ptr,   sizeof(int) * (size_t)N);
    cudaMalloc(&initial_degree_ptr,  sizeof(int) * (size_t)N);

    // initial_degree[v] = degree_list[v] (which was just filled by init_degree)
    cudaMemcpy(initial_degree_ptr, d.degree_list,
               sizeof(int) * (size_t)N, cudaMemcpyDeviceToDevice);

    // sorted_S[i] = i; sorted_degree[i] = degree[i] (will be sorted together)
    thrust::device_ptr<int> sorted_S_dev(sorted_S_ptr);
    thrust::sequence(sorted_S_dev, sorted_S_dev + N);

    // Use sorted_degree_ptr as the sort key (copy of initial degrees)
    cudaMemcpy(sorted_degree_ptr, initial_degree_ptr,
               sizeof(int) * (size_t)N, cudaMemcpyDeviceToDevice);

    // Sort both sorted_S[] and sorted_degree[] by degree ascending
    thrust::device_ptr<int> key_dev(sorted_degree_ptr);
    thrust::sort_by_key(key_dev, key_dev + N, sorted_S_dev);
    // After sort: sorted_degree[i] = i-th smallest initial degree (sequential access!)
    //             sorted_S[i]      = vertex id with i-th smallest initial degree

    // Publish to device symbols
    cudaMemcpyToSymbol(bb_sorted_S,       &sorted_S_ptr,       sizeof(int*));
    cudaMemcpyToSymbol(bb_sorted_degree,  &sorted_degree_ptr,  sizeof(int*));
    cudaMemcpyToSymbol(bb_initial_degree, &initial_degree_ptr, sizeof(int*));
    int zero = 0;
    cudaMemcpyToSymbol(bb_S_ptr, &zero, sizeof(int));
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

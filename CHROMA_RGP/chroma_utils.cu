#include "chroma_utils.cuh"
#include "globals.cuh"
#include <algorithm>
#include <cstdio>

/* ----------------- allocAndInit ----------------- */
void allocAndInit(const ECLgraph& g, DevPtr& d)
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
}

/* ----------------- ECL_GC_run ------------------- */
void ECL_GC_run(int blocks, const ECLgraph& g, DevPtr& d)
{
    init<<<blocks, ThreadsPerBlock>>>(g.nodes, g.edges,
              d.nidx_d, d.nlist_d, d.nlist2_d,
              d.posscol_d, d.posscol2_d,
              d.color_d, d.wl_d,
              d.iteration_list_d);

    runLarge<<<blocks, ThreadsPerBlock>>>(g.nodes,
              d.nidx_d, d.nlist2_d,
              d.posscol_d, d.posscol2_d,
              d.color_d, d.wl_d);

    runSmall<<<blocks, ThreadsPerBlock>>>(g.nodes,
              d.nidx_d, d.nlist_d,
              d.posscol_d,
              d.color_d);

    cudaDeviceSynchronize();
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

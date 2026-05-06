#pragma once
#include "ECLgraph.h"
#include "globals.cuh"

// GPU pointer container
struct DevPtr {
    int *nidx_d{}, *nlist_d{}, *nlist2_d{};
    int *posscol_d{}, *posscol2_d{}, *color_d{}, *wl_d{};
    unsigned int *degree_list{}, *iteration_list_d{};
};

struct ColorReductionStats {
    int colors_before{};
    int colors_after{};
    float runtime_sec{};
    bool attempted{};
    bool applied{};
};

/* Space allocation + Pre-initialization */
void allocAndInit(const ECLgraph& g, DevPtr& d, int fuzzy_number);

/* Three-step kernel wrapper (need to complete iteration_list with sl_allocate first) */
void ECL_GC_run(int blocks, const ECLgraph& g, DevPtr& d);

/* Coloring only (runLarge + runSmall) — used when init is fused into PA */
void ECL_GC_coloring_only(int blocks, const ECLgraph& g, DevPtr& d);

/* Post-process color reduction wrapper */
ColorReductionStats run_post_color_reduction(int blocks,
                                             const ECLgraph& g,
                                             DevPtr& d,
                                             int* color_host_buffer);

/* verify & Stats Output */
void verifyAndPrintStats(const ECLgraph& g,
                         const int* color,
                         float runtime);

/* BB-cuSL split-phase host driver (diagnostic; use for NCU per-phase profiling) */
void run_bb_split(int blocks, const ECLgraph& g, DevPtr& d);

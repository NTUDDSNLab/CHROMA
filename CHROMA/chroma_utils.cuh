#pragma once
#include "ECLgraph.h"
#include "globals.cuh"

// GPU pointer container
struct DevPtr {
    int *nidx_d{}, *nlist_d{}, *nlist2_d{};
    int *posscol_d{}, *posscol2_d{}, *color_d{}, *wl_d{};
    unsigned int *degree_list{}, *iteration_list_d{};
};

/* Space allocation + Pre-initialization */
void allocAndInit(const ECLgraph& g, DevPtr& d);

/* Three-step kernel wrapper (need to complete iteration_list with sl_allocate first) */
void ECL_GC_run(int blocks, const ECLgraph& g, DevPtr& d);

/* verify & Stats Output */
void verifyAndPrintStats(const ECLgraph& g,
                         const int* color,
                         float runtime);

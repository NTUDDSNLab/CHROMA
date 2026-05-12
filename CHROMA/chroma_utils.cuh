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

/* Reset per-run device state without re-allocating: clears iteration_list,
   nlist2, all PA/BB device globals; ready for another PA pass on the same
   graph. degree_list is NOT touched here — caller must re-launch
   init_degree<<<>>> before each run. */
void resetForRun(const ECLgraph& g, DevPtr& d);

/* Three-step kernel wrapper (need to complete iteration_list with sl_allocate first) */
void ECL_GC_run(int blocks, const ECLgraph& g, DevPtr& d);

/* Coloring only (runLarge + runSmall) — used when init is fused into PA */
void ECL_GC_coloring_only(int blocks, const ECLgraph& g, DevPtr& d);

/* Post-process color reduction wrapper.
 *   enabled = false → skip reduction entirely, just count colours.
 *   enabled = true  → dispatch heuristic 1 (avg_deg > 10) or heuristic 2
 *                     (avg_deg ≤ 10), matching ECL-GC's choice.
 *
 * Note: runtime_sec returned in ColorReductionStats is the time the
 * reduction kernels took (0 when enabled=false).
 */
ColorReductionStats run_post_color_reduction(int blocks,
                                             const ECLgraph& g,
                                             DevPtr& d,
                                             int* color_host_buffer,
                                             bool enabled = true);

/* verify & Stats Output */
void verifyAndPrintStats(const ECLgraph& g,
                         const int* color,
                         float runtime);

/* BB-cuSL split-phase host driver (diagnostic; use for NCU per-phase profiling) */
void run_bb_split(int blocks, const ECLgraph& g, DevPtr& d);

/* Path A: build sorted-by-initial-degree permutation for BB global-min hint */
void bb_setup_sorted_S(const ECLgraph& g, DevPtr& d);

/* Per-phase timing returned by SDC split-mode launchers (milliseconds,
 * summed across all outer iterations).  scan_ms covers every
 * P_SL_ELS_SDC_split_scan kernel launch; decrement_ms covers every
 * Phase-2 split-decrement kernel launch.  Excludes the advance kernel
 * and the cudaMemcpyFromSymbol(worker) round-trip. */
struct PaSplitStats {
    float scan_ms{};
    float decrement_ms{};
};

/* SDC-cuSL split-phase host driver (diagnostic; use for NCU per-phase profiling) */
PaSplitStats run_sdc_split(int blocks, const ECLgraph& g, DevPtr& d);

/* Set dynamic control parameters (DYNAMIC_THETA) */
#ifdef DYNAMIC_THETA
void setDynamicParameters(int nodes, int K, float rate, int step, int cap);
#endif

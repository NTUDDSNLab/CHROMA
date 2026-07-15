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

// CTA-balanced removal cursor (used by P_SL_ELS_SDC_CTA)
__device__ int  cursor_remove = 0;

// CTA_S dispatch threshold η: remove_size < η → SDC warp-per-vertex path,
// else CTA-balanced. Overridden from host via the --eta CLI flag.
__device__ int  cta_s_threshold = 4 * ThreadsPerBlock;   // = 2048

// JP-Series PA globals (used by JP_ADG; defined here so a unified pa_dumper
// can link CHROMA + JP-Series PA against shared globals).
__device__ int  avg_deg      = 0;
__device__ int  total_deg    = 0;
__device__ int  total_worker = 0;

// ── BB-cuSL device globals ────────────────────────────────────────────────
__device__ int   bb_window           = 0;
__device__ int   bb_bucket_capacity  = 0;
__device__ int*  bb_bucket_data      = nullptr;
__device__ int*  bb_bucket_count     = nullptr;
__device__ int   bb_init_done        = 0;
__device__ int   bb_overflow_needed  = 0;
__device__ int   bb_peel_iter        = 0;

// Path A: sorted-S hint for global-min visibility
__device__ int*  bb_sorted_S       = nullptr;
__device__ int*  bb_sorted_degree  = nullptr;
__device__ int*  bb_initial_degree = nullptr;
__device__ int   bb_S_ptr          = 0;

#ifdef DYNAMIC_THETA
// ─── Online θ controller state (compile-gated) ─────────────────────
// Number of vertices in graph (set once before kernel by setDynamicParameters)
__device__ int g_nodes = 0;

// Number of vertices already removed at the last controller checkpoint.
// Initialised to 0; updated every CTRL_K iterations.
__device__ int last_remove_size = 0;

// Tunables (set by setDynamicParameters before each kernel launch).
__device__ int   CTRL_K               = 0;       // 0 disables controller at runtime
__device__ float CTRL_RATE_THRESHOLD  = 0.0f;    // bump trigger (fraction of V removed per iter)
__device__ int   CTRL_STEP            = 1;       // amount to bump FuzzyNumber
__device__ int   CTRL_CAP             = 0;       // upper bound on FuzzyNumber (0 = no cap)

// Trajectory log (single-writer: block 0 thread 0, race-free).
#define BUMP_LOG_MAX 32
__device__ int bump_count = 0;
__device__ int bump_iter [BUMP_LOG_MAX] = {0};
__device__ int bump_theta[BUMP_LOG_MAX] = {0};
#endif

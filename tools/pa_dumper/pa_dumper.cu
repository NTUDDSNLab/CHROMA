// pa_dumper.cu — unified PA + CA driver for collecting priority orderings
// across all CHROMA / JP-Series / CPU SDL frameworks.
//
//   * One binary, one CLI, one dump format (uint32 priority per vertex).
//   * Always emits machine-readable `key=value` lines so the Python driver
//     can parse PA/CA timing and color count.
//   * Picks framework + algorithm via --algo; everything else is shared.
//
// Build:  see Makefile in this directory.
//
// Algorithms recognised:
//   CPU:        SDL                          (JP-SL^M heap-based, sequential)
//   CHROMA-PA:  cuSL_ELS, cuSL_ELS_SDC, cuSL_ELS_BB,
//               cuSL_ELS_FUSED, cuSL_ELS_SDC_FUSED,
//               cuSL_ELS_SDC_FUSED_BATCH
//   JP-Series:  JP_SL (= cuSL alias), JP_SLL, JP_ADG
//
// Note: passing --no-coloring skips the CA/colour-reduction phase, so we
// dump the priority and report only PA timing — useful when the only metric
// of interest is consistency vs another priority list.

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "ECLgraph.h"
#include "globals.cuh"
#include "chroma_utils.cuh"
#include "cpu_sdl.h"

// m2cgen-emitted predictor (model/model.cpp). Linked when -DPRED_MODEL is set
// in the Makefile. Returns predicted θ as a real number; we round to nearest int.
#ifdef PRED_MODEL
extern double score(double* input);
#endif

#include <cmath>

// JP-Series kernel prototypes (defined in jp_pa.cu)
__global__ void JP_SL(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list1,
    unsigned int* __restrict__ iteration_list);
__global__ void JP_SLL(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list1,
    unsigned int* __restrict__ iteration_list);
__global__ void JP_ADG(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list1,
    unsigned int* __restrict__ iteration_list);

__global__ static void setFuzzy(int v) { FuzzyNumber = v; }

__global__ static void init_degree(const int nodes,
                                   const int* __restrict__ nidx,
                                   unsigned int* __restrict__ degree_list)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int v = tid; v < nodes; v += blockDim.x * gridDim.x) {
        degree_list[v] = nidx[v + 1] - nidx[v];
    }
}

struct GPUTimer {
    cudaEvent_t beg, end;
    GPUTimer() { cudaEventCreate(&beg); cudaEventCreate(&end); }
    ~GPUTimer() { cudaEventDestroy(beg); cudaEventDestroy(end); }
    void start() { cudaEventRecord(beg, 0); }
    float stop_ms() {
        cudaEventRecord(end, 0); cudaEventSynchronize(end);
        float ms; cudaEventElapsedTime(&ms, beg, end); return ms;
    }
};

enum class Family { CPU, CHROMA, JP };

struct AlgoEntry {
    const char* name;
    Family      family;
    void*       kernel;     // null for CPU
    bool        is_fused;
};

// Resolve algo name → entry. Kernel pointers are filled in main() (host
// addresses of __global__ functions are not constexpr-friendly).
static AlgoEntry algo_table[] = {
    {"SDL",                       Family::CPU,    nullptr, false},
    {"cuSL_ELS",                  Family::CHROMA, nullptr, false},
    {"cuSL_ELS_SDC",              Family::CHROMA, nullptr, false},
    {"cuSL_ELS_SDC_CTA",          Family::CHROMA, nullptr, false},
    {"cuSL_ELS_SDC_CTA_W",        Family::CHROMA, nullptr, false},
    {"cuSL_ELS_SDC_CTA_S",        Family::CHROMA, nullptr, false},
    {"cuSL_ELS_BB",               Family::CHROMA, nullptr, false},
    {"cuSL_ELS_FUSED",            Family::CHROMA, nullptr, true },
    {"cuSL_ELS_SDC_FUSED",        Family::CHROMA, nullptr, true },
    {"cuSL_ELS_SDC_FUSED_BATCH",  Family::CHROMA, nullptr, true },
    {"JP_SL",                     Family::JP,     nullptr, false},
    {"JP_SLL",                    Family::JP,     nullptr, false},
    {"JP_ADG",                    Family::JP,     nullptr, false},
};

static constexpr int N_ALGO = sizeof(algo_table) / sizeof(AlgoEntry);

static const char* family_name(Family f) {
    switch (f) {
        case Family::CPU:    return "CPU";
        case Family::CHROMA: return "CHROMA";
        case Family::JP:     return "JP-Series";
    }
    return "?";
}

static void resolve_kernels() {
    int i = 0;
    algo_table[i++].kernel = nullptr;                         // SDL
    algo_table[i++].kernel = (void*)P_SL_ELS;
    algo_table[i++].kernel = (void*)P_SL_ELS_SDC;
    algo_table[i++].kernel = (void*)P_SL_ELS_SDC_CTA;
    algo_table[i++].kernel = (void*)P_SL_ELS_SDC_CTA_W;
    algo_table[i++].kernel = (void*)P_SL_ELS_SDC_CTA_S;
    algo_table[i++].kernel = (void*)P_SL_ELS_BB;
    algo_table[i++].kernel = (void*)P_SL_ELS_FUSED;
    algo_table[i++].kernel = (void*)P_SL_ELS_SDC_FUSED;
    algo_table[i++].kernel = (void*)P_SL_ELS_SDC_FUSED_BATCH;
    algo_table[i++].kernel = (void*)JP_SL;
    algo_table[i++].kernel = (void*)JP_SLL;
    algo_table[i++].kernel = (void*)JP_ADG;
}

static void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " -f <graph.egr> -a <algo> [options]\n\n";
    std::cout << "Algorithms:\n";
    std::cout << "  CPU:       SDL\n";
    std::cout << "  CHROMA-PA: cuSL_ELS, cuSL_ELS_SDC, cuSL_ELS_SDC_CTA, cuSL_ELS_BB,\n";
    std::cout << "             cuSL_ELS_FUSED, cuSL_ELS_SDC_FUSED, cuSL_ELS_SDC_FUSED_BATCH\n";
    std::cout << "  JP-Series: JP_SL, JP_SLL, JP_ADG\n\n";
    std::cout << "Options:\n";
    std::cout << "  -f, --file <path>          Input .egr graph (required)\n";
    std::cout << "  -a, --algo <name>          Algorithm (required)\n";
    std::cout << "  -e, --theta <int>          Elastic / FuzzyNumber (default: 0)\n";
    std::cout << "  -p, --predict              Set θ from learned predictor\n"
                 "                             (requires PRED_MODEL=1 build)\n";
    std::cout << "      --dump-priority <p>    Write 30-bit priority dump to FILE\n";
    std::cout << "      --no-coloring          Skip CA + color-reduction\n";
    std::cout << "  -h, --help                 Show this help\n";
}

static const AlgoEntry* find_algo(const std::string& name) {
    for (int i = 0; i < N_ALGO; ++i)
        if (name == algo_table[i].name) return &algo_table[i];
    return nullptr;
}

// ── Run a CPU SDL pipeline (PA + greedy color) ─────────────────────────────
static int run_cpu_sdl(const ECLgraph& g,
                       std::vector<unsigned int>& priority_out,
                       int& colors_used,
                       float& pa_ms, float& ca_ms,
                       bool do_color)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    cpu_sdl_peel(g, priority_out);
    auto t1 = clk::now();
    pa_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (do_color) {
        std::vector<int> color;
        colors_used = cpu_sdl_greedy_color(g, priority_out, color);
        auto t2 = clk::now();
        ca_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();
        // Verify
        for (int v = 0; v < g.nodes; ++v) {
            for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
                int u = g.nlist[e];
                if (u != v && color[v] == color[u]) {
                    fprintf(stderr,
                            "ERROR: adjacent nodes share color %d (%d %d)\n",
                            color[v], v, u);
                    return -1;
                }
            }
        }
    } else {
        colors_used = -1;
        ca_ms = 0.0f;
    }
    return 0;
}

// ── Launch a GPU PA kernel + (optionally) CA + reduction ────────────────────
static int run_gpu(const AlgoEntry& algo,
                   const ECLgraph& g,
                   int fuzzy_number,
                   std::vector<unsigned int>& priority_out,
                   int& colors_before,
                   int& colors_after,
                   float& pa_ms, float& ca_ms, float& reduce_ms,
                   bool do_color)
{
    // GPU info
    cudaSetDevice(Device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, Device);
    const int SMs    = prop.multiProcessorCount;
    const int mTpSM  = prop.maxThreadsPerMultiProcessor;
    const int blocks = SMs * mTpSM / ThreadsPerBlock;

    DevPtr d;
    allocAndInit(g, d, fuzzy_number);
    setFuzzy<<<1, 1>>>(fuzzy_number);
    cudaDeviceSynchronize();

    GPUTimer pa_timer;
    pa_timer.start();

    init_degree<<<(g.nodes + 511) / 512, 512>>>(g.nodes, d.nidx_d, d.degree_list);
    cudaDeviceSynchronize();

    int blkPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM,
        algo.kernel, ThreadsPerBlock, 0);
    int gridDim = blkPerSM * SMs;

    cudaError_t launch_err;
    if (algo.is_fused) {
        int edges_val = g.edges;
        void* args[] = { (void*)&g.nodes, (void*)&edges_val,
                         (void*)&d.nidx_d, (void*)&d.nlist_d,
                         (void*)&d.nlist2_d, (void*)&d.posscol_d,
                         (void*)&d.posscol2_d, (void*)&d.color_d,
                         (void*)&d.wl_d, (void*)&d.degree_list,
                         (void*)&d.iteration_list_d };
        launch_err = cudaLaunchCooperativeKernel(algo.kernel,
            dim3(gridDim), dim3(ThreadsPerBlock), args);
    } else {
        void* args[] = { (void*)&g.nodes, (void*)&d.nidx_d, (void*)&d.nlist_d,
                         (void*)&d.degree_list, (void*)&d.iteration_list_d };
        launch_err = cudaLaunchCooperativeKernel(algo.kernel,
            dim3(gridDim), dim3(ThreadsPerBlock), args);
    }
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[pa_dumper] kernel launch failed: %s\n",
                cudaGetErrorString(launch_err));
    }
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[pa_dumper] sync after PA failed: %s\n",
                cudaGetErrorString(sync_err));
    }
    pa_ms = pa_timer.stop_ms();

    // Copy iteration_list out as priority (mask off prio/large_deg bits)
    priority_out.assign(g.nodes, 0);
    cudaMemcpy(priority_out.data(), d.iteration_list_d,
               g.nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (int v = 0; v < g.nodes; ++v) priority_out[v] &= 0x3FFFFFFFu;

    if (do_color) {
        GPUTimer ca_timer;
        ca_timer.start();
        if (algo.is_fused) ECL_GC_coloring_only(blocks, g, d);
        else               ECL_GC_run(blocks, g, d);
        ca_ms = ca_timer.stop_ms();

        std::vector<int> color(g.nodes);
        ColorReductionStats rs =
            run_post_color_reduction(blocks, g, d, color.data());
        colors_before = rs.colors_before;
        colors_after  = rs.colors_after;
        reduce_ms     = rs.runtime_sec * 1000.0f;

        // Verify
        for (int v = 0; v < g.nodes; ++v) {
            for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
                int u = g.nlist[e];
                if (u != v && color[v] == color[u]) {
                    fprintf(stderr,
                            "ERROR: adjacent nodes share color %d (%d %d)\n",
                            color[v], v, u);
                    cudaFree(d.wl_d); cudaFree(d.color_d);
                    cudaFree(d.posscol2_d); cudaFree(d.posscol_d);
                    cudaFree(d.nlist2_d); cudaFree(d.nlist_d);
                    cudaFree(d.nidx_d);
                    return -1;
                }
            }
        }
    } else {
        ca_ms = 0.0f;
        reduce_ms = 0.0f;
        colors_before = -1;
        colors_after  = -1;
    }

    cudaFree(d.wl_d); cudaFree(d.color_d);
    cudaFree(d.posscol2_d); cudaFree(d.posscol_d);
    cudaFree(d.nlist2_d); cudaFree(d.nlist_d);
    cudaFree(d.nidx_d);
    cudaFree(d.degree_list); cudaFree(d.iteration_list_d);
    return 0;
}

int main(int argc, char** argv)
{
    resolve_kernels();

    std::string graph_path, algo_name, dump_path;
    int  fuzzy_number  = 0;
    bool do_color      = true;
    bool use_predict   = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_help(argv[0]); return 0; }
        else if ((a == "-f" || a == "--file") && i + 1 < argc) graph_path = argv[++i];
        else if ((a == "-a" || a == "--algo") && i + 1 < argc) algo_name  = argv[++i];
        else if ((a == "-e" || a == "--theta") && i + 1 < argc) {
            fuzzy_number = std::atoi(argv[++i]);
            use_predict = false;
        }
        else if (a == "-p" || a == "--predict") use_predict = true;
        else if (a == "--dump-priority" && i + 1 < argc) dump_path = argv[++i];
        else if (a == "--no-coloring") do_color = false;
        else { fprintf(stderr, "Unknown/incomplete option: %s\n", a.c_str());
               print_help(argv[0]); return 1; }
    }
    if (graph_path.empty() || algo_name.empty()) {
        fprintf(stderr, "ERROR: -f and -a are required\n");
        print_help(argv[0]); return 1;
    }
    const AlgoEntry* algo = find_algo(algo_name);
    if (!algo) {
        fprintf(stderr, "ERROR: unknown algorithm '%s'\n", algo_name.c_str());
        return 1;
    }

    ECLgraph g = readECLgraph(graph_path.c_str());

    if (use_predict) {
#ifdef PRED_MODEL
        double input[2] = { (double)g.nodes, (double)g.edges };
        fuzzy_number = (int)std::round(score(input));
        if (fuzzy_number < 0) fuzzy_number = 0;
#else
        fprintf(stderr, "ERROR: --predict requires PRED_MODEL=1 build\n");
        freeECLgraph(g);
        return 3;
#endif
    }

    std::vector<unsigned int> priority;
    int colors_used = -1, colors_before = -1, colors_after = -1;
    float pa_ms = 0, ca_ms = 0, reduce_ms = 0;
    int rc;
    if (algo->family == Family::CPU) {
        rc = run_cpu_sdl(g, priority, colors_used, pa_ms, ca_ms, do_color);
        colors_before = colors_after = colors_used;
    } else {
        rc = run_gpu(*algo, g, fuzzy_number, priority,
                     colors_before, colors_after,
                     pa_ms, ca_ms, reduce_ms, do_color);
        colors_used = colors_after;
    }

    if (rc != 0) { freeECLgraph(g); return 2; }

    if (!dump_path.empty()) {
        FILE* fp = fopen(dump_path.c_str(), "wb");
        if (!fp) {
            fprintf(stderr, "ERROR: cannot open %s for write\n", dump_path.c_str());
        } else {
            size_t w = fwrite(priority.data(), sizeof(unsigned int),
                              priority.size(), fp);
            fclose(fp);
            if (w != priority.size())
                fprintf(stderr, "WARN: short write on %s (%zu/%zu)\n",
                        dump_path.c_str(), w, priority.size());
        }
    }

    // Machine-readable result block (parsed by Python driver)
    printf("---PA_DUMPER_RESULT---\n");
    printf("framework=%s\n", family_name(algo->family));
    printf("algo=%s\n", algo_name.c_str());
    printf("theta=%d\n", fuzzy_number);
    printf("predicted_theta=%s\n", use_predict ? "1" : "0");
    printf("graph=%s\n", graph_path.c_str());
    printf("nodes=%d\n", g.nodes);
    printf("edges=%d\n", g.edges);
    printf("pa_ms=%.6f\n", pa_ms);
    printf("ca_ms=%.6f\n", ca_ms);
    printf("reduce_ms=%.6f\n", reduce_ms);
    printf("total_ms=%.6f\n", pa_ms + ca_ms + reduce_ms);
    printf("colors_used=%d\n", colors_used);
    printf("colors_before_reduction=%d\n", colors_before);
    printf("colors_after_reduction=%d\n", colors_after);
    printf("dumped_priority=%s\n",
           dump_path.empty() ? "none" : dump_path.c_str());
    printf("verification=%s\n", do_color ? "passed" : "skipped");
    printf("---PA_DUMPER_END---\n");

    freeECLgraph(g);
    return 0;
}

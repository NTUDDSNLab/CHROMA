#include <cuda.h>
#include <random>
#include "ECLgraph.h"
#include "globals.cuh"
#include <pthread.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
// #include <cooperative_groups.h>
#include <iostream>
#include <string>
#include <algorithm>
#include "chroma_utils.cuh"
#include <cstring>
#include "graph.h"
#include "graph_features.h"
#if defined(PRED_MODEL) && defined(HAVE_V2_MODEL)
#include "scaler.h"
#endif

// namespace cg = cooperative_groups;

#ifdef PRED_MODEL
extern double score(double *input);
#endif


enum GraphFormat {
    ECLGraph,
    SNAPGraph,
    MMIOGraph
};


// Add helper function
void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -f, --file <path>         Input graph file path (required)\n";
    std::cout << "                            Supported formats:\n";
    std::cout << "                            .egr  : ECL graph format (binary)\n";
    std::cout << "                            .txt  : Text format (CSR graph)\n";
    std::cout << "                            .bin  : Binary format (CSR graph)\n";
    std::cout << "  -a, --algorithm <algo>    Select algorithm:\n";
    std::cout << "                            0 or cuSL_ELS       : cuSL_ELS algorithm\n";
    std::cout << "                            1 or cuSL_ELS_SDC   : cuSL_ELS_SDC algorithm\n";
    std::cout << "                            2 or cuSL_ELS_FUSED : PA+Init post-loop fused\n";
    std::cout << "                            3 or cuSL_ELS_SDC_FUSED : SDC PA+Init post-loop fused\n";
    std::cout << "                            4 or cuSL_ELS_SDC_FUSED_BATCH : SDC PA+Init per-batch fused\n";
    std::cout << "                            5 or cuSL_ELS_BB    : sliding-window bucket-based PA\n";
    std::cout << "                            6 or cuSL_ELS_BB_SPLIT : per-phase split-kernel diagnostic variant (NCU profiling)\n";
    std::cout << "                            7 or cuSL_ELS_SDC_SPLIT : SDC per-phase split-kernel diagnostic variant (NCU profiling)\n";
    std::cout << "                            8 or cuSL_ELS_SDC_CTA   : SDC with CTA-balanced removal (cub::BlockScan + binary search)\n";
    std::cout << "                            9 or cuSL_ELS_SDC_CTA_W : SDC + dynamic dispatch (warp-balanced for small remove_size, CTA otherwise)\n";
    std::cout << "                           10 or cuSL_ELS_SDC_CTA_S : SDC + dynamic dispatch (SDC warp-per-vertex for small remove_size, CTA otherwise) [recommended]\n";
    std::cout << "                            (default: cuSL_ELS)\n";
    std::cout << "  -e, --elastic <number>    Set elastic number θ value (default: 0)\n";
    std::cout << "  -p, --predict             Use prediction model for elastic parameter\n";
    std::cout << "      --v2-model            Use 7-feature v2 predictor (requires PRE_MODEL=1 V2_MODEL=1 build)\n";
    std::cout << "  -r, --runs <int>          Repeat PA+CA N times on the same loaded\n"
                 "                            graph and print per-run timings + final\n"
                 "                            avg/min/max summary (default: 1)\n";
    std::cout << "  --reduce / --no-reduce    Run / skip post-CA color reduction\n"
                 "                            (default: --reduce). When enabled,\n"
                 "                            dispatches heuristic 1 if avg_deg > 10\n"
                 "                            else heuristic 2 (matches ECL-GC).\n";
    std::cout << "  --dump-priority <path>    Dump PA priority list (uint32 per vertex) to FILE\n";
    std::cout << "  -h, --help                Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " -f graph.txt\n";
    std::cout << "  " << program_name << " --file graph.egr -a cuSL_ELS_SDC\n";
    std::cout << "  " << program_name << " -f graph.bin --algorithm 1 --elastic 15\n";
    std::cout << "  " << program_name << " -f graph.egr --predict\n";
}

// Parse algorithm parameters
void* select_algorithm(const std::string& algo_str, std::string& algo_name, bool& is_fused) {
    is_fused = false;
    if (algo_str == "0" || algo_str == "cuSL_ELS") {
        algo_name = "cuSL_ELS";
        return (void*)P_SL_ELS;
    } else if (algo_str == "1" || algo_str == "cuSL_ELS_SDC") {
        algo_name = "cuSL_ELS_SDC";
        return (void*)P_SL_ELS_SDC;
    } else if (algo_str == "2" || algo_str == "cuSL_ELS_FUSED") {
        algo_name = "cuSL_ELS_FUSED";
        is_fused = true;
        return (void*)P_SL_ELS_FUSED;
    } else if (algo_str == "3" || algo_str == "cuSL_ELS_SDC_FUSED") {
        algo_name = "cuSL_ELS_SDC_FUSED";
        is_fused = true;
        return (void*)P_SL_ELS_SDC_FUSED;
    } else if (algo_str == "4" || algo_str == "cuSL_ELS_SDC_FUSED_BATCH") {
        algo_name = "cuSL_ELS_SDC_FUSED_BATCH";
        is_fused = true;
        return (void*)P_SL_ELS_SDC_FUSED_BATCH;
    } else if (algo_str == "5" || algo_str == "cuSL_ELS_BB") {
        algo_name = "cuSL_ELS_BB";
        return (void*)P_SL_ELS_BB;
    } else if (algo_str == "6" || algo_str == "cuSL_ELS_BB_SPLIT") {
        algo_name = "cuSL_ELS_BB_SPLIT";
        return (void*)P_SL_ELS_BB;  // placeholder pointer; not actually launched via cooperative kernel
    } else if (algo_str == "7" || algo_str == "cuSL_ELS_SDC_SPLIT") {
        algo_name = "cuSL_ELS_SDC_SPLIT";
        return (void*)P_SL_ELS_SDC;  // placeholder; main flow detects by name
    } else if (algo_str == "8" || algo_str == "cuSL_ELS_SDC_CTA") {
        algo_name = "cuSL_ELS_SDC_CTA";
        return (void*)P_SL_ELS_SDC_CTA;
    } else if (algo_str == "9" || algo_str == "cuSL_ELS_SDC_CTA_W") {
        algo_name = "cuSL_ELS_SDC_CTA_W";
        return (void*)P_SL_ELS_SDC_CTA_W;
    } else if (algo_str == "10" || algo_str == "cuSL_ELS_SDC_CTA_S") {
        algo_name = "cuSL_ELS_SDC_CTA_S";
        return (void*)P_SL_ELS_SDC_CTA_S;
    } else {
        std::cerr << "Error: Invalid algorithm '" << algo_str << "'. Using default cuSL_ELS.\n";
        algo_name = "cuSL_ELS";
        return (void*)P_SL_ELS;
    }
}

__global__ void setParameters(int fuzzy_number) {
  FuzzyNumber = fuzzy_number;
}

static __global__
void init_degree(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, unsigned int* const __restrict__ degree_list) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int vertex_id = thread_id; vertex_id < nodes; vertex_id += blockDim.x * gridDim.x) {
        int deg = nidx[vertex_id + 1] - nidx[vertex_id];
        if (deg < 0) printf("Degree is negative: %d\n", deg);
        degree_list[vertex_id] = deg;
    }
}
void init_random_values(int nodes) {
  unsigned int* random_vals = new unsigned int[nodes];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> dist;
  for (int i = 0; i < nodes; ++i) {
    random_vals[i] = dist(gen);
  }

  unsigned int* random_vals_device;
  cudaMalloc((void**)&random_vals_device, nodes * sizeof(unsigned int));
  cudaMemcpy(random_vals_device, random_vals, nodes * sizeof(unsigned int), cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(random_vals_d, &random_vals_device, sizeof(unsigned int*)); // Use cudaMemcpyToSymbol to copy pointer
  
  delete[] random_vals;
}



struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg);  cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg);  cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  float stop() {cudaEventRecord(end, 0);  cudaEventSynchronize(end);  float ms;  cudaEventElapsedTime(&ms, beg, end);  return 0.001f * ms;}
};


int main(int argc, char* argv[])
{
    // Default settings
    std::string filename;
    int fuzzy_number = 0;  // EGC θ default value is 0
    void* kernel_to_launch = (void*)P_SL_ELS;
    std::string algo_name = "cuSL_ELS";
    bool is_fused = false;
    bool use_predicted_elastic = false;  // Mark whether to use predicted elastic value
    bool use_v2_model = false;
    std::string dump_priority_path;
    int  num_runs = 1;
    bool enable_reduce = true;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) {
            if (i + 1 < argc) {
                filename = argv[++i];
            } else {
                std::cerr << "Error: File option requires an argument.\n";
                print_help(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--algorithm") == 0) {
            if (i + 1 < argc) {
                kernel_to_launch = select_algorithm(argv[++i], algo_name, is_fused);
            } else {
                std::cerr << "Error: Algorithm option requires an argument.\n";
                print_help(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--elastic") == 0) {
            if (i + 1 < argc) {
                try {
                    fuzzy_number = std::stoi(argv[++i]);
                    use_predicted_elastic = false;  // Do not use prediction when manually specified
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid EGC value '" << argv[i] << "'.\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: EGC option requires an argument.\n";
                print_help(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--predict") == 0) {
            use_predicted_elastic = true;
        } else if (strcmp(argv[i], "--v2-model") == 0) {
            use_v2_model = true;
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--runs") == 0) {
            if (i + 1 < argc) {
                try {
                    num_runs = std::stoi(argv[++i]);
                    if (num_runs < 1) {
                        std::cerr << "Error: --runs must be >= 1.\n";
                        return 1;
                    }
                } catch (const std::exception&) {
                    std::cerr << "Error: Invalid --runs value '" << argv[i] << "'.\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --runs option requires an integer.\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--dump-priority") == 0) {
            if (i + 1 < argc) {
                dump_priority_path = argv[++i];
            } else {
                std::cerr << "Error: --dump-priority requires a path argument.\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--reduce") == 0) {
            enable_reduce = true;
        } else if (strcmp(argv[i], "--no-reduce") == 0) {
            enable_reduce = false;
        } else if (argv[i][0] == '-') {
            std::cerr << "Error: Unknown option '" << argv[i] << "'.\n";
            print_help(argv[0]);
            return 1;
        } else {
            std::cerr << "Error: Unexpected argument '" << argv[i] << "'. Use -f to specify input file.\n";
            print_help(argv[0]);
            return 1;
        }
    }

    // Check if graph file is provided
    if (filename.empty()) {
        std::cerr << "Error: No graph file specified. Use -f or --file to specify input file.\n";
        print_help(argv[0]);
        return 1;
    }

    // Read graph file
    ECLgraph g;
    CSRGraph graph;
    
    // Determine format based on file extension
    std::string file_extension = filename.substr(filename.find_last_of(".") + 1);
    
    
    if (file_extension == "egr") {
        // Use ECLgraph to read .egr file
        g = readECLgraph(filename.c_str());
        // Create an empty CSRGraph for compatibility with subsequent code
        graph.file_num_nodes = g.nodes;
        graph.file_num_edges = g.edges;
        graph.max_node_id = g.nodes - 1;
        std::cout << "Read .egr file using ECLgraph" << std::endl;
    } else if (file_extension == "txt" || file_extension == "bin") {
        std::string binary_filename = filename.substr(0, filename.find_last_of(".")) + ".bin";
        
        bool loaded_from_binary = false;
        if (file_extension == "bin") {
             if (graph.loadFromBinary(filename)) {
                loaded_from_binary = true;
             }
        } else {
            // Check if binary file exists
            FILE* f = fopen(binary_filename.c_str(), "rb");
            if (f != NULL) {
                fclose(f);
                std::cout << "Found binary cache: " << binary_filename << ", loading..." << std::endl;
                if (graph.loadFromBinary(binary_filename)) {
                    loaded_from_binary = true;
                }
            }
        }

        if (!loaded_from_binary) {
             std::cout << "Binary cache not found or failed to load. Reading from text file..." << std::endl;
             graph.buildFromTxtFile(filename, false);
             
             std::cout << "Saving to binary file: " << binary_filename << std::endl;
             graph.saveToBinary(binary_filename);
             
             // Verify consistency
             std::cout << "Verifying binary file consistency..." << std::endl;
             CSRGraph check_graph;
             if (check_graph.loadFromBinary(binary_filename)) {
                 if (graph == check_graph) {
                     std::cout << "Verification SUCCESS: Binary file matches original graph." << std::endl;
                 } else {
                     std::cerr << "Verification FAILED: Binary file does not match original graph!" << std::endl;
                 }
             } else {
                 std::cerr << "Verification FAILED: Could not load generated binary file!" << std::endl;
             }
        }

        std::cout << "build from " << (loaded_from_binary ? "binary" : "text") << " file" << std::endl;
        std::cout << filename << " is zero based: " << graph.is_zero_based << std::endl;
        // Transform to ECLgraph
        graph.transfromToECLGraph(g);
    } else {
        std::cerr << "Error: Unsupported file format '" << file_extension << "'. Supported formats: .egr, .txt, .bin" << std::endl;
        return 1;
    }
    
    // If prediction model is enabled, use score function to predict elastic parameter
    #ifdef PRED_MODEL
    if (use_predicted_elastic) {
        double score_result;
    #ifdef HAVE_V2_MODEL
        if (use_v2_model) {
            GraphFeatures f = compute_graph_features(g);
            double input[chroma_predictor::FEATURE_COUNT];
            for (int i = 0; i < chroma_predictor::FEATURE_COUNT; ++i) {
                input[i] = (f.as_array[i] - chroma_predictor::SCALER_MEAN[i])
                         /  chroma_predictor::SCALER_STD[i];
            }
            score_result = score(input);
        } else
    #endif
        {
            // Legacy 2-feature predictor: V, E only
            double input[2] = {(double)g.nodes, (double)g.edges};
            score_result = score(input);
        }
        fuzzy_number = (int)round(score_result);
        if (fuzzy_number < 0) fuzzy_number = 0;
    }
    #else
    if (use_predicted_elastic) {
        std::cerr << "Error: Prediction model is not compiled. Please compile with PRED_MODEL flag.\n";
        return 1;
    }
    if (use_v2_model) {
        std::cerr << "Warning: --v2-model has no effect without --predict.\n";
    }
    #endif

    // Display setting information
    printf("Input file: %s\n", filename.c_str());
    printf("Algorithm: %s\n", algo_name.c_str());
    if (use_predicted_elastic) {
        printf("EGC θ: %d (Predicted)\n", fuzzy_number);
    } else {
        printf("EGC θ: %d\n", fuzzy_number);
    }
    printf("File num nodes: %d\n", graph.file_num_nodes);
    printf("File num edges: %d\n", graph.file_num_edges);
    printf("Nodes: %d\n", g.nodes);
    printf("Edges: %d\n", g.edges);
    printf("Max node id: %d\n", graph.max_node_id);
    printf("Average degree: %.2f\n", 1.0 * g.edges / g.nodes);
    
    
    setParameters<<<1, 1>>>(fuzzy_number);

    
    // GPU INFO
    cudaSetDevice(Device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);
    const int SMs = deviceProp.multiProcessorCount;
    const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
    const int blocks = SMs * mTpSM / ThreadsPerBlock;
    int* const color = new int [g.nodes];



    DevPtr d;
    allocAndInit(g, d, fuzzy_number);

    // Analyze the degree distribution of the graph
    int max_degree = 0;
    int max_degree_node = 0;
    unsigned int sum_degree = 0;
    for (int i = 0; i < g.nodes; i++) {
        int deg = g.nindex[i+1] - g.nindex[i];
        sum_degree += deg;
        if (deg > max_degree) {
            max_degree = deg;
            max_degree_node = i;
        }
        if (deg < 0) printf("[%d]'s Degree is negative: %d\n", i, deg);
    }
    printf("Max degree: %d, Max degree node: %d\n", max_degree, max_degree_node);

    printf("----- [GPU INFO] -----\n");
    printf("SMs: %d\n", SMs);
    printf("mTpSM: %d\n", mTpSM);
    printf("blocks: %d\n", blocks);
    printf("----- [GPU INFO] -----\n");

    // Check if sum_degree is 2*edges
    // if (sum_degree != 2*g.edges) {
    //     printf("Sum of degree is not 2*edges: %d %d\n", sum_degree, 2*g.edges);
    //     exit(-1);
    // }

    // Finding if there is 0 id node
    // for (int v = 0; v < g.edges; v++) {
    //     if (g.nlist[v] == 0) {
    //         printf("Found 0 id node: %d\n", v);
    //         exit(-1);
    //     }
    // }

    // Finding self-loop
    // for (int v = 0; v < g.nodes; v++) {
    //     for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
    //         int u = g.nlist[i];
    //         if (u == (v+1) {
    //             printf("Found self-loop: %d %d\n", v, u);
    //             exit(-1);
    //         }
    //     }
    // }


    // Check if the graph is undirected
    // for (int v = 0; v < g.nodes; v++) {
    //     for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
    //         int u = g.nlist[i];
    //         bool found = false;
    //         for (int j = g.nindex[u]; j < g.nindex[u + 1]; j++) {
    //             int w = g.nlist[j];
    //             if (w == (v+1)) {
    //                 found = true;
    //                 break;
    //             }
    //         }
    //         if (!found) {
    //             printf("(%d, %d) is not undirected\n", v, u);
    //         }
    //     }
    // }


    // Per-run stats (size = num_runs)
    std::vector<float> pa_ms_arr, ca_ms_arr, reduce_ms_arr, total_ms_arr;
    std::vector<int>   colors_arr, iter_count_arr;

    for (int run_idx = 0; run_idx < num_runs; ++run_idx) {
        if (run_idx > 0) {
            // Reset device state in place (allocAndInit already initialised
            // run 0). degree_list is re-filled by the init_degree<<<>>> launch
            // a few lines below.
            resetForRun(g, d);
        }

        GPUTimer timer_PA;
        GPUTimer timer_CA;

        // ================PA===================
        timer_PA.start();
        init_degree<<<ceil(g.nodes/512.0),512>>>(g.nodes, d.nidx_d, d.nlist_d,d.degree_list);
        cudaDeviceSynchronize();

        // Path A: pre-sort vertices by initial degree (only needed for BB variants)
        if (algo_name == "cuSL_ELS_BB" ||
            algo_name == "cuSL_ELS_BB_SPLIT") {
            bb_setup_sorted_S(g, d);
        }

        if (algo_name == "cuSL_ELS_BB_SPLIT") {
            int blkPerSM_split;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM_split,
                bb_split_phase1_peel, ThreadsPerBlock, 0);
            int gridDim_split = blkPerSM_split * SMs;
            run_bb_split(gridDim_split, g, d);
        } else if (algo_name == "cuSL_ELS_SDC_SPLIT") {
            int blkPerSM_sdc_split;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM_sdc_split,
                P_SL_ELS_SDC_split_scan, ThreadsPerBlock, 0);
            int gridDim_sdc_split = blkPerSM_sdc_split * SMs;
            run_sdc_split(gridDim_sdc_split, g, d);
        } else {
            int blkPerSM;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM,
              kernel_to_launch, ThreadsPerBlock, 0);
            int gridDim = blkPerSM * SMs;
            if (is_fused) {
              int edges_val = g.edges;
              void* args[] = { &g.nodes, &edges_val, &d.nidx_d, &d.nlist_d,
                &d.nlist2_d, &d.posscol_d, &d.posscol2_d, &d.color_d, &d.wl_d,
                &d.degree_list, &d.iteration_list_d };
              cudaLaunchCooperativeKernel(
                      (void*)kernel_to_launch,
                      dim3(gridDim), dim3(ThreadsPerBlock), args);
            } else {
              void* args[] = { &g.nodes, &d.nidx_d, &d.nlist_d,
                &d.degree_list, &d.iteration_list_d };
              cudaLaunchCooperativeKernel(
                      (void*)kernel_to_launch,
                      dim3(gridDim), dim3(ThreadsPerBlock), args);
            }
        }
        cudaDeviceSynchronize();
        float runtime_PA = timer_PA.stop();   // GPUTimer::stop returns seconds

        // Optional priority dump — only on the first run (output file is one
        // path; later runs would just overwrite with byte-identical data).
        if (run_idx == 0 && !dump_priority_path.empty()) {
            std::vector<unsigned int> host_iter(g.nodes);
            cudaMemcpy(host_iter.data(), d.iteration_list_d,
                       g.nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            for (int v = 0; v < g.nodes; ++v) host_iter[v] &= 0x3FFFFFFFu;
            FILE* fp = fopen(dump_priority_path.c_str(), "wb");
            if (fp == nullptr) {
                std::cerr << "ERROR: cannot open " << dump_priority_path
                          << " for write\n";
            } else {
                size_t w = fwrite(host_iter.data(), sizeof(unsigned int),
                                  host_iter.size(), fp);
                fclose(fp);
                std::cout << "dumped " << w << " priorities to "
                          << dump_priority_path << std::endl;
            }
        }

        // ================CA===================
        timer_CA.start();
        if (is_fused) {
          ECL_GC_coloring_only(blocks, g, d);
        } else {
          ECL_GC_run(blocks, g, d);
        }
        float runtime_CA = timer_CA.stop();

        ColorReductionStats reduction_stats =
            run_post_color_reduction(blocks, g, d, color, enable_reduce);

        float total_runtime = runtime_PA + runtime_CA + reduction_stats.runtime_sec;

        // iter_count: device global, no longer guarded by PROFILE — always 0
        // unless the kernel actually incremented it (which all PA kernels do
        // unconditionally now).
        int host_iter_count = 0;
        cudaMemcpyFromSymbol(&host_iter_count, iter_count, sizeof(int), 0,
                             cudaMemcpyDeviceToHost);

        // Verify (abort on failure).
        for (int v = 0; v < g.nodes; v++) {
            if (color[v] < 0) {
                printf("[Run %d/%d] ERROR: unprocessed node %d\n",
                       run_idx + 1, num_runs, v);
                exit(-1);
            }
            for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
                if (color[g.nlist[i]] == color[v] && g.nlist[i] != v) {
                    printf("[Run %d/%d] ERROR: adjacent nodes %d %d share "
                           "color %d\n", run_idx + 1, num_runs,
                           v, g.nlist[i], color[v]);
                    exit(-1);
                }
            }
        }

        // Pass color count from reduction_stats (after color reduction).
        int colors_after = reduction_stats.colors_after;

        // Per-run line.
        if (num_runs > 1) {
            printf("[Run %d/%d] PA: %8.3f ms  CA: %7.3f ms  Reduce: %6.3f ms  "
                   "Total: %8.3f ms  colors: %d  iters: %d\n",
                   run_idx + 1, num_runs,
                   runtime_PA * 1000, runtime_CA * 1000,
                   reduction_stats.runtime_sec * 1000, total_runtime * 1000,
                   colors_after, host_iter_count);
        } else {
            // Backward-compatible single-run output (verbose).
            std::cout << "Finish PA" << (is_fused ? "+Init" : "") << std::endl;
            std::cout << "Finish CA" << (is_fused ? " (coloring only)" : "")
                      << std::endl;
            printf("PA runtime: %.6f ms\n", runtime_PA * 1000);
            printf("CA runtime: %.6f ms\n", runtime_CA * 1000);
            printf("Post reduction runtime: %.6f ms\n",
                   reduction_stats.runtime_sec * 1000);
            printf("Total runtime: %.6f ms\n", total_runtime * 1000);
            printf("colors before reduction: %d\n", reduction_stats.colors_before);
            printf("colors after reduction: %d\n", reduction_stats.colors_after);
            printf("color reduction delta: %d\n",
                   reduction_stats.colors_before - reduction_stats.colors_after);
            printf("result verification passed\n");
            printf("colors used: %d\n", colors_after);
            printf("Iter count: %d\n", host_iter_count);
        }

        pa_ms_arr.push_back(runtime_PA * 1000);
        ca_ms_arr.push_back(runtime_CA * 1000);
        reduce_ms_arr.push_back(reduction_stats.runtime_sec * 1000);
        total_ms_arr.push_back(total_runtime * 1000);
        colors_arr.push_back(colors_after);
        iter_count_arr.push_back(host_iter_count);
    }

    // Final summary across runs (only when num_runs > 1).
    if (num_runs > 1) {
        auto stats_f = [](const std::vector<float>& v) {
            float s = 0, mn = v[0], mx = v[0];
            for (float x : v) { s += x; mn = std::min(mn, x); mx = std::max(mx, x); }
            char buf[160];
            snprintf(buf, sizeof(buf),
                     "avg=%9.3f  min=%9.3f  max=%9.3f", s / v.size(), mn, mx);
            return std::string(buf);
        };
        auto stats_i = [](const std::vector<int>& v) {
            double s = 0; int mn = v[0], mx = v[0];
            for (int x : v) { s += x; mn = std::min(mn, x); mx = std::max(mx, x); }
            char buf[160];
            snprintf(buf, sizeof(buf),
                     "avg=%9.3f  min=%9d  max=%9d", s / v.size(), mn, mx);
            return std::string(buf);
        };
        printf("\n=== Statistics over %d runs (ms) ===\n", num_runs);
        printf("PA time     : %s\n", stats_f(pa_ms_arr).c_str());
        printf("CA time     : %s\n", stats_f(ca_ms_arr).c_str());
        printf("Reduce time : %s\n", stats_f(reduce_ms_arr).c_str());
        printf("Total time  : %s\n", stats_f(total_ms_arr).c_str());
        printf("colors used : %s\n", stats_i(colors_arr).c_str());
        printf("iter count  : %s\n", stats_i(iter_count_arr).c_str());
    }
    
    cudaFree(d.wl_d);  cudaFree(d.color_d);  cudaFree(d.posscol2_d);  cudaFree(d.posscol_d);  cudaFree(d.nlist2_d);  cudaFree(d.nlist_d);  cudaFree(d.nidx_d);
    delete [] color;
    freeECLgraph(g);
    return 0;
}

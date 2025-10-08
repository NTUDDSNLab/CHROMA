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

// namespace cg = cooperative_groups;

#ifdef PRED_MODEL
extern double score(double *input);
#endif


enum GraphFormat {
    ECLGraph,
    SNAPGraph,
    MMIOGraph
};


// 添加幫助函數
void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -f, --file <path>         Input graph file path (required)\n";
    std::cout << "                            Supported formats:\n";
    std::cout << "                            .egr  : ECL graph format (binary)\n";
    std::cout << "                            .txt  : Text format (CSR graph)\n";
    std::cout << "                            .bin  : Binary format (CSR graph)\n";
    std::cout << "  -a, --algorithm <algo>    Select algorithm:\n";
    std::cout << "                            0 or P_SL_WBR     : P_SL_WBR algorithm\n";
    std::cout << "                            1 or P_SL_WBR_SDC : P_SL_WBR_SDC algorithm\n";
    std::cout << "                            (default: P_SL_WBR)\n";
    std::cout << "  -r, --resilient <number>  Set resilient number θ value (default: 0)\n";
    std::cout << "  -p, --predict             Use prediction model for resilient parameter\n";
    std::cout << "  -h, --help                Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " -f graph.txt\n";
    std::cout << "  " << program_name << " --file graph.egr -a P_SL_WBR_SDC\n";
    std::cout << "  " << program_name << " -f graph.bin --algorithm 1 --resilient 15\n";
    std::cout << "  " << program_name << " -f graph.egr --predict\n";
}

// 解析算法參數
void* select_algorithm(const std::string& algo_str, std::string& algo_name) {
    if (algo_str == "0" || algo_str == "P_SL_WBR") {
        algo_name = "P_SL_WBR";
        return (void*)P_SL_WBR;
    } else if (algo_str == "1" || algo_str == "P_SL_WBR_SDC") {
        algo_name = "P_SL_WBR_SDC";
        return (void*)P_SL_WBR_SDC;
    } else {
        std::cerr << "Error: Invalid algorithm '" << algo_str << "'. Using default P_SL_WBR.\n";
        algo_name = "P_SL_WBR";
        return (void*)P_SL_WBR;
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

  cudaMemcpyToSymbol(random_vals_d, &random_vals_device, sizeof(unsigned int*)); // 使用 cudaMemcpyToSymbol 复制指针
  
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
    // 默認設置
    std::string filename;
    int fuzzy_number = 0;  // RGC θ 默認值為 0
    void* kernel_to_launch = (void*)P_SL_WBR;
    std::string algo_name = "P_SL_WBR";
    bool use_predicted_resilient = false;  // 標記是否使用預測的 resilient 值

    // 解析命令行參數
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
                kernel_to_launch = select_algorithm(argv[++i], algo_name);
            } else {
                std::cerr << "Error: Algorithm option requires an argument.\n";
                print_help(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--resilient") == 0) {
            if (i + 1 < argc) {
                try {
                    fuzzy_number = std::stoi(argv[++i]);
                    use_predicted_resilient = false;  // 手動指定時不使用預測
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid RGC value '" << argv[i] << "'.\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: RGC option requires an argument.\n";
                print_help(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--predict") == 0) {
            use_predicted_resilient = true;
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

    // 檢查是否提供了圖形文件
    if (filename.empty()) {
        std::cerr << "Error: No graph file specified. Use -f or --file to specify input file.\n";
        print_help(argv[0]);
        return 1;
    }

    // 讀取圖形文件
    // 讀取圖形文件
    ECLgraph g;
    CSRGraph graph;
    
    // 根據文件擴展名決定使用哪種格式
    std::string file_extension = filename.substr(filename.find_last_of(".") + 1);
    
    if (file_extension == "egr") {
        // 使用 ECLgraph 讀取 .egr 文件
        g = readECLgraph(filename.c_str());
        // 創建一個空的 CSRGraph 用於兼容後續代碼
        graph.file_num_nodes = g.nodes;
        graph.file_num_edges = g.edges;
        graph.max_node_id = g.nodes - 1;
        std::cout << "Read .egr file using ECLgraph" << std::endl;
    } else if (file_extension == "txt" || file_extension == "bin") {
        // 使用 CSRGraph 讀取 .txt 或 .bin 文件
        if (file_extension == "bin") {
            graph.loadFromBinary(filename);
        } else {
            graph.buildFromTxtFile(filename, false);
        }
        std::cout << "build from " << file_extension << " file" << std::endl;
        std::cout << filename << " is zero based: " << graph.is_zero_based << std::endl;
        // Transform to ECLgraph
        graph.transfromToECLGraph(g);
    } else {
        std::cerr << "Error: Unsupported file format '" << file_extension << "'. Supported formats: .egr, .txt, .bin" << std::endl;
        return 1;
    }
    
    // 如果啟用預測模型，使用 score function 預測 resilient parameter
    #ifdef PRED_MODEL
    if (use_predicted_resilient) {
        // 使用圖形的節點數和邊數作為預測模型的輸入
        double input[2] = {(double)g.nodes, (double)g.edges};
        double score_result = score(input);
        fuzzy_number = (int)round(score_result);
    }
    #else
    if (use_predicted_resilient) {
        std::cerr << "Error: Prediction model is not compiled. Please compile with PRED_MODEL flag.\n";
        return 1;
    }
    #endif

    // 顯示設置信息
    printf("Input file: %s\n", filename.c_str());
    printf("Algorithm: %s\n", algo_name.c_str());
    if (use_predicted_resilient) {
        printf("RGC θ: %d (Predicted)\n", fuzzy_number);
    } else {
        printf("RGC θ: %d\n", fuzzy_number);
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
    allocAndInit(g, d);

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


    GPUTimer timer;
    timer.start();
    // ================PA===================
    init_degree<<<ceil(g.nodes/512.0),512>>>(g.nodes, d.nidx_d, d.nlist_d,d.degree_list);
    cudaDeviceSynchronize();
    int blkPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM,
      kernel_to_launch, ThreadsPerBlock, 0);
    int gridDim = blkPerSM * SMs;      // 一定能同時常駐
    int *out_d;  cudaMalloc(&out_d, g.nodes * sizeof(int));
    void* args[] = { &g.nodes, &d.nidx_d, &d.nlist_d,
      &d.degree_list, &d.iteration_list_d };
    cudaLaunchCooperativeKernel(
            (void*)kernel_to_launch,
            dim3(gridDim), dim3(ThreadsPerBlock), args);
    cudaDeviceSynchronize();
    // ================CA===================
    ECL_GC_run(blocks, g, d);
    const float runtime = timer.stop();
    if (cudaSuccess != cudaMemcpy(color, d.color_d, g.nodes * sizeof(int), cudaMemcpyDeviceToHost)) {printf("ERROR: copying color from device failed\n\n");  exit(-1);}
    verifyAndPrintStats(g, color, runtime);

    #ifdef PROFILE
    int host_iter_count = 0;
    cudaMemcpyFromSymbol(&host_iter_count, iter_count, sizeof(int), 0, cudaMemcpyDeviceToHost);
    printf("Iter count: %d\n", host_iter_count);
    #endif
    
    cudaFree(d.wl_d);  cudaFree(d.color_d);  cudaFree(d.posscol2_d);  cudaFree(d.posscol_d);  cudaFree(d.nlist2_d);  cudaFree(d.nlist_d);  cudaFree(d.nidx_d);
    delete [] color;
    freeECLgraph(g);
    return 0;
}
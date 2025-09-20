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

// namespace cg = cooperative_groups;

// 添加幫助函數
void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -f, --file <path>         Input graph file path (required)\n";
    std::cout << "  -a, --algorithm <algo>    Select algorithm:\n";
    std::cout << "                            0 or P_SL_WBR     : P_SL_WBR algorithm\n";
    std::cout << "                            1 or P_SL_WBR_SDC : P_SL_WBR_SDC algorithm\n";
    std::cout << "                            (default: P_SL_WBR)\n";
    std::cout << "  -r, --resilient <number>  Set resilient number θ value (default: 10)\n";
    std::cout << "  -h, --help                Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " -f graph.txt\n";
    std::cout << "  " << program_name << " --file graph.txt -a P_SL_WBR_SDC\n";
    std::cout << "  " << program_name << " -f graph.txt --algorithm 1 --resilient 15\n";
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
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id < nodes) {
        int deg = nidx[vertex_id + 1] - nidx[vertex_id];
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
    int fuzzy_number = 10;  // RGC θ 默認值為 10
    void* kernel_to_launch = (void*)P_SL_WBR;
    std::string algo_name = "P_SL_WBR";

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
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid RGC value '" << argv[i] << "'.\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: RGC option requires an argument.\n";
                print_help(argv[0]);
                return 1;
            }
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
    ECLgraph g = readECLgraph(filename.c_str());
    
    // 顯示設置信息
    printf("Input file: %s\n", filename.c_str());
    printf("Algorithm: %s\n", algo_name.c_str());
    printf("RGC θ: %d\n", fuzzy_number);
    printf("Nodes: %d\n", g.nodes);
    printf("Edges: %d\n", g.edges);
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
    
    cudaFree(d.wl_d);  cudaFree(d.color_d);  cudaFree(d.posscol2_d);  cudaFree(d.posscol_d);  cudaFree(d.nlist2_d);  cudaFree(d.nlist_d);  cudaFree(d.nidx_d);
    delete [] color;
    freeECLgraph(g);
    return 0;
}
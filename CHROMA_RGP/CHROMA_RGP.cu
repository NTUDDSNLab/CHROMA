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

// namespace cg = cooperative_groups;


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
  // SEETING
  std::string filename;
  int fuzzy_number = 10;

  std::cout << "Enter input filename: ";
  std::cin >> filename;
  std::cin.ignore();

  std::string input_line;
  std::cout << "Enter RGC (default \u03B8=10): ";
  std::getline(std::cin, input_line);

  if (!input_line.empty()) {
    fuzzy_number = std::stoi(input_line);  // 有輸入，轉成 int
  }
  std::cout << "Using RGC \u03B8 = " << fuzzy_number << std::endl;
  ECLgraph g = readECLgraph(filename.c_str());
  std::string sdc_choice;
  std::cout << "Enable SDC optimization? (y/n): ";
  std::getline(std::cin, sdc_choice); 

  void* kernel_to_launch = nullptr;
  if (!sdc_choice.empty() && (sdc_choice[0] == 'y' || sdc_choice[0] == 'Y')) {
      kernel_to_launch = (void*)P_SL_WBR_SDC;
      std::cout << "Using P_SL_WBR_SDC\n";
  } else {
      kernel_to_launch = (void*)P_SL_WBR;
      std::cout << "Using P_SL_WBR\n";
  }
  int nParts = 0;
  std::string line;
  
  // 提示輸入
  std::cout << "Enter number of partitions (nParts > 1): ";
  std::getline(std::cin, line);
  
  // 轉為整數並檢查
  if (!line.empty()) {
      nParts = std::stoi(line);
  }
  if (nParts <= 1) {
      std::cerr << "ERROR: nParts must be greater than 1.\n";
      exit(EXIT_FAILURE);
  }
  printf("input: %s\n", filename.c_str());
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);
  setParameters<<<1, 1>>>(fuzzy_number);
  // METIS
  std::vector<ECLgraph> subGraphs;
  std::vector<std::vector<int>> sub_localToGlobal, sub_boundaryToLocal;
  ECLgraph boundaryGraph;
  std::vector<int> part;  // 每個節點的分區結果
  
  partitionGraphWithBoundary(
      g, nParts,
      subGraphs,
      sub_localToGlobal,
      sub_boundaryToLocal,
      boundaryGraph,
      part);
  
  // 2. 執行邊界圖上著色（boundaryColor_d 等會回傳 device ptr）
  unsigned int* boundaryColor_d    = nullptr;
  unsigned int* boundaryPosColor_d = nullptr;
  unsigned int* boundaryIterList_d = nullptr;
  
  GPUTimer timer;
  timer.start();
  
  runBoundaryGC(0, boundaryGraph,
                &boundaryColor_d,
                &boundaryPosColor_d,
                &boundaryIterList_d);
  
  cudaDeviceSynchronize();
  
  // 3. 每張子圖由多 GPU 平行處理著色
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  
  std::vector<unsigned int*> partialColorVec(nParts, nullptr);
  std::vector<unsigned int*> partialIterVec(nParts,  nullptr);
  
  #pragma omp parallel num_threads(deviceCount)
  {
      int threadID = omp_get_thread_num();
      for (int i = threadID; i < nParts; i += deviceCount) {
          runOnGPU(threadID,
                   subGraphs[i],
                   sub_boundaryToLocal[i],
                   &partialColorVec[i],
                   &partialIterVec[i],
                   boundaryColor_d,
                   boundaryPosColor_d);
          cudaDeviceSynchronize();
      }
  }
  unsigned int* d_color_list     = nullptr;
  unsigned int* d_iteration_list = nullptr;
  cudaMalloc(&d_color_list,     g.nodes*sizeof(unsigned int));
  cudaMalloc(&d_iteration_list, g.nodes*sizeof(unsigned int));
  cudaMemset(d_color_list,     0, g.nodes*sizeof(unsigned int));
  cudaMemset(d_iteration_list, 0, g.nodes*sizeof(unsigned int));
  // printf("===================Merge Start===================\n");
  // For each partition i, copy sub_localToGlobal[i] to GPU0, then run merge
  for (int i = 0; i < nParts; i++) {
    // Upload localToGlobal array
    int subSize = (int)sub_localToGlobal[i].size();
    int* d_localToGlobal = nullptr;
    cudaMalloc(&d_localToGlobal, subSize * sizeof(int));
    cudaMemcpy(d_localToGlobal,
               sub_localToGlobal[i].data(),
               subSize * sizeof(int),
               cudaMemcpyHostToDevice);

    // ✅ Upload partialColorVec[i] to device
    unsigned int* d_partialColor = nullptr;
    cudaMalloc(&d_partialColor, subSize * sizeof(unsigned int));
    cudaMemcpy(d_partialColor,
               partialColorVec[i],
               subSize * sizeof(unsigned int),
               cudaMemcpyHostToDevice);

    // ✅ Upload partialIterVec[i] to device
    unsigned int* d_partialIter = nullptr;
    cudaMalloc(&d_partialIter, subSize * sizeof(unsigned int));
    cudaMemcpy(d_partialIter,
               partialIterVec[i],
               subSize * sizeof(unsigned int),
               cudaMemcpyHostToDevice);

    // Now call your merge kernel
    int gridSize = (subSize + ThreadsPerBlock - 1) / ThreadsPerBlock;
    mergeIterListOnGPU<<<gridSize, ThreadsPerBlock>>>(
        subSize,
        d_localToGlobal,
        d_partialColor,
        (int*)d_color_list,
        d_partialIter,
        d_iteration_list
    );
    cudaDeviceSynchronize();

    // Free device copies
    cudaFree(d_localToGlobal);
    cudaFree(d_partialColor);
    cudaFree(d_partialIter);
}

  if (cudaSuccess != cudaMemcpy(color, d.color_d, g.nodes * sizeof(int), cudaMemcpyDeviceToHost)) {printf("ERROR: copying color from device failed\n\n");  exit(-1);}
  verifyAndPrintStats(g, color, runtime);
  
  cudaFree(d.wl_d);  cudaFree(d.color_d);  cudaFree(d.posscol2_d);  cudaFree(d.posscol_d);  cudaFree(d.nlist2_d);  cudaFree(d.nlist_d);  cudaFree(d.nidx_d);
  delete [] color;
  freeECLgraph(g);
  return 0;
}
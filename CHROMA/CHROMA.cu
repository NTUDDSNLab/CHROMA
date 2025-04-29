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
  printf("input: %s\n", filename.c_str());
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);
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
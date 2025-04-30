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
  int fuzzy_number = 0;

  std::cout << "Enter input filename: ";
  std::cin >> filename;
  std::cin.ignore();
  ECLgraph g = readECLgraph(filename.c_str());

  std::string user_choice;
  void* kernel_to_launch = nullptr;
  
  std::cout << "Select algorithm:\n";
  std::cout << "1) JP-SL\n";
  std::cout << "2) JP-ADG\n";
  std::cout << "3) JP-SLL\n";
  std::cout << "Enter your choice (1/2/3): ";
  std::getline(std::cin, user_choice);
  
  if (!user_choice.empty()) {
      switch (user_choice[0]) {
          case '1':
              kernel_to_launch = (void*)JP_SL;
              std::cout << "Using JP-SL\n";
              break;
          case '2':
              kernel_to_launch = (void*)JP_ADG;
              std::cout << "Using JP-ADG\n";
              break;
          case '3':
              kernel_to_launch = (void*)JP_SLL;
              std::cout << "Using JP-SLL\n";
              break;
          default:
              std::cerr << "Invalid choice. Defaulting to JP-SL.\n";
              kernel_to_launch = (void*)JP_SL;
      }
  } else {
      std::cerr << "No input. Defaulting to JP-SL.\n";
      kernel_to_launch = (void*)JP_SL;
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
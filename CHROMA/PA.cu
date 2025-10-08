#include "globals.cuh"
#include <cuda.h>

__global__ void P_SL_WBR_SDC(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
const int threads = gridDim.x * blockDim.x;
const int lane    = threadIdx.x & 31;
const int warpId  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
const int numWarp = (gridDim.x * blockDim.x) >> 5;    
cg::grid_group grid = cg::this_grid();

do {
int add_num=0;
unsigned int localMin = 0x7FFFFFFF;
for (int v = tid; v < nodes; v += threads) {
  int iteration_list_v=iteration_list[v];
  unsigned int prio = (iteration_list_v >> 30) & 0x1;
  unsigned int iteration_v = (iteration_list_v) & 0x3FFFFFFF;
  unsigned int large_deg=0;
  unsigned int degree=degree_list[v];
  if(prio==0){
      if(degree<=(theta+FuzzyNumber)){
      prio=1;
      add_num++;
      int beg = nidx[v];
      int end = nidx[v + 1];
      if((end-beg)>=WS)large_deg=1;
      iteration_list[v]= (large_deg << 31)|(prio << 30) |(iteration_v+(degree-theta)+1);
      remove_list[atomicAdd(&remove_size, 1)] = v;
      }else{
        if (degree < localMin) localMin = degree;
        iteration_list[v]= (large_deg << 31)|(prio << 30) |(iteration_v+FuzzyNumber+1);
      }
  }
}
if (localMin < 0x7FFFFFFF)atomicMin(&g_minDegree, localMin);
if (add_num!=0) atomicAdd(&worker, add_num);

grid.sync();
unsigned int warpMin;

for (int k = warpId; k < remove_size; k += numWarp){
  int v = remove_list[k];
  int beg, end;
  if (lane == 0) {
      beg = nidx[v];
      end = nidx[v + 1];
  }
  beg = __shfl_sync(0xffffffff, beg, 0); 
  end = __shfl_sync(0xffffffff, end, 0);

  warpMin = 0x7fffffff;
  for (int i = beg + lane; i < end; i += 32)
  {
      int nei      = __ldg(nlist + i);
      unsigned int iter = __ldg(iteration_list + nei);

      if (!(iter & 0x40000000u)) { 
          warpMin = min(warpMin, atomicSub(&degree_list[nei], 1));
      }
  }
    warpMin = warpReduceMin(warpMin);
    if (lane == 0 && warpMin < 0x7fffffff){
      atomicMin(&g_minDegree, warpMin);
    }
}

grid.sync();

if(grid.thread_rank()==0){
  remove_size=0;
  theta=g_minDegree;
  atomicExch(&g_minDegree, 0x7FFFFFFF);
  iteration=iteration+1+FuzzyNumber;
  #ifdef PROFILE
  iter_count++;
  #endif
}
grid.sync();
}while(worker!=nodes);
}

__global__ void P_SL_WBR(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
const int threads = gridDim.x * blockDim.x;
const int lane    = threadIdx.x & 31;
const int warpId  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
const int numWarp = (gridDim.x * blockDim.x) >> 5;    
cg::grid_group grid = cg::this_grid();

do {
int add_num=0;
unsigned int localMin = 0x7FFFFFFF;
for (int v = tid; v < nodes; v += threads) {
  int iteration_list_v=iteration_list[v];
  unsigned int prio = (iteration_list_v >> 30) & 0x1;
  unsigned int iteration_v = (iteration_list_v) & 0x3FFFFFFF;
  unsigned int large_deg=0;
  unsigned int degree=degree_list[v];
  if(prio==0){
      if(degree<=(theta+FuzzyNumber)){
      prio=1;
      add_num++;
      int beg = nidx[v];
      int end = nidx[v + 1];
      if((end-beg)>=WS)large_deg=1;
      iteration_list[v]= (large_deg << 31)|(prio << 30) |(iteration_v+(degree-theta)+1);
      remove_list[atomicAdd(&remove_size, 1)] = v;
      }else{
        if (degree < localMin) localMin = degree;
        iteration_list[v]= (large_deg << 31)|(prio << 30) |(iteration_v+FuzzyNumber+1);
      }
  }
}
if (localMin < 0x7FFFFFFF)atomicMin(&g_minDegree, localMin);
if (add_num!=0) atomicAdd(&worker, add_num);

grid.sync();
for (int k = warpId; k < remove_size; k += numWarp){
  int v = remove_list[k];
  int beg, end;
  if (lane == 0) {
      beg = nidx[v];
      end = nidx[v + 1];
  }
  beg = __shfl_sync(0xffffffff, beg, 0); 
  end = __shfl_sync(0xffffffff, end, 0);
  for (int i = beg + lane; i < end; i += 32)
  {
      int nei      = __ldg(nlist + i);
      unsigned int iter = __ldg(iteration_list + nei);

      if (!(iter & 0x40000000u)) { 
        atomicSub(&degree_list[nei], 1);
      }
  }
}

grid.sync();

if(grid.thread_rank()==0){
  remove_size=0;
  theta=g_minDegree;
  atomicExch(&g_minDegree, 0x7FFFFFFF);
  iteration=iteration+1+FuzzyNumber;
  #ifdef PROFILE
  iter_count++;
  #endif
}
grid.sync();
}while(worker!=nodes);
}
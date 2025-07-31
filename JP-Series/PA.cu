#include "globals.cuh"
#include <cuda.h>

__global__ void JP_SL(
  const int  nodes,
  const int* __restrict__ nidx,
  const int* __restrict__ nlist,
  unsigned int* __restrict__ degree_list1,
  unsigned int* __restrict__ iteration_list)
{
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  const int threads = gridDim.x * blockDim.x;
  cg::grid_group grid = cg::this_grid();
  /*--------------- Phase-1: stride 走訪所有頂點 -------------------*/
  do {
    int add_num=0;
    unsigned int localMin = 0x7FFFFFFF;
    for (int v = tid; v < nodes; v += threads) {
      unsigned int degree=degree_list1[v];
      int iteration_list_v=iteration_list[v];
      unsigned int prio = (iteration_list_v >> 30) & 0x1;         // Extract the highest bit (MSB)

      if(prio==0){
        if (degree < localMin) localMin = degree;
        }
    }
    if (localMin < 0x7FFFFFFF)atomicMin(&g_minDegree, localMin);
    grid.sync();
    for (int v = tid; v < nodes; v += threads) {
      int iteration_list_v=iteration_list[v];
      unsigned int prio = (iteration_list_v >> 30) & 0x1;         // Extract the highest bit (MSB)
      unsigned int iteration_v = (iteration_list_v) & 0x3FFFFFFF;
      unsigned int large_deg=0;
      unsigned int degree=degree_list1[v];
      if(prio==0){
          if(degree<=(g_minDegree+FuzzyNumber)){
          prio=1;
          add_num++;
          int beg = nidx[v];
          int end = nidx[v + 1];
          if((end-beg)>=WS)large_deg=1;
          iteration_list[v]= (large_deg << 31)|(prio << 30) |(iteration_v+(degree-g_minDegree)+1);
          for (int i = beg; i < end; i++) {
            int nei = nlist[i];
            if(((iteration_list[nei] >> 30) & 0x1)==0){
              int temp_min=atomicSub(&degree_list1[nei], 1);

            }
          }
          }else{
            iteration_list[v]= (large_deg << 31)|(prio << 30) |(iteration_v+FuzzyNumber+1);
          }
      }
    }
    if (add_num!=0) atomicAdd(&worker, add_num);

    grid.sync();
    if(grid.thread_rank()==0){
    atomicExch(&g_minDegree, 0x7FFFFFFF);
    iteration=iteration+1+FuzzyNumber;
    }
    grid.sync();
  }while(worker!=nodes);
}

__global__ void JP_SLL(
  const int  nodes,
  const int* __restrict__ nidx,
  const int* __restrict__ nlist,
  unsigned int* __restrict__ degree_list1,
  unsigned int* __restrict__ iteration_list)
{
const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
const int threads = gridDim.x * blockDim.x;

cg::grid_group grid = cg::this_grid();

/*--------------- Phase-1: stride 走訪所有頂點 -------------------*/
do {
int add_num=0;
grid.sync();
for (int v = tid; v < nodes; v += threads) {
// if(v>=nodes) break;
int iteration_list_v=iteration_list[v];
unsigned int prio = (iteration_list_v >> 30) & 0x1;         // Extract the highest bit (MSB)
unsigned int iteration_v = (iteration_list_v) & 0x3FFFFFFF;
unsigned int large_deg=0;
unsigned int degree=degree_list1[v];
if(prio==0){
    if(degree<=((1u << iteration)+FuzzyNumber)){
    prio=1;
    add_num++;
    int beg = nidx[v];
    int end = nidx[v + 1];
    if((end-beg)>=WS)large_deg=1;
    iteration_list[v]= (large_deg << 31)|(prio << 30) |(iteration_v);
    // printf("pro[%d]:%d\n",v,(degree-theta));
    for (int i = beg; i < end; i++) {
      int nei = nlist[i];
      if(((iteration_list[nei] >> 30) & 0x1)==0){
        int temp_min=atomicSub(&degree_list1[nei], 1);
      }
    }
    }else{
      iteration_list[v]= (large_deg << 31)|(prio << 30) |(iteration_v+1);
    }
}
}
if (add_num!=0) atomicAdd(&worker, add_num);

grid.sync();
if(grid.thread_rank()==0){
iteration=iteration+1+FuzzyNumber;
}
grid.sync();
}while(worker!=nodes);
}

__global__ void JP_ADG(
  const int  nodes,
  const int* __restrict__ nidx,
  const int* __restrict__ nlist,
  unsigned int* __restrict__ degree_list1,
  unsigned int* __restrict__ iteration_list)
{
const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
const int threads = gridDim.x * blockDim.x;

cg::grid_group grid = cg::this_grid();

/*--------------- Phase-1: stride 走訪所有頂點 -------------------*/
do {
int add_num=0;
// unsigned int localMin = 0x7FFFFFFF;
for (int v = tid; v < nodes; v += threads) {
unsigned int degree=degree_list1[v];
int iteration_list_v=iteration_list[v];
unsigned int prio = (iteration_list_v >> 30) & 0x1;         // Extract the highest bit (MSB)
if(prio==0){
  atomicAdd(&total_deg, degree);
  atomicAdd(&total_worker, 1);
}
}
// if (localMin < 0x7FFFFFFF)atomicMin(&g_minDegree, localMin);

grid.sync();

if(grid.thread_rank()==0){
atomicExch(&avg_deg, int(total_deg/total_worker));
// printf("avg is :%d\n",avg_deg);
atomicExch(&total_deg, 0);
atomicExch(&total_worker, 0);
}

grid.sync();
for (int v = tid; v < nodes; v += threads) {
// if(v>=nodes) break;
int iteration_list_v=iteration_list[v];
unsigned int prio = (iteration_list_v >> 30) & 0x1;         // Extract the highest bit (MSB)
unsigned int iteration_v = (iteration_list_v) & 0x3FFFFFFF;
unsigned int large_deg=0;
unsigned int degree=degree_list1[v];
if(prio==0){
    if(degree<=(avg_deg)){
    prio=1;
    add_num++;
    int beg = nidx[v];
    int end = nidx[v + 1];
    if((end-beg)>=WS)large_deg=1;
    iteration_list[v]= (large_deg << 31)|(prio << 30) |(iteration_v);
    // printf("pro[%d]:%d\n",v,(degree-avg_deg));
    for (int i = beg; i < end; i++) {
      int nei = nlist[i];
      if(((iteration_list[nei] >> 30) & 0x1)==0){
        int temp_min=atomicSub(&degree_list1[nei], 1);

      }
    }
    }else{
      iteration_list[v]= (large_deg << 31)|(prio << 30) |(iteration_v);
    }
}
}
if (add_num!=0) atomicAdd(&worker, add_num);

grid.sync();
if(grid.thread_rank()==0){
// printf("iteration:%d\n",iteration);
// atomicExch(&g_minDegree, 0x7FFFFFFF);
iteration=iteration+1+FuzzyNumber;
}
grid.sync();
}while(worker!=nodes);
// if(grid.thread_rank()==0)printf("iteration: %d\n", iteration);
}

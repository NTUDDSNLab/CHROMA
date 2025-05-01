#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <metis.h>
#include "ECLgraph.h"
#include <omp.h>
#include <cuda.h>
#include <random>
#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <cuda_runtime.h>
#ifndef MAXCOLOR
#define MAXCOLOR 1024  // 假設我們容許的最大顏色編號 (可依情況調整)
#endif
static const int Device = 0;
static const int ThreadsPerBlock = 512;
static const unsigned int Warp = 0xffffffff;
static const int WS = 32;  // warp size and bits per int
static const int MSB = 1 << (WS - 1);
static const int Mask = (1 << (WS / 2)) - 1;
static __device__ int FuzzyNumber =0;
static __device__ int wlsize = 0;

__device__ unsigned int* random_vals_d;
// *=======================[STRUCT]=====================*
// =====================================
// ||                                 ||
// ||            GPU STATUS           ||
// ||                                 ||
// =====================================
__device__ __forceinline__
unsigned int warpReduceMin(unsigned int val)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

struct GPUState {
    int worker;
    int theta;
    int iteration;
    int g_minDegree;
    int  remove_size;
};
// =====================================
// ||                                 ||
// ||              TIMER              ||
// ||                                 ||
// =====================================

struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg);  cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg);  cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  float stop() {cudaEventRecord(end, 0);  cudaEventSynchronize(end);  float ms;  cudaEventElapsedTime(&ms, beg, end);  return 0.001f * ms;}
};

// *=======================[DEVICE]=====================*
// =====================================
// ||                                 ||
// ||               HASH              ||
// ||                                 ||
// =====================================

static __device__ unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}
// *=======================[GPU]=========================*

// =====================================
// ||                                 ||
// ||   applyBoundaryColorToSubgraph  ||
// ||                                 ||
// =====================================
__global__
void applyBoundaryColorToSubgraph(
    int nodes,
    const int* __restrict__ boundaryToLocal, 
    const unsigned int* __restrict__ boundaryColor, 
    unsigned int* __restrict__ subColor,
    const unsigned int* __restrict__ boundaryPosColor, 
    unsigned int* __restrict__ posscol_d
) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int threads = gridDim.x * blockDim.x;
  if (thread >= nodes) return;
  for (int idx = thread; idx < nodes; idx += threads) {
        int localID = boundaryToLocal[idx]; 
        if (localID >= 0) { 
            unsigned color=boundaryColor[idx];
            subColor[localID] = color;
            posscol_d[localID] = boundaryPosColor[idx];
        }
    }
}


__global__
void applyBoundaryPrio(
    int nodes,
    const int* __restrict__ boundaryToLocal,
    unsigned int* __restrict__ subItr
) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int threads = gridDim.x * blockDim.x;
  if (thread >= nodes) return;
  for (int idx = thread; idx < nodes; idx += threads) {

        int localID = boundaryToLocal[idx];
        if (localID >= 0) {
          subItr[localID] = (static_cast<unsigned int>(1) << 31) | subItr[localID];
        }
    }
}
// =====================================
// ||                                 ||
// ||          SET PARAMETER          ||
// ||                                 ||
// =====================================

__global__ void setParameters(int fuzzy_number) {
  FuzzyNumber = fuzzy_number;
}

// =====================================
// ||                                 ||
// ||           CHROMA                ||
// ||                                 ||
// =====================================
__global__ void CHROMA(
  const int  nodes,
  const int* __restrict__ nidx,
  const int* __restrict__ nlist,
  unsigned int* __restrict__ degree_list1,
  unsigned int* __restrict__ iteration_list,
  GPUState* __restrict__ state,
  int* __restrict__ remove_list
)
{
const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
const int threads = gridDim.x * blockDim.x;
const int lane    = threadIdx.x & 31;                 // 0‥31
const int warpId  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
const int numWarp = (gridDim.x * blockDim.x) >> 5;    // 全 Grid warp 數
cg::grid_group grid = cg::this_grid();

do {
  int add_num=0;
  unsigned int localMin = 0x7FFFFFFF;
  for (int v = tid; v < nodes; v += threads) {
  if(v>=nodes) break;
  int iteration_list_v=iteration_list[v];
  unsigned int prio = (iteration_list_v >> 29) & 0x1;
  unsigned int iteration_v = iteration_list_v & 0x0FFFFFFF;
  unsigned int degree=degree_list1[v];
  if(prio==0){
      if(degree<=(state->theta+FuzzyNumber)){
        prio=1;
        add_num++;
        int beg = nidx[v];
        int end = nidx[v + 1];
        remove_list[atomicAdd(&(state->remove_size), 1)] = v;
      }else{
        if (degree < localMin) localMin = degree;
      }
      iteration_list[v]= (prio << 29) |(state->iteration+FuzzyNumber+1);
  }
  }
  if (localMin < 0x7FFFFFFF)atomicMin(&(state->g_minDegree), localMin);
  if (add_num != 0) atomicAdd(&(state->worker), add_num);

  grid.sync();
  unsigned int warpMin;

  for (int k = warpId; k < state->remove_size; k += numWarp){
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

      if (!(iter & 0x20000000u)) {
          warpMin = min(warpMin, atomicSub(&degree_list1[nei], 1));
      }
  }
    warpMin = warpReduceMin(warpMin);
    if (lane == 0 && warpMin < 0x7fffffff){
      atomicMin(&(state->g_minDegree), warpMin);
    }
  }

  grid.sync();

  if(grid.thread_rank()==0){
    state->remove_size=0;
    state->theta=state->g_minDegree;
    atomicExch(&state->g_minDegree, 0x7FFFFFFF);
    state->iteration=state->iteration+1+FuzzyNumber;
  }
  grid.sync();
}while(state->worker!=nodes);
}

// =====================================
// ||                                 ||
// ||         CAL INIT DEGREE         ||
// ||                                 ||
// =====================================

static __global__
void init_degree(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, unsigned int* const __restrict__ degree_list1) {
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id < nodes) {
        int deg = nidx[vertex_id + 1] - nidx[vertex_id];
        degree_list1[vertex_id] = deg;
    }
}
// =====================================
// ||                                 ||
// ||          MERGE PRIOLITY         ||
// ||                                 ||
// =====================================
__global__
void mergeIterListOnGPU(
    int subSize,
    const int* __restrict__ localToGlobal,
    const unsigned int* __restrict__ partialColorList,
    int* __restrict__ colorList,
    const unsigned int* __restrict__ partialIterList,
    unsigned int* __restrict__ iterationList
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < subSize) {
        int globalID = localToGlobal[i];
        iterationList[globalID] += partialIterList[i]& 0x3FFFFFFF;
        colorList[globalID] = partialColorList[i];
    }
}


// =====================================
// ||                                 ||
// ||          ECL-GC KERNAL          ||
// ||                                 ||
// =====================================
static __global__
void init(const int nodes, const int edges, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nlist2, int* const __restrict__ posscol, int* const __restrict__ posscol2, int* const __restrict__ color, int* const __restrict__ wl,unsigned int* const __restrict__ priority)
{
  const int lane = threadIdx.x % WS;
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;
  
  int maxrange = -1;
  for (int v = thread; __any_sync(Warp, v < nodes); v += threads) {
    bool cond = false;
    int beg, end, pos, degv, active;
    if (v < nodes) {
      beg = nidx[v];
      end = nidx[v + 1];
      degv = end - beg;
      cond = (degv >= WS);
      unsigned int v_priority=priority[v]| (cond << 30);
      if (cond) {
        wl[atomicAdd(&wlsize, 1)] = v;

      } else {
        active = 0;
        pos = beg;
        for (int i = beg; i < end; i++) {
          const int nei = nlist[i];
          const int degn = nidx[nei + 1] - nidx[nei];
          const unsigned int priority_n=priority[nei]|((degn >= WS)<< 30);
          unsigned int hash_v = hash(v);
          unsigned int hash_nei = hash(nei);
          if ((v_priority < priority_n) || ((v_priority == priority_n) && (hash_v < hash_nei)) || ((v_priority == priority_n) && (hash_v == hash_nei) && (v < nei))) {
            active |= (unsigned int)MSB >> (i - beg);
            pos++;
          }
        }
      }
    }

    int bal = __ballot_sync(Warp, cond);
    while (bal != 0) {
      const int who = __ffs(bal) - 1;
      bal &= bal - 1;
      const int wv = __shfl_sync(Warp, v, who);
      const int wbeg = __shfl_sync(Warp, beg, who);
      const int wend = __shfl_sync(Warp, end, who);
      const int wdegv = wend - wbeg;
      unsigned int wvpriority=priority[wv]|((wdegv >= WS)<< 30);
      int wpos = wbeg;
      for (int i = wbeg + lane; __any_sync(Warp, i < wend); i += WS) {
        int wnei;
        bool prio = false;
        if (i < wend) {
          wnei = nlist[i];
          const int wdegn = nidx[wnei + 1] - nidx[wnei];
          unsigned int wnpriority = priority[wnei]|((wdegn >= WS)<< 30);
          unsigned int hash_wv = hash(wv);
          unsigned int hash_wnei = hash(wnei);

          prio = ((wvpriority < wnpriority) || ((wvpriority == wnpriority) && (hash_wv < hash_wnei)) || ((wvpriority == wnpriority) && (hash_wv == hash_wnei) && (wv < wnei)));
        }
        const int b = __ballot_sync(Warp, prio);
        const int offs = __popc(b & ((1 << lane) - 1));
        if (prio) nlist2[wpos + offs] = wnei;
        wpos += __popc(b);
      }
      if (who == lane) pos = wpos;
    }

    if (v < nodes) {
      const int range = pos - beg;
      maxrange = max(maxrange, range);
      color[v] = (cond || (range == 0)) ? (range << (WS / 2)) : active;
      posscol[v] = (range >= WS) ? -1 : (MSB >> range);
    }
  }
  if (maxrange >= Mask) {printf("too many active neighbors\n"); asm("trap;");}

  for (int i = thread; i < edges / WS + 1; i += threads) posscol2[i] = -1;
}


static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void runLarge(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ posscol, int* const __restrict__ posscol2, volatile int* const __restrict__ color, const int* const __restrict__ wl)
{
  const int stop = wlsize;
  if (stop != 0) {
    const int lane = threadIdx.x % WS;
    const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    const int threads = gridDim.x * ThreadsPerBlock;
    bool again;
    do {
      again = false;
      for (int w = thread; __any_sync(Warp, w < stop); w += threads) {
        bool shortcut, done, cond = false;
        int v, data, range, beg, pcol;
        if (w < stop) {
          v = wl[w];
          data = color[v];
          range = data >> (WS / 2);
          if (range > 0) {
            beg = nidx[v];
            pcol = posscol[v];
            cond = true;
          }
        }

        int bal = __ballot_sync(Warp, cond);
        while (bal != 0) {
          const int who = __ffs(bal) - 1;
          bal &= bal - 1;
          const int wdata = __shfl_sync(Warp, data, who);
          const int wrange = wdata >> (WS / 2);
          const int wbeg = __shfl_sync(Warp, beg, who);
          const int wmincol = wdata & Mask;
          const int wmaxcol = wmincol + wrange;
          const int wend = wbeg + wmaxcol;
          const int woffs = wbeg / WS;
          int wpcol = __shfl_sync(Warp, pcol, who);

          bool wshortcut = true;
          bool wdone = true;
          for (int i = wbeg + lane; __any_sync(Warp, i < wend); i += WS) {
            int nei, neidata, neirange;
            if (i < wend) {
              nei = nlist[i];
              neidata = color[nei];
              neirange = neidata >> (WS / 2);
              const bool neidone = (neirange == 0);
              wdone &= neidone; 
              if (neidone) {
                const int neicol = neidata;
                if (neicol < WS) {
                  wpcol &= ~((unsigned int)MSB >> neicol);
                } else {
                  if ((wmincol <= neicol) && (neicol < wmaxcol) && ((posscol2[woffs + neicol / WS] << (neicol % WS)) < 0)) {
                    atomicAnd((int*)&posscol2[woffs + neicol / WS], ~((unsigned int)MSB >> (neicol % WS)));
                  }
                }
              } else {
                const int neimincol = neidata & Mask;
                const int neimaxcol = neimincol + neirange;
                if ((neimincol <= wmincol) && (neimaxcol >= wmincol)) wshortcut = false; 
              }
            }
          }
          wshortcut = __all_sync(Warp, wshortcut);
          wdone = __all_sync(Warp, wdone);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 1);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 2);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 4);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 8);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 16);
          if (who == lane) pcol = wpcol;
          if (who == lane) done = wdone;
          if (who == lane) shortcut = wshortcut;
        }

        if (w < stop) {
          if (range > 0) {
            const int mincol = data & Mask;
            int val = pcol, mc = 0;
            if (pcol == 0) {
              const int offs = beg / WS;
              mc = max(1, mincol / WS);
              while ((val = posscol2[offs + mc]) == 0) mc++;
            }
            int newmincol = mc * WS + __clz(val);
            if (mincol != newmincol) shortcut = false;
            if (shortcut || done) {
              pcol = (newmincol < WS) ? ((unsigned int)MSB >> newmincol) : 0;
            } else {
              const int maxcol = mincol + range;
              const int range = maxcol - newmincol;
              newmincol = (range << (WS / 2)) | newmincol;
              again = true;
            }
            posscol[v] = pcol;
            color[v] = newmincol;
          }
        }
      }
    } while (__any_sync(Warp, again));
  }
}


static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void runSmall(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, volatile int* const __restrict__ posscol, int* const __restrict__ color)
{
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;

  if (thread == 0) wlsize = 0;

  bool again;
  do {
    again = false;
    for (int v = thread; v < nodes; v += threads) {
      __syncthreads();  // optional
      int pcol = posscol[v];
      if (__popc(pcol) > 1) {
        const int beg = nidx[v];
        int active = color[v];
        int allnei = 0;
        int keep = active;
        do {
          const int old = active;
          active &= active - 1;
          const int curr = old ^ active;
          const int i = beg + __clz(curr);
          const int nei = nlist[i];
          const int neipcol = posscol[nei];
          allnei |= neipcol;
          if ((pcol & neipcol) == 0) {
            pcol &= pcol - 1;
            keep ^= curr;
          } else if (__popc(neipcol) == 1) {
            pcol ^= neipcol;
            keep ^= curr;
          }
        } while (active != 0);
        if (keep != 0) {
          const int best = (unsigned int)MSB >> __clz(pcol);
          if ((best & ~allnei) != 0) {
            pcol = best;
            keep = 0;
          }
        }
        again |= keep;
        if (keep == 0) keep = __clz(pcol);
        color[v] = keep;
        posscol[v] = pcol;
      }
    }
  } while (again);
}

// *=======================[HOST]======================*
// =====================================
// ||                                 ||
// ||           RANDOM VAL            ||
// ||                                 ||
// =====================================
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
  
// =====================================
// ||                                 ||
// ||         OPENMIPS RUN PAR        ||
// ||                                 ||
// =====================================

void runBoundaryGC(
    int deviceID,
    const ECLgraph &g,
    unsigned int** partialColor_d_out,
    unsigned int** partialPosColor_d_out,
    unsigned int** partialIterList_d_out 
){
    cudaSetDevice(deviceID);
    cudaGetLastError();
    GPUState* d_state = nullptr;
    cudaMalloc(&d_state, sizeof(GPUState));

    GPUState h_state;
    h_state.remove_size=0;
    h_state.worker      = 0;
    h_state.theta       = 1;
    h_state.iteration   = 0;
    h_state.g_minDegree = INT_MAX;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);  
    const int SMs = deviceProp.multiProcessorCount;
    const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
    const int blocks = SMs * mTpSM / ThreadsPerBlock;
    cudaMemcpy(d_state, &h_state, sizeof(GPUState), cudaMemcpyHostToDevice);
    int *d_nidx = nullptr, *d_nlist = nullptr;
    unsigned int *d_degree = nullptr, *d_iterList = nullptr;

    cudaMalloc(&d_nidx,     (g.nodes + 1)*sizeof(int));
    cudaMalloc(&d_nlist,    g.edges*sizeof(int));
    cudaMalloc(&d_degree,   g.nodes*sizeof(unsigned int));
    cudaMalloc(&d_iterList, g.nodes*sizeof(unsigned int));
    int *nlist2_d, *posscol_d, *posscol2_d, *color_d, *wl_d;
    if (cudaSuccess != cudaMalloc((void **)&nlist2_d, g.edges * sizeof(int))) {printf("ERROR: could not allocate nlist2_d\n\n");  exit(-1);}
    if (cudaSuccess != cudaMalloc((void **)&posscol_d, g.nodes * sizeof(int))) {printf("ERROR: could not allocate posscol_d\n\n");  exit(-1);}
    if (cudaSuccess != cudaMalloc((void **)&posscol2_d, (g.edges / WS + 1) * sizeof(int))) {printf("ERROR: could not allocate posscol2_d\n\n");  exit(-1);}
    if (cudaSuccess != cudaMalloc((void **)&color_d, g.nodes * sizeof(int))) {printf("ERROR: could not allocate color_d\n\n");  exit(-1);}
    if (cudaSuccess != cudaMalloc((void **)&wl_d, g.nodes * sizeof(int))) {printf("ERROR: could not allocate wl_d\n\n");  exit(-1);}
    cudaMemcpy(d_nidx,  g.nindex, (g.nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nlist, g.nlist,   g.edges   *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_iterList, 0, g.nodes*sizeof(unsigned int));
    {
        init_degree<<<ceil(g.nodes/512.0),512>>>(g.nodes, d_nidx, d_nlist, d_degree);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    {
      int h_nodes = g.nodes;
      int* d_nodes;
      cudaMalloc(&d_nodes, sizeof(int));
      cudaMemcpy(d_nodes, &h_nodes, sizeof(int), cudaMemcpyHostToDevice);
      int* remove_list;
      cudaMalloc(&remove_list, g.nodes*sizeof(int));
      cudaMemset(remove_list, 0, g.nodes*sizeof(int));  
        printf("h_nodes %d \n",h_nodes);
        int blkPerSM;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM,
          CHROMA, ThreadsPerBlock, 0);
        int gridDim = blkPerSM * SMs;
        int *out_d;  cudaMalloc(&out_d, g.nodes * sizeof(int));
        void* args[] = {
          &h_nodes,  
          &d_nidx,   
          &d_nlist,  
          &d_degree,
          &d_iterList,
          &d_state,
          &remove_list
      };
        cudaLaunchCooperativeKernel(
                (void*)CHROMA,
                dim3(gridDim), dim3(ThreadsPerBlock), args);
                
        }
    cudaDeviceSynchronize();
    cudaMemcpy(&h_state, d_state, sizeof(GPUState), cudaMemcpyDeviceToHost);
    std::cout << "[GPU " << deviceID << "] final iteration = "
              << h_state.iteration << ", final theta = " << h_state.theta
               << std::endl;
    init<<<blocks, ThreadsPerBlock>>>(g.nodes, g.edges, d_nidx, d_nlist, nlist2_d, posscol_d, posscol2_d, color_d, wl_d,d_iterList);
    cudaDeviceSynchronize();
    printf("INIT FINISH\n");

    runLarge<<<blocks, ThreadsPerBlock>>>(g.nodes, d_nidx, nlist2_d, posscol_d, posscol2_d, color_d, wl_d);
    cudaDeviceSynchronize();
    printf("RUN LARGE FINISH\n");

    runSmall<<<blocks, ThreadsPerBlock>>>(g.nodes, d_nidx, d_nlist, posscol_d, color_d);
    cudaDeviceSynchronize();
    printf("RUN SMALL FINISH\n");
    {
        std::vector<unsigned int> hostColor(g.nodes);
        cudaMemcpy(hostColor.data(), color_d, g.nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);

        std::vector<unsigned int> hostPosColor(g.nodes);
        cudaMemcpy(hostPosColor.data(), posscol_d, g.nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
        std::vector<unsigned int> hostIterList(g.nodes);
        cudaMemcpy(hostIterList.data(), d_iterList, g.nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        *partialColor_d_out = new unsigned int[g.nodes];        
        memcpy(*partialColor_d_out, hostColor.data(), g.nodes*sizeof(unsigned int));
        *partialPosColor_d_out = new unsigned int[g.nodes];
        memcpy(*partialPosColor_d_out, hostPosColor.data(), g.nodes*sizeof(unsigned int));
        *partialIterList_d_out = new unsigned int[g.nodes];
        memcpy(*partialIterList_d_out, hostIterList.data(), g.nodes*sizeof(unsigned int));

    }
    cudaFree(d_state);
    cudaFree(d_nidx);
    cudaFree(d_nlist);
    cudaFree(d_degree);
}

void runOnGPU(
  int deviceID,
  const ECLgraph &g,
  std::vector<int> &boundaryToLocal,
  unsigned int** partialColor_d_out,
  unsigned int** partialIterList_d_out,
  unsigned int* boundaryColor_d,
  unsigned int* boundaryPosColor_d

){
  cudaSetDevice(deviceID);

  cudaGetLastError(); 
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("GPU %d ERROR: cudaSetDevice failed: %s\n", deviceID, cudaGetErrorString(err));
      exit(-1);
  }
  int boundarySize = boundaryToLocal.size();

  GPUState* d_state = nullptr;
  cudaMalloc(&d_state, sizeof(GPUState));

  GPUState h_state;
  h_state.worker      = 0;
  h_state.theta       = 1;
  h_state.iteration   = 0;
  h_state.g_minDegree = INT_MAX;
  h_state.remove_size = 0;

  cudaMemcpy(d_state, &h_state, sizeof(GPUState), cudaMemcpyHostToDevice);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);  
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * mTpSM / ThreadsPerBlock;

  cudaMemcpy(d_state, &h_state, sizeof(GPUState), cudaMemcpyHostToDevice);

  int *d_nidx = nullptr, *d_nlist = nullptr;
  unsigned int *d_degree = nullptr, *d_iterList = nullptr;

  cudaMalloc(&d_nidx,     (g.nodes + 1)*sizeof(int));
  cudaMalloc(&d_nlist,    g.edges*sizeof(int));
  cudaMalloc(&d_degree,   g.nodes*sizeof(unsigned int));
  cudaMalloc(&d_iterList, g.nodes*sizeof(unsigned int));
  
  int *nlist2_d, *posscol_d, *posscol2_d, *color_d, *wl_d;

  cudaError_t err2 = cudaMalloc((void **)&nlist2_d, g.edges * sizeof(int));
  if (err2 != cudaSuccess) {
      printf("GPU %d ERROR: cudaMalloc(nlist2_d) failed: %s\n", deviceID, cudaGetErrorString(err2));
      exit(-1);
  }
  if (cudaSuccess != cudaMalloc((void **)&posscol_d, g.nodes * sizeof(int))) {printf("ERROR: could not allocate posscol_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&posscol2_d, (g.edges / WS + 1) * sizeof(int))) {printf("ERROR: could not allocate posscol2_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&color_d, g.nodes * sizeof(int))) {printf("ERROR: could not allocate color_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&wl_d, g.nodes * sizeof(int))) {printf("ERROR: could not allocate wl_d\n\n");  exit(-1);}

  cudaMemcpy(d_nidx,  g.nindex, (g.nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nlist, g.nlist,   g.edges   *sizeof(int), cudaMemcpyHostToDevice);

  cudaMemset(d_iterList, 0, g.nodes*sizeof(unsigned int));

  {
      init_degree<<<ceil(g.nodes/512.0),512>>>(g.nodes, d_nidx, d_nlist, d_degree);
      cudaDeviceSynchronize();
  }

  cudaDeviceSynchronize();
  {
    int h_nodes = g.nodes;
    int* d_nodes;
    cudaMalloc(&d_nodes, sizeof(int));
    cudaMemcpy(d_nodes, &h_nodes, sizeof(int), cudaMemcpyHostToDevice);
    int* remove_list;
    cudaMalloc(&remove_list, g.nodes*sizeof(int));
    cudaMemset(remove_list, 0, g.nodes*sizeof(int));  
      int blkPerSM;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM,
        CHROMA, ThreadsPerBlock, 0);
      int gridDim = blkPerSM * SMs;  
      int *out_d;  cudaMalloc(&out_d, g.nodes * sizeof(int));
      void* args[] = {
        &h_nodes,  
        &d_nidx,   
        &d_nlist,  
        &d_degree,
        &d_iterList,
        &d_state,
        &remove_list
    };
      cudaLaunchCooperativeKernel(
              (void*)CHROMA,
              dim3(gridDim), dim3(ThreadsPerBlock), args);
              
      }
  cudaDeviceSynchronize();

  int* d_boundaryToLocal = nullptr;
  cudaMalloc(&d_boundaryToLocal, boundarySize * sizeof(int));
  cudaMemcpy(d_boundaryToLocal,
             boundaryToLocal.data(),
             boundarySize*sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMemcpy(&h_state, d_state, sizeof(GPUState), cudaMemcpyDeviceToHost);
  applyBoundaryPrio<<<blocks, ThreadsPerBlock>>>(
    boundarySize,
    d_boundaryToLocal,
    d_iterList
  );
  cudaDeviceSynchronize();
          
  init<<<blocks, ThreadsPerBlock>>>(g.nodes, g.edges, d_nidx, d_nlist, nlist2_d, posscol_d, posscol2_d, color_d, wl_d,d_iterList);
  cudaDeviceSynchronize();
  unsigned int *boundaryColor_d_device = nullptr;
  unsigned int *boundaryPosColor_d_device = nullptr;
  cudaMalloc(&boundaryColor_d_device, boundarySize * sizeof(unsigned int));
  cudaMemcpy(boundaryColor_d_device, boundaryColor_d, boundarySize * sizeof(unsigned int), cudaMemcpyHostToDevice);

  cudaMalloc(&boundaryPosColor_d_device, boundarySize * sizeof(unsigned int));
  cudaMemcpy(boundaryPosColor_d_device, boundaryPosColor_d, boundarySize * sizeof(unsigned int), cudaMemcpyHostToDevice);

  applyBoundaryColorToSubgraph<<<blocks, ThreadsPerBlock>>>(
    boundarySize,
    d_boundaryToLocal,
    boundaryColor_d_device,
    (unsigned int*) color_d,
    boundaryPosColor_d_device,
    (unsigned int*) posscol_d
  );
  cudaDeviceSynchronize();

  runLarge<<<blocks, ThreadsPerBlock>>>(g.nodes, d_nidx, nlist2_d, posscol_d, posscol2_d, color_d, wl_d);
  cudaDeviceSynchronize();

  runSmall<<<blocks, ThreadsPerBlock>>>(g.nodes, d_nidx, d_nlist, posscol_d, color_d);
  cudaDeviceSynchronize();
  {
      std::vector<unsigned int> hostColor(g.nodes);
      cudaMemcpy(hostColor.data(), color_d, g.nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);
      std::vector<unsigned int> hostIterList(g.nodes);
      cudaMemcpy(hostIterList.data(), d_iterList, g.nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);

      *partialColor_d_out = new unsigned int[g.nodes];
      memcpy(*partialColor_d_out, hostColor.data(), g.nodes*sizeof(unsigned int));
      *partialIterList_d_out = new unsigned int[g.nodes];
      memcpy(*partialIterList_d_out, hostIterList.data(), g.nodes*sizeof(unsigned int));

  }

  cudaFree(d_state);
  cudaFree(d_nidx);
  cudaFree(d_nlist);
  cudaFree(d_degree);
  cudaFree(boundaryColor_d_device);
  cudaFree(boundaryPosColor_d_device);
}


void createSubgraphWithCopies(
  const ECLgraph &g,
  const std::vector<int> &partition_nodes,
  const std::vector<int> &copied_nodes,  
  std::vector<int> &sub_nindex,
  std::vector<int> &sub_nlist,
  std::vector<int> &localToGlobal,
  std::vector<int> &copiedMapping  
) {
  std::map<int, int> globalToLocal;
  int localIdx = 0;

  copiedMapping.resize(copied_nodes.size(), -1);

  for (auto node : partition_nodes) {
      if (globalToLocal.find(node) == globalToLocal.end()) {
          globalToLocal[node] = localIdx++;
          localToGlobal.push_back(node);
      }
  }

  for (size_t i = 0; i < copied_nodes.size(); i++) {
      int node = copied_nodes[i];
      int localId;

      auto it = globalToLocal.find(node);
      if (it == globalToLocal.end()) {
          localId = localIdx++;
          globalToLocal[node] = localId;
          localToGlobal.push_back(node);
      } else {
          localId = it->second;
      }
      
      copiedMapping[i] = localId;
  }

  sub_nindex.resize(globalToLocal.size() + 1, 0);
  int edge_count = 0;

  for (int i = 0; i < (int)localToGlobal.size(); i++) {
      int globalNode = localToGlobal[i];
      for (int j = g.nindex[globalNode]; j < g.nindex[globalNode + 1]; j++) {
          int neighbor = g.nlist[j];
          if (globalToLocal.find(neighbor) != globalToLocal.end()) {
              sub_nlist.push_back(globalToLocal[neighbor]);
              edge_count++;
          }
      }
      sub_nindex[i + 1] = edge_count;
  }
}

// =====================================
// ||                                 ||
// ||      createMergeWithCopies      ||
// ||                                 ||
// =====================================
void createMergeWithCopies(
  const ECLgraph &g,
  const std::set<int> &copied_nodes,
  std::vector<int> &sub_nindex,
  std::vector<int> &sub_nlist,
  std::vector<int> &localToGlobal
) {
  std::map<int, int> globalToLocal;
  int localIdx = 0;

  for (auto node : copied_nodes) {
      if (globalToLocal.find(node) == globalToLocal.end()) {
          globalToLocal[node] = localIdx++;
          localToGlobal.push_back(node);
      }
  }

  sub_nindex.resize(globalToLocal.size() + 1, 0);
  int edge_count = 0;

  for (int i = 0; i < (int)localToGlobal.size(); i++) {
      int globalNode = localToGlobal[i];
      for (int j = g.nindex[globalNode]; j < g.nindex[globalNode + 1]; j++) {
          int neighbor = g.nlist[j];
          if (globalToLocal.find(neighbor) != globalToLocal.end()) {
              sub_nlist.push_back(globalToLocal[neighbor]);
              edge_count++;
          }
      }
      sub_nindex[i + 1] = edge_count;  
  }
}

int main(int argc, char* argv[]) {
    // ========================================PRE========================================
    std::string filename;
    int fuzzy_number = 10;
  
    std::cout << "Enter input filename: ";
    std::cin >> filename;
    std::cin.ignore();
    std::string input_line;

    std::cout << "Enter RGC (default \u03B8=10): ";
    std::getline(std::cin, input_line);
    if (!input_line.empty()) {
      fuzzy_number = std::stoi(input_line);
    }
    std::cout << "Using RGC \u03B8 = " << fuzzy_number << std::endl;
    int nParts=2;
    std::cout << "nPart: ";
    std::getline(std::cin, input_line);
    if (!input_line.empty()) {
      nParts = std::stoi(input_line);
    }
    std::cout << "Devide to = " << nParts <<"graphs"<< std::endl;


    ECLgraph g = readECLgraph(filename.c_str());

    if (nParts < 1) {
        printf("ERROR: nParts must be >= 1\n");
        exit(-1);
    }
    printf("input: %s\n", filename.c_str());
    printf("nodes: %d\n", g.nodes);
    printf("edges: %d\n", g.edges);
    printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

    int* const color = new int [g.nodes];
    setParameters<<<1, 1>>>(fuzzy_number);
  
    cudaSetDevice(Device);
    cudaGetLastError(); // 立刻 check，有錯就 exit
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
    const int SMs = deviceProp.multiProcessorCount;
    const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  
    // ========================================METIS========================================
    idx_t nVertices = g.nodes;
    idx_t ncon      = 1;
    idx_t objval;
    std::vector<idx_t> part(nVertices, 0); 

    std::vector<real_t> tpwgts(nParts * ncon, 1.0 / (real_t)nParts);
    std::vector<real_t> ubvec(ncon, 1.05);

    int ret = METIS_PartGraphKway(
        &nVertices,
        &ncon,
        g.nindex,
        g.nlist,
        nullptr, // vwgt
        nullptr, // vsize
        nullptr, // adjwgt
        (idx_t*)&nParts,
        tpwgts.data(),
        ubvec.data(),
        nullptr, // options
        &objval,
        part.data()
    );
    if (ret != METIS_OK) {
        printf("METIS partition fail, code=%d\n", ret);
        exit(-1);
    }
    // -----------------------------------------------------------------------
    //  Collect partitioned nodes & boundary
    // -----------------------------------------------------------------------
    // Partition i => partition_nodes[i]
    std::vector<std::vector<int>> partition_nodes(nParts);
    for (int v = 0; v < g.nodes; v++) {
        partition_nodes[ part[v] ].push_back(v);
    }

    // Build global boundary set: any vertex that has an edge crossing partitions
    std::set<int> boundary_nodes;
    for (int v = 0; v < g.nodes; v++) {
        int pv = part[v];
        for (int j = g.nindex[v]; j < g.nindex[v+1]; j++) {
            int nei = g.nlist[j];
            int pn  = part[nei];
            if (pn != pv) {
                boundary_nodes.insert(v);
                boundary_nodes.insert(nei);
            }
        }
    }
    std::vector<int> boundary_nodes_vec(boundary_nodes.begin(), boundary_nodes.end());
    printf("Boundary node count = %zu\n", boundary_nodes_vec.size());
    // -----------------------------------------------------------------------
    //  Build subgraphs for each partition, plus a “boundary graph”
    // -----------------------------------------------------------------------
    // We store subgraph data in arrays of size nParts
    std::vector<ECLgraph> subGraphs(nParts);
    std::vector<std::vector<int>> sub_nindex(nParts), sub_nlist(nParts);
    std::vector<std::vector<int>> sub_localToGlobal(nParts);
    std::vector<std::vector<int>> sub_boundaryToLocal(nParts);

    // Single boundary graph
    std::vector<int> boundary_nindex, boundary_nlist, boundary_localToGlobal;
    // Create subgraphs
    for (int i = 0; i < nParts; i++) {
      createSubgraphWithCopies(
          g,
          partition_nodes[i],      // partition i's nodes
          boundary_nodes_vec,      // all boundary nodes
          sub_nindex[i],
          sub_nlist[i],
          sub_localToGlobal[i],
          sub_boundaryToLocal[i]   // boundary->local mapping
      );
      subGraphs[i].nodes  = (int)sub_localToGlobal[i].size();
      subGraphs[i].edges  = (int)sub_nlist[i].size();
      subGraphs[i].nindex = sub_nindex[i].data();
      subGraphs[i].nlist  = sub_nlist[i].data();
      printf("Partition %d => subgraph has %d nodes, %d edges\n",
             i, subGraphs[i].nodes, subGraphs[i].edges);
  }
    // Build a single boundary graph
    createMergeWithCopies(
      g,
      boundary_nodes,     // pass as set<int>
      boundary_nindex,
      boundary_nlist,
      boundary_localToGlobal
    );
    ECLgraph boundaryGraph;
    boundaryGraph.nodes  = (int)boundary_localToGlobal.size();
    boundaryGraph.edges  = (int)boundary_nlist.size();
    boundaryGraph.nindex = boundary_nindex.data();
    boundaryGraph.nlist  = boundary_nlist.data();
    printf("Boundary graph has %d nodes, %d edges\n",
           boundaryGraph.nodes, boundaryGraph.edges);
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
    cudaSetDevice(Device);
    cudaGetLastError(); 
    unsigned int* d_color_list     = nullptr;
    unsigned int* d_iteration_list = nullptr;
    cudaMalloc(&d_color_list,     g.nodes*sizeof(unsigned int));
    cudaMalloc(&d_iteration_list, g.nodes*sizeof(unsigned int));
    cudaMemset(d_color_list,     0, g.nodes*sizeof(unsigned int));
    cudaMemset(d_iteration_list, 0, g.nodes*sizeof(unsigned int));
    for (int i = 0; i < nParts; i++) {
      int subSize = (int)sub_localToGlobal[i].size();
      int* d_localToGlobal = nullptr;
      cudaMalloc(&d_localToGlobal, subSize * sizeof(int));
      cudaMemcpy(d_localToGlobal,
                 sub_localToGlobal[i].data(),
                 subSize * sizeof(int),
                 cudaMemcpyHostToDevice);
  
      unsigned int* d_partialColor = nullptr;
      cudaMalloc(&d_partialColor, subSize * sizeof(unsigned int));
      cudaMemcpy(d_partialColor,
                 partialColorVec[i],
                 subSize * sizeof(unsigned int),
                 cudaMemcpyHostToDevice);
  
      unsigned int* d_partialIter = nullptr;
      cudaMalloc(&d_partialIter, subSize * sizeof(unsigned int));
      cudaMemcpy(d_partialIter,
                 partialIterVec[i],
                 subSize * sizeof(unsigned int),
                 cudaMemcpyHostToDevice);
  
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
  
    const float runtime = timer.stop();
  
    printf("runtime:    %.6f ms\n", runtime*1000);
    printf("throughput: %.6f Mnodes/s\n", g.nodes * 0.000001 / runtime);
    printf("throughput: %.6f Medges/s\n", g.edges * 0.000001 / runtime);
    if (cudaSuccess != cudaMemcpy(color, d_color_list, g.nodes * sizeof(int), cudaMemcpyDeviceToHost)) {printf("ERROR: copying color from device failed\n\n");  exit(-1);}

    for (int v = 0; v < g.nodes; v++) {
      if (color[v] < 0) {printf("ERROR: found unprocessed node in graph (node %d with deg %d)\n\n", v, g.nindex[v + 1] - g.nindex[v]);  exit(-1);}
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (color[g.nlist[i]] == color[v]&&g.nlist[i]!=v) {printf("ERROR: found adjacent nodes with same color %d (%d %d)\n\n", color[v], v, g.nlist[i]);exit(-1);  }
      }
    }

    printf("result verification passed\n"); 
    const int vals = 16;
    int c[vals];
    for (int i = 0; i < vals; i++) c[i] = 0;
    int cols = -1;
    for (int v = 0; v < g.nodes; v++) {
      cols = std::max(cols, color[v]);
      if (color[v] < vals) c[color[v]]++;
    }
    cols++;
    printf("colors used: %d\n", cols);

    delete [] color;

    cudaFree(d_color_list);
    freeECLgraph(g);  // 釋放圖資源
    return 0;
}

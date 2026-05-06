#include "globals.cuh"
#include <cuda.h>
#include <stdio.h>

__global__ void P_SL_ELS_SDC(
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
// int add_num=0;
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
      // add_num++;
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
// if (add_num!=0) atomicAdd(&worker, add_num);

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
          warpMin = min(warpMin, atomicSub(&degree_list[nei], 1) - 1);
      }
  }
    warpMin = warpReduceMin(warpMin);
    if (lane == 0 && warpMin < 0x7fffffff){
      atomicMin(&g_minDegree, warpMin);
    }
}

grid.sync();

if(grid.thread_rank()==0){
  worker+=remove_size;
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

__global__ void P_SL_ELS(
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
// int add_num=0;
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
      // add_num++;
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
// if (add_num!=0) atomicAdd(&worker, add_num);

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
  worker+=remove_size;
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


// ═══════════════════════════════════════════════════════════
//  Fused PA + CA-init kernel
//  Runs the full PA loop first, then immediately executes
//  the CA init pass (identical to ECLGC init()) using the
//  final iteration_list — all inside one cooperative kernel
//  to save a kernel launch and benefit from cache locality.
// ═══════════════════════════════════════════════════════════
__global__ void P_SL_ELS_FUSED(
    const int  nodes,
    const int  edges,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    int* __restrict__ nlist2,
    int* __restrict__ posscol,
    int* __restrict__ posscol2,
    int* __restrict__ color,
    int* __restrict__ wl,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
const int threads = gridDim.x * blockDim.x;
const int lane    = threadIdx.x & 31;
const int warpId  = tid >> 5;
const int numWarp = threads >> 5;
cg::grid_group grid = cg::this_grid();

// ════════════════════════════════════════════════════════════
//  Stage 1: PA — identical to P_SL_ELS
// ════════════════════════════════════════════════════════════
do {
unsigned int localMin = 0x7FFFFFFF;
for (int v = tid; v < nodes; v += threads) {
  int iteration_list_v = iteration_list[v];
  unsigned int prio = (iteration_list_v >> 30) & 0x1;
  unsigned int iteration_v = (iteration_list_v) & 0x3FFFFFFF;
  unsigned int large_deg = 0;
  unsigned int degree = degree_list[v];
  if (prio == 0) {
    if (degree <= (theta + FuzzyNumber)) {
      prio = 1;
      int beg = nidx[v];
      int end = nidx[v + 1];
      if ((end - beg) >= WS) large_deg = 1;
      iteration_list[v] = (large_deg << 31) | (prio << 30)
                        | (iteration_v + (degree - theta) + 1);
      remove_list[atomicAdd(&remove_size, 1)] = v;
    } else {
      if (degree < localMin) localMin = degree;
      iteration_list[v] = (large_deg << 31) | (prio << 30)
                        | (iteration_v + FuzzyNumber + 1);
    }
  }
}
if (localMin < 0x7FFFFFFF) atomicMin(&g_minDegree, localMin);

grid.sync();
for (int k = warpId; k < remove_size; k += numWarp) {
  int v = remove_list[k];
  int beg, end;
  if (lane == 0) {
    beg = nidx[v];
    end = nidx[v + 1];
  }
  beg = __shfl_sync(0xffffffff, beg, 0);
  end = __shfl_sync(0xffffffff, end, 0);
  for (int i = beg + lane; i < end; i += 32) {
    int nei      = __ldg(nlist + i);
    unsigned int iter = __ldg(iteration_list + nei);
    if (!(iter & 0x40000000u)) {
      atomicSub(&degree_list[nei], 1);
    }
  }
}

grid.sync();

if (grid.thread_rank() == 0) {
  worker += remove_size;
  remove_size = 0;
  theta = g_minDegree;
  atomicExch(&g_minDegree, 0x7FFFFFFF);
  iteration = iteration + 1 + FuzzyNumber;
  #ifdef PROFILE
  iter_count++;
  #endif
}
grid.sync();
} while (worker != nodes);

// ════════════════════════════════════════════════════════════
//  Stage 2: CA init — identical to ECLGC init()
//  iteration_list is now final for all vertices.
// ════════════════════════════════════════════════════════════
grid.sync();

int maxrange = -1;
for (int v = tid; __any_sync(Warp, v < nodes); v += threads) {
  bool cond = false;
  int beg, end, pos, degv, active;
  if (v < nodes) {
    beg = nidx[v];
    end = nidx[v + 1];
    degv = end - beg;
    unsigned int v_priority = iteration_list[v];

    cond = (degv >= WS);
    if (cond) {
      wl[atomicAdd(&wlsize, 1)] = v;
    } else {
      active = 0;
      pos = beg;
      for (int i = beg; i < end; i++) {
        const int nei = nlist[i];
        const int degn = nidx[nei + 1] - nidx[nei];
        const unsigned int priority_n = iteration_list[nei];
        unsigned int hash_v = hash(v);
        unsigned int hash_nei = hash(nei);

        if ((degn >= WS) || (v_priority < priority_n) ||
            ((v_priority == priority_n) && (degv < degn)) ||
            ((v_priority == priority_n) && (degv == degn) && (hash_v < hash_nei)) ||
            ((v_priority == priority_n) && (degv == degn) && (hash_v == hash_nei) && (v < nei))) {
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
    unsigned int wvpriority = iteration_list[wv];
    int wpos = wbeg;
    for (int i = wbeg + lane; __any_sync(Warp, i < wend); i += WS) {
      int wnei;
      bool prio = false;
      if (i < wend) {
        wnei = nlist[i];
        const int wdegn = nidx[wnei + 1] - nidx[wnei];
        unsigned int wnpriority = iteration_list[wnei];
        unsigned int hash_wv = hash(wv);
        unsigned int hash_wnei = hash(wnei);

        prio = (wdegn >= WS && (((wvpriority < wnpriority) ||
          ((wvpriority == wnpriority) && (wdegv < wdegn)) ||
          ((wvpriority == wnpriority) && (wdegv == wdegn) && (hash_wv < hash_wnei)) ||
          ((wvpriority == wnpriority) && (wdegv == wdegn) && (hash_wv == hash_wnei) && (wv < wnei)))));
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
if (maxrange >= Mask) { printf("too many active neighbors\n"); asm("trap;"); }

for (int i = tid; i < edges / WS + 1; i += threads) posscol2[i] = -1;
}


// ═══════════════════════════════════════════════════════════
//  P_SL_ELS_SDC_FUSED: Post-loop fusion (SDC variant)
//  PA uses SDC's warpReduceMin for tighter theta tracking,
//  then runs CA init after PA loop completes.
// ═══════════════════════════════════════════════════════════
__global__ void P_SL_ELS_SDC_FUSED(
    const int  nodes,
    const int  edges,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    int* __restrict__ nlist2,
    int* __restrict__ posscol,
    int* __restrict__ posscol2,
    int* __restrict__ color,
    int* __restrict__ wl,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
const int threads = gridDim.x * blockDim.x;
const int lane    = threadIdx.x & 31;
const int warpId  = tid >> 5;
const int numWarp = threads >> 5;
cg::grid_group grid = cg::this_grid();

// ════════════════════════════════════════════════════════════
//  Stage 1: PA — identical to P_SL_ELS_SDC
// ════════════════════════════════════════════════════════════
do {
unsigned int localMin = 0x7FFFFFFF;
for (int v = tid; v < nodes; v += threads) {
  int iteration_list_v = iteration_list[v];
  unsigned int prio = (iteration_list_v >> 30) & 0x1;
  unsigned int iteration_v = (iteration_list_v) & 0x3FFFFFFF;
  unsigned int large_deg = 0;
  unsigned int degree = degree_list[v];
  if (prio == 0) {
    if (degree <= (theta + FuzzyNumber)) {
      prio = 1;
      int beg = nidx[v];
      int end = nidx[v + 1];
      if ((end - beg) >= WS) large_deg = 1;
      iteration_list[v] = (large_deg << 31) | (prio << 30)
                        | (iteration_v + (degree - theta) + 1);
      remove_list[atomicAdd(&remove_size, 1)] = v;
    } else {
      if (degree < localMin) localMin = degree;
      iteration_list[v] = (large_deg << 31) | (prio << 30)
                        | (iteration_v + FuzzyNumber + 1);
    }
  }
}
if (localMin < 0x7FFFFFFF) atomicMin(&g_minDegree, localMin);

grid.sync();
unsigned int warpMin;

for (int k = warpId; k < remove_size; k += numWarp) {
  int v = remove_list[k];
  int beg, end;
  if (lane == 0) {
    beg = nidx[v];
    end = nidx[v + 1];
  }
  beg = __shfl_sync(0xffffffff, beg, 0);
  end = __shfl_sync(0xffffffff, end, 0);

  warpMin = 0x7fffffff;
  for (int i = beg + lane; i < end; i += 32) {
    int nei      = __ldg(nlist + i);
    unsigned int iter = __ldg(iteration_list + nei);
    if (!(iter & 0x40000000u)) {
      warpMin = min(warpMin, atomicSub(&degree_list[nei], 1) - 1);
    }
  }
  warpMin = warpReduceMin(warpMin);
  if (lane == 0 && warpMin < 0x7fffffff) {
    atomicMin(&g_minDegree, warpMin);
  }
}

grid.sync();

if (grid.thread_rank() == 0) {
  worker += remove_size;
  remove_size = 0;
  theta = g_minDegree;
  atomicExch(&g_minDegree, 0x7FFFFFFF);
  iteration = iteration + 1 + FuzzyNumber;
  #ifdef PROFILE
  iter_count++;
  #endif
}
grid.sync();
} while (worker != nodes);

// ════════════════════════════════════════════════════════════
//  Stage 2: CA init — identical to ECLGC init()
// ════════════════════════════════════════════════════════════
grid.sync();

int maxrange = -1;
for (int v = tid; __any_sync(Warp, v < nodes); v += threads) {
  bool cond = false;
  int beg, end, pos, degv, active;
  if (v < nodes) {
    beg = nidx[v];
    end = nidx[v + 1];
    degv = end - beg;
    unsigned int v_priority = iteration_list[v];

    cond = (degv >= WS);
    if (cond) {
      wl[atomicAdd(&wlsize, 1)] = v;
    } else {
      active = 0;
      pos = beg;
      for (int i = beg; i < end; i++) {
        const int nei = nlist[i];
        const int degn = nidx[nei + 1] - nidx[nei];
        const unsigned int priority_n = iteration_list[nei];
        unsigned int hash_v = hash(v);
        unsigned int hash_nei = hash(nei);

        if ((degn >= WS) || (v_priority < priority_n) ||
            ((v_priority == priority_n) && (degv < degn)) ||
            ((v_priority == priority_n) && (degv == degn) && (hash_v < hash_nei)) ||
            ((v_priority == priority_n) && (degv == degn) && (hash_v == hash_nei) && (v < nei))) {
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
    unsigned int wvpriority = iteration_list[wv];
    int wpos = wbeg;
    for (int i = wbeg + lane; __any_sync(Warp, i < wend); i += WS) {
      int wnei;
      bool prio = false;
      if (i < wend) {
        wnei = nlist[i];
        const int wdegn = nidx[wnei + 1] - nidx[wnei];
        unsigned int wnpriority = iteration_list[wnei];
        unsigned int hash_wv = hash(wv);
        unsigned int hash_wnei = hash(wnei);

        prio = (wdegn >= WS && (((wvpriority < wnpriority) ||
          ((wvpriority == wnpriority) && (wdegv < wdegn)) ||
          ((wvpriority == wnpriority) && (wdegv == wdegn) && (hash_wv < hash_wnei)) ||
          ((wvpriority == wnpriority) && (wdegv == wdegn) && (hash_wv == hash_wnei) && (wv < wnei)))));
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
if (maxrange >= Mask) { printf("too many active neighbors\n"); asm("trap;"); }

for (int i = tid; i < edges / WS + 1; i += threads) posscol2[i] = -1;
}


// ═══════════════════════════════════════════════════════════
//  P_SL_ELS_SDC_FUSED_BATCH: Per-batch fusion (SDC variant)
//  Uses monotonic priority encoding: max(degree-theta, 0)+1
//  so that later-peeled vertices always have higher iteration
//  values, enabling correct per-batch init of peeled vertices.
//  Unpeeled neighbors are guaranteed to have higher final
//  priority, so they are always marked as "higher" in init.
// ═══════════════════════════════════════════════════════════
__global__ void P_SL_ELS_SDC_FUSED_BATCH(
    const int  nodes,
    const int  edges,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    int* __restrict__ nlist2,
    int* __restrict__ posscol,
    int* __restrict__ posscol2,
    int* __restrict__ color,
    int* __restrict__ wl,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
const int threads = gridDim.x * blockDim.x;
const int lane    = threadIdx.x & 31;
const int warpId  = tid >> 5;
const int numWarp = threads >> 5;
cg::grid_group grid = cg::this_grid();

do {
// ── Phase 1: PA peel (monotonic encoding) ────────────────
unsigned int localMin = 0x7FFFFFFF;
for (int v = tid; v < nodes; v += threads) {
  int iteration_list_v = iteration_list[v];
  unsigned int prio = (iteration_list_v >> 30) & 0x1;
  unsigned int iteration_v = (iteration_list_v) & 0x3FFFFFFF;
  unsigned int large_deg = 0;
  unsigned int degree = degree_list[v];
  if (prio == 0) {
    if (degree <= (theta + FuzzyNumber)) {
      prio = 1;
      int beg = nidx[v];
      int end = nidx[v + 1];
      if ((end - beg) >= WS) large_deg = 1;
      // Monotonic fix: clamp (degree - theta) to >= 0
      unsigned int delta = (degree > (unsigned int)theta)
                         ? (degree - (unsigned int)theta) : 0u;
      iteration_list[v] = (large_deg << 31) | (prio << 30)
                        | (iteration_v + delta + 1);
      remove_list[atomicAdd(&remove_size, 1)] = v;
    } else {
      if (degree < localMin) localMin = degree;
      iteration_list[v] = (large_deg << 31) | (prio << 30)
                        | (iteration_v + FuzzyNumber + 1);
    }
  }
}
if (localMin < 0x7FFFFFFF) atomicMin(&g_minDegree, localMin);

grid.sync();

// ── Phase 2: Degree decrement (SDC with warpReduceMin) ───
unsigned int warpMin;
for (int k = warpId; k < remove_size; k += numWarp) {
  int v = remove_list[k];
  int beg, end;
  if (lane == 0) {
    beg = nidx[v];
    end = nidx[v + 1];
  }
  beg = __shfl_sync(0xffffffff, beg, 0);
  end = __shfl_sync(0xffffffff, end, 0);

  warpMin = 0x7fffffff;
  for (int i = beg + lane; i < end; i += 32) {
    int nei      = __ldg(nlist + i);
    unsigned int iter = __ldg(iteration_list + nei);
    if (!(iter & 0x40000000u)) {
      warpMin = min(warpMin, atomicSub(&degree_list[nei], 1) - 1);
    }
  }
  warpMin = warpReduceMin(warpMin);
  if (lane == 0 && warpMin < 0x7fffffff) {
    atomicMin(&g_minDegree, warpMin);
  }
}

grid.sync();

// ── Phase 3: Per-batch init of newly peeled vertices ─────
// Because of monotonic encoding, all unpeeled neighbors are
// guaranteed to have higher final iteration values than any
// vertex peeled so far.  We can therefore correctly init now.
int cur_rs = remove_size;

// Phase 3a: Small-degree vertices (thread-per-vertex)
for (int k = tid; k < cur_rs; k += threads) {
  int v   = remove_list[k];
  int beg = nidx[v];
  int end = nidx[v + 1];
  int degv = end - beg;

  if (degv < WS) {
    unsigned int v_priority = iteration_list[v];
    int active = 0;
    int pos    = beg;
    for (int i = beg; i < end; i++) {
      const int nei  = nlist[i];
      const int degn = nidx[nei + 1] - nidx[nei];
      const unsigned int priority_n = iteration_list[nei];
      const unsigned int nei_peeled = (priority_n >> 30) & 0x1;

      bool is_higher;
      if (nei_peeled == 0) {
        // Unpeeled → monotonic guarantee: will have higher priority
        is_higher = true;
      } else {
        const unsigned int hash_v   = hash(v);
        const unsigned int hash_nei = hash(nei);
        is_higher = (degn >= WS)
          || (v_priority < priority_n)
          || ((v_priority == priority_n) && (degv < degn))
          || ((v_priority == priority_n) && (degv == degn)
              && (hash_v < hash_nei))
          || ((v_priority == priority_n) && (degv == degn)
              && (hash_v == hash_nei) && (v < nei));
      }

      if (is_higher) {
        active |= (unsigned int)MSB >> (i - beg);
        pos++;
      }
    }
    int range = pos - beg;
    color[v]   = (range == 0) ? (range << (WS / 2)) : active;
    posscol[v] = (range >= WS) ? -1 : (MSB >> range);
  } else {
    // Large-degree → add to worklist
    wl[atomicAdd(&wlsize, 1)] = v;
  }
}

// Phase 3b: Large-degree vertices (warp-per-vertex, nlist2 compaction)
for (int k = warpId; k < cur_rs; k += numWarp) {
  int v = remove_list[k];
  int beg, end;
  if (lane == 0) {
    beg = nidx[v];
    end = nidx[v + 1];
  }
  beg = __shfl_sync(0xffffffff, beg, 0);
  end = __shfl_sync(0xffffffff, end, 0);
  int degv = end - beg;

  if (degv >= WS) {
    unsigned int v_priority = iteration_list[v];
    int wpos = beg;
    for (int i = beg + lane; __any_sync(Warp, i < end); i += WS) {
      bool prio_flag = false;
      if (i < end) {
        int cur_nei = nlist[i];
        const int degn = nidx[cur_nei + 1] - nidx[cur_nei];
        const unsigned int priority_n = iteration_list[cur_nei];
        const unsigned int nei_peeled = (priority_n >> 30) & 0x1;

        bool is_higher;
        if (nei_peeled == 0) {
          is_higher = true;
        } else {
          unsigned int hash_v   = hash(v);
          unsigned int hash_nei = hash(cur_nei);
          is_higher = (v_priority < priority_n)
            || ((v_priority == priority_n) && (degv < degn))
            || ((v_priority == priority_n) && (degv == degn)
                && (hash_v < hash_nei))
            || ((v_priority == priority_n) && (degv == degn)
                && (hash_v == hash_nei) && (v < cur_nei));
        }

        // Only keep large-degree higher-priority neighbours in nlist2
        prio_flag = (degn >= WS && is_higher);
      }
      const int b    = __ballot_sync(Warp, prio_flag);
      const int offs = __popc(b & ((1 << lane) - 1));
      if (prio_flag) nlist2[wpos + offs] = nlist[i];
      wpos += __popc(b);
    }
    if (lane == 0) {
      int range = wpos - beg;
      color[v]   = range << (WS / 2);
      posscol[v] = (range >= WS) ? -1 : (MSB >> range);
    }
  }
}

grid.sync();

// ── Phase 4: Update globals ──────────────────────────────
if (grid.thread_rank() == 0) {
  worker += remove_size;
  remove_size = 0;
  theta = g_minDegree;
  atomicExch(&g_minDegree, 0x7FFFFFFF);
  iteration = iteration + 1 + FuzzyNumber;
  #ifdef PROFILE
  iter_count++;
  #endif
}
grid.sync();
} while (worker != nodes);

// ── Post-loop: Initialise posscol2 for runLarge ──────────
for (int i = tid; i < edges / WS + 1; i += threads) posscol2[i] = -1;
}
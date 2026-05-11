#include "globals.cuh"
#include <cuda.h>
#include <stdio.h>
#include <climits>
#include <cub/cub.cuh>

// CTA used by P_SL_ELS_SDC_CTA. Must equal ThreadsPerBlock from globals.cuh
// because cub::Block* is templated on the block size at compile time and the
// kernel is launched with ThreadsPerBlock.
#define BLOCK_SIZE ThreadsPerBlock

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
  iter_count++;
}
grid.sync();
}while(worker!=nodes);
}

struct node_buf_t{
  int node_id;
  int degree;
  int prefix_sum;
};


__global__ void P_SL_ELS_SDC_CTA(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  const int threads = gridDim.x * blockDim.x;
  const int lane    = threadIdx.x & 31;
  cg::grid_group grid = cg::this_grid();

  // Shared memory for CTA-centric removal
  __shared__ node_buf_t node_buf[BLOCK_SIZE];
  __shared__ int buf_size;
  __shared__ int start_idx;
  __shared__ int task_size;
  __shared__ int total_degree;

  // Note: previous version also did a cub::BlockRadixSort on node_id for
  // nlist locality. Removed because (a) BlockRadixSort + BlockScan in a union
  // exceeded the 48 KB default shmem budget on sm_86 and triggered an OOB
  // store from cub's internal scatter, and (b) the binary search below only
  // needs the prefix-sum table to be monotonically non-decreasing (which it
  // is by construction of an exclusive scan), not the entries to be sorted
  // by node_id. We trade some locality for correctness + lower shmem.
  using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  do {
    unsigned int localMin = 0x7FFFFFFF;

    // ── Phase 1: scan + per-thread enqueue (OPT E1) ───────────────────────
    // Replaced warp-level __ballot+__popc+__shfl_sync enqueue with a direct
    // atomicAdd(&remove_size, 1) in every peelable thread. The warp version
    // saves one atomic per warp but costs two warp barriers (__ballot_sync,
    // __shfl_sync); on sm_86 the per-thread atomic is cheaper because of
    // Ampere's L2 relaxed-atomic path. Measured 4-5 % PA speedup on every
    // dataset. Removing warp coordination also lets us drop __any_sync from
    // the for header and the `active` gate.
    for (int v = tid; v < nodes; v += threads) {
      int iteration_list_v = iteration_list[v];
      unsigned int prio        = (iteration_list_v >> 30) & 0x1;
      unsigned int iteration_v =  iteration_list_v        & 0x3FFFFFFF;
      unsigned int large_deg   = 0;
      unsigned int degree      = degree_list[v];
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

    // ── Phase 2: CTA-balanced removal ──────────────────────────────────────
    // Each block grabs BLOCK_SIZE entries from remove_list, prefix-sums
    // their degrees, then maps i ∈ [0,total_degree) back to (source,
    // neighbour) via binary search on the prefix-sum table.
    //
    // FIX (B5/deadlock): the outer loop condition MUST be uniform across
    // all threads in the block. The previous `while (cursor_remove <
    // remove_size)` reads a __device__ global with no atomicity; another
    // block's atomicAdd can flip the value mid-read so thread 0 enters
    // the body while thread 31 doesn't. Once thread 0 hits __syncthreads()
    // inside, the block hangs forever and the cooperative grid sync below
    // hangs with it (manifests as a pa_dumper run pinning a CPU core at
    // 99% with no GPU forward progress). Fix: enter unconditionally and
    // let the post-sync `task_size <= 0` break be the uniform exit.
    while (true) {
      // FIX (B3): warpMin must be reset per-batch — previous code declared
      //           it once outside the while loop and never initialised.
      unsigned int warpMin = 0x7FFFFFFF;

      if (threadIdx.x == 0) {
        start_idx = atomicAdd(&cursor_remove, BLOCK_SIZE);
        if (start_idx >= remove_size) {
          task_size = 0;
        } else {
          int remaining = remove_size - start_idx;
          task_size = (remaining < BLOCK_SIZE) ? remaining : BLOCK_SIZE;
        }
      }
      __syncthreads();
      if (task_size <= 0) break;

      // Load entry (one per thread). Padding entries (idx >= remove_size)
      // get node_id=-1 with degree=0 so they contribute 0 to the scan.
      node_buf_t val;
      {
        int idx = start_idx + threadIdx.x;
        val.node_id    = (idx < remove_size && idx >= start_idx)
                       ? remove_list[idx] : -1;
        val.degree     = (val.node_id == -1) ? 0
                       : nidx[val.node_id + 1] - nidx[val.node_id];
        val.prefix_sum = 0;
      }
      __syncthreads();

      // Exclusive prefix-sum of degrees so each entry knows where its
      // [start, start+degree) work range lies in the flattened work array.
      int deg[1]  = { val.degree };
      int pref[1] = { 0 };
      BlockScan(temp_storage).ExclusiveSum(deg, pref);
      __syncthreads();

      // FIX (B2): index into shared via threadIdx.x (local), not tid (global).
      val.prefix_sum = pref[0];
      node_buf[threadIdx.x] = val;

      // FIX (B6/race): without this sync, thread 0 below can read
      // node_buf[task_size-1] before thread (task_size-1) finishes writing
      // it, yielding a stale `total_degree`. Manifests as run-to-run
      // priority drift even though phase 1 / theta / remove_list are all
      // deterministic across runs.
      __syncthreads();

      // FIX (B1): record valid count so total_degree probe is in-bounds.
      // task_size is the index of the last real entry + 1; padding sits at
      // [task_size .. BLOCK_SIZE-1] but contributes 0 to the prefix sum.
      if (threadIdx.x == 0) {
        buf_size     = task_size;
        total_degree = node_buf[task_size - 1].prefix_sum
                     + node_buf[task_size - 1].degree;
      }
      __syncthreads();

      // Distribute total_degree work units across threads
      for (int i = threadIdx.x; i < total_degree; i += BLOCK_SIZE) {
        int low = 0, high = buf_size - 1;
        while (low <= high) {
          int mid   = (low + high) >> 1;
          int start = node_buf[mid].prefix_sum;
          int end   = start + node_buf[mid].degree;
          if      (i >= end)   low  = mid + 1;
          else if (i <  start) high = mid - 1;
          else {
            int source   = node_buf[mid].node_id;
            int neighbor = __ldg(nlist + nidx[source] + (i - start));
            unsigned int it = __ldg(iteration_list + neighbor);
            if (!(it & 0x40000000u)) {
              warpMin = min(warpMin,
                            atomicSub(&degree_list[neighbor], 1) - 1);
            }
            break;     // hit found, no need to continue narrowing
          }
        }
      }
      warpMin = warpReduceMin(warpMin);
      if (lane == 0 && warpMin < 0x7FFFFFFF) atomicMin(&g_minDegree, warpMin);
      __syncthreads();
    }
    grid.sync();

    if (grid.thread_rank() == 0) {
      worker       += remove_size;
      remove_size   = 0;
      theta         = g_minDegree;
      atomicExch(&g_minDegree, 0x7FFFFFFF);
      iteration     = iteration + 1 + FuzzyNumber;
      cursor_remove = 0;
      iter_count++;
    }
    grid.sync();
  } while (worker != nodes);
}


// ═══════════════════════════════════════════════════════════════════════════
// P_SL_ELS_SDC_CTA_W — hybrid CTA / warp-level removal
//
// Identical to P_SL_ELS_SDC_CTA except phase 2 picks one of two paths per
// outer iteration based on remove_size:
//
//   * remove_size <  CTA_W_THRESHOLD : warp-balanced. Each warp grabs WS=32
//     vertices via lane-0 atomicAdd(&cursor_remove, WS), runs an inclusive
//     warp scan on degrees, and distributes the resulting flat work units
//     across its 32 lanes. Source-vertex lookup is a 32-iteration shfl
//     loop over the warp-local prefix sums (cheap because everything is
//     in registers, no shared mem).
//
//   * remove_size >= CTA_W_THRESHOLD : the existing CTA-balanced path
//     (cub::BlockScan over BLOCK_SIZE entries + log-time binary search
//     in shared memory).
//
// Motivation: when remove_size is small, the CTA path leaves most blocks
// idle (only one CTA grabs work; the rest atomicAdd, find task_size <= 0,
// and bail) and the BlockScan over BLOCK_SIZE lanes (with up to BLOCK_SIZE-
// task_size padding) is wasted. The warp path skips the BlockScan entirely
// and pays only WS-lane shuffles per batch.
//
// CTA_W_THRESHOLD is the smallest remove_size at which the CTA path wins;
// chosen empirically from the sweep — tunable.
// ═══════════════════════════════════════════════════════════════════════════
#ifndef CTA_W_THRESHOLD
#define CTA_W_THRESHOLD (BLOCK_SIZE * 4)   // = 2048 with BLOCK_SIZE=512
#endif

__global__ void P_SL_ELS_SDC_CTA_W(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  const int threads = gridDim.x * blockDim.x;
  const int lane    = threadIdx.x & 31;
  cg::grid_group grid = cg::this_grid();

  __shared__ node_buf_t node_buf[BLOCK_SIZE];
  __shared__ int buf_size;
  __shared__ int start_idx;
  __shared__ int task_size;
  __shared__ int total_degree;
  __shared__ int use_warp_path;          // dispatch flag, set per outer iter

  using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  do {
    unsigned int localMin = 0x7FFFFFFF;

    // ── Phase 1: scan + per-thread enqueue (OPT E1, identical to CTA) ─────
    for (int v = tid; v < nodes; v += threads) {
      int iteration_list_v = iteration_list[v];
      unsigned int prio        = (iteration_list_v >> 30) & 0x1;
      unsigned int iteration_v =  iteration_list_v        & 0x3FFFFFFF;
      unsigned int large_deg   = 0;
      unsigned int degree      = degree_list[v];
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

    // ── Phase 2 dispatch ───────────────────────────────────────────────────
    if (threadIdx.x == 0) {
      use_warp_path = (remove_size < CTA_W_THRESHOLD) ? 1 : 0;
    }
    __syncthreads();

    if (use_warp_path) {
      // ── Warp-balanced removal ────────────────────────────────────────────
      while (true) {
        int base = 0;
        if (lane == 0) base = atomicAdd(&cursor_remove, WS);
        base = __shfl_sync(0xffffffff, base, 0);
        if (base >= remove_size) break;

        int idx   = base + lane;
        int v_id  = (idx < remove_size) ? remove_list[idx] : -1;
        int v_deg = (v_id >= 0) ? (nidx[v_id + 1] - nidx[v_id]) : 0;

        // Inclusive warp scan on v_deg → exclusive prefix sum (per-lane regs).
        int incl = v_deg;
        #pragma unroll
        for (int s = 1; s < WS; s <<= 1) {
          int t = __shfl_up_sync(0xffffffff, incl, s);
          if (lane >= s) incl += t;
        }
        int excl       = incl - v_deg;
        int total_w    = __shfl_sync(0xffffffff, incl, WS - 1);
        int my_end     = excl + v_deg;

        unsigned int warpMin = 0x7FFFFFFF;
        // Distribute total_w work units across WS lanes.
        // FIX (W1/UB): all 32 lanes must enter every iteration of the outer
        // for, otherwise __shfl_sync(0xffffffff, ...) inside is called by a
        // partial warp (UB; manifests as cooperative grid-sync hang). Round
        // the loop bound up to a multiple of WS and gate the actual work +
        // atomicSub on `in_range`. The shfl_syncs run unconditionally so
        // every lane participates regardless of in_range.
        int max_i = ((total_w + WS - 1) / WS) * WS;
        for (int i = lane; i < max_i; i += WS) {
          bool in_range = (i < total_w);
          // Source lookup: smallest j s.t. (excl[j] + v_deg[j]) > i.
          int source_lid = 0;
          #pragma unroll
          for (int j = 0; j < WS; ++j) {
            int j_end = __shfl_sync(0xffffffff, my_end, j);
            if (in_range && j_end <= i) source_lid = j + 1;
          }
          // Unconditional shuffle — all 32 lanes participate.
          int source = __shfl_sync(0xffffffff, v_id, source_lid);
          int s_excl = __shfl_sync(0xffffffff, excl,  source_lid);
          if (in_range) {
            int neighbor = __ldg(nlist + nidx[source] + (i - s_excl));
            unsigned int it = __ldg(iteration_list + neighbor);
            if (!(it & 0x40000000u)) {
              warpMin = min(warpMin,
                            atomicSub(&degree_list[neighbor], 1) - 1);
            }
          }
        }
        warpMin = warpReduceMin(warpMin);
        if (lane == 0 && warpMin < 0x7FFFFFFF) atomicMin(&g_minDegree, warpMin);
      }
    } else {
      // ── CTA-balanced removal (identical to P_SL_ELS_SDC_CTA) ─────────────
      while (true) {
        unsigned int warpMin = 0x7FFFFFFF;

        if (threadIdx.x == 0) {
          start_idx = atomicAdd(&cursor_remove, BLOCK_SIZE);
          if (start_idx >= remove_size) {
            task_size = 0;
          } else {
            int remaining = remove_size - start_idx;
            task_size = (remaining < BLOCK_SIZE) ? remaining : BLOCK_SIZE;
          }
        }
        __syncthreads();
        if (task_size <= 0) break;

        node_buf_t val;
        {
          int idx = start_idx + threadIdx.x;
          val.node_id    = (idx < remove_size && idx >= start_idx)
                         ? remove_list[idx] : -1;
          val.degree     = (val.node_id == -1) ? 0
                         : nidx[val.node_id + 1] - nidx[val.node_id];
          val.prefix_sum = 0;
        }
        __syncthreads();

        int deg[1]  = { val.degree };
        int pref[1] = { 0 };
        BlockScan(temp_storage).ExclusiveSum(deg, pref);
        __syncthreads();

        val.prefix_sum = pref[0];
        node_buf[threadIdx.x] = val;
        __syncthreads();

        if (threadIdx.x == 0) {
          buf_size     = task_size;
          total_degree = node_buf[task_size - 1].prefix_sum
                       + node_buf[task_size - 1].degree;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_degree; i += BLOCK_SIZE) {
          int low = 0, high = buf_size - 1;
          while (low <= high) {
            int mid   = (low + high) >> 1;
            int start = node_buf[mid].prefix_sum;
            int end   = start + node_buf[mid].degree;
            if      (i >= end)   low  = mid + 1;
            else if (i <  start) high = mid - 1;
            else {
              int source   = node_buf[mid].node_id;
              int neighbor = __ldg(nlist + nidx[source] + (i - start));
              unsigned int it = __ldg(iteration_list + neighbor);
              if (!(it & 0x40000000u)) {
                warpMin = min(warpMin,
                              atomicSub(&degree_list[neighbor], 1) - 1);
              }
              break;
            }
          }
        }
        warpMin = warpReduceMin(warpMin);
        if (lane == 0 && warpMin < 0x7FFFFFFF) atomicMin(&g_minDegree, warpMin);
        __syncthreads();
      }
    }

    grid.sync();

    if (grid.thread_rank() == 0) {
      worker       += remove_size;
      remove_size   = 0;
      theta         = g_minDegree;
      atomicExch(&g_minDegree, 0x7FFFFFFF);
      iteration     = iteration + 1 + FuzzyNumber;
      cursor_remove = 0;
      iter_count++;
    }
    grid.sync();
  } while (worker != nodes);
}


// ═══════════════════════════════════════════════════════════════════════════
// P_SL_ELS_SDC_CTA_S — dynamic dispatch CTA / SDC-warp-per-vertex
//
// Phase 2 chooses one of two paths per outer iteration:
//
//   * remove_size <  CTA_S_THRESHOLD : SDC's classic warp-per-vertex pattern
//     (no work-balance, no per-warp atomicAdd on cursor_remove). Each warp
//     consumes remove_list entries via grid-stride k = warpId, k+numWarp,
//     ... and processes that vertex's neighbours across its 32 lanes.
//     This is exactly the inner loop from P_SL_ELS_SDC and was proven (in
//     the SDC vs CTA sweep) to win on small-/medium-sized graphs because
//     it avoids the cursor_remove atomic contention and the per-work-unit
//     source lookup entirely.
//
//   * remove_size >= CTA_S_THRESHOLD : the existing CTA-balanced path
//     (cub::BlockScan + binary search), which beats SDC on big sparse
//     low-degree-variance graphs.
//
// Threshold default: BLOCK_SIZE * 4 (= 2048). Tuned later if needed.
// ═══════════════════════════════════════════════════════════════════════════
#ifndef CTA_S_THRESHOLD
#define CTA_S_THRESHOLD (BLOCK_SIZE * 4)
#endif

__global__ void P_SL_ELS_SDC_CTA_S(
    const int  nodes,
    const int* __restrict__ nidx,
    const int* __restrict__ nlist,
    unsigned int* __restrict__ degree_list,
    unsigned int* __restrict__ iteration_list)
{
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  const int threads = gridDim.x * blockDim.x;
  const int lane    = threadIdx.x & 31;
  const int warpId  = tid >> 5;
  const int numWarp = threads >> 5;
  cg::grid_group grid = cg::this_grid();

  __shared__ node_buf_t node_buf[BLOCK_SIZE];
  __shared__ int buf_size;
  __shared__ int start_idx;
  __shared__ int task_size;
  __shared__ int total_degree;
  __shared__ int use_sdc_path;

  using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  do {
    unsigned int localMin = 0x7FFFFFFF;

    // ── Phase 1: scan + per-thread enqueue (OPT E1, identical to CTA) ─────
    for (int v = tid; v < nodes; v += threads) {
      int iteration_list_v = iteration_list[v];
      unsigned int prio        = (iteration_list_v >> 30) & 0x1;
      unsigned int iteration_v =  iteration_list_v        & 0x3FFFFFFF;
      unsigned int large_deg   = 0;
      unsigned int degree      = degree_list[v];
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

    // ── Phase 2 dispatch (uniform across grid: every block reads the same
    //    remove_size after the grid.sync above) ─────────────────────────────
    if (threadIdx.x == 0) {
      use_sdc_path = (remove_size < CTA_S_THRESHOLD) ? 1 : 0;
    }
    __syncthreads();

    if (use_sdc_path) {
      // ── SDC warp-per-vertex (lifted from P_SL_ELS_SDC's inner loop) ──────
      // Each warp consumes one vertex per iteration via grid-stride. No
      // cursor_remove atomicAdd, no per-work-unit source lookup. Wins
      // consistently on remove_size < a few thousand.
      for (int k = warpId; k < remove_size; k += numWarp) {
        int v = remove_list[k];
        int beg, end;
        if (lane == 0) {
          beg = nidx[v];
          end = nidx[v + 1];
        }
        beg = __shfl_sync(0xffffffff, beg, 0);
        end = __shfl_sync(0xffffffff, end, 0);

        unsigned int warpMin = 0x7FFFFFFF;
        for (int i = beg + lane; i < end; i += 32) {
          int nei = __ldg(nlist + i);
          unsigned int it = __ldg(iteration_list + nei);
          if (!(it & 0x40000000u)) {
            warpMin = min(warpMin,
                          atomicSub(&degree_list[nei], 1) - 1);
          }
        }
        warpMin = warpReduceMin(warpMin);
        if (lane == 0 && warpMin < 0x7FFFFFFF) atomicMin(&g_minDegree, warpMin);
      }
    } else {
      // ── CTA-balanced removal (identical to P_SL_ELS_SDC_CTA) ─────────────
      while (true) {
        unsigned int warpMin = 0x7FFFFFFF;

        if (threadIdx.x == 0) {
          start_idx = atomicAdd(&cursor_remove, BLOCK_SIZE);
          if (start_idx >= remove_size) {
            task_size = 0;
          } else {
            int remaining = remove_size - start_idx;
            task_size = (remaining < BLOCK_SIZE) ? remaining : BLOCK_SIZE;
          }
        }
        __syncthreads();
        if (task_size <= 0) break;

        node_buf_t val;
        {
          int idx = start_idx + threadIdx.x;
          val.node_id    = (idx < remove_size && idx >= start_idx)
                         ? remove_list[idx] : -1;
          val.degree     = (val.node_id == -1) ? 0
                         : nidx[val.node_id + 1] - nidx[val.node_id];
          val.prefix_sum = 0;
        }
        __syncthreads();

        int deg[1]  = { val.degree };
        int pref[1] = { 0 };
        BlockScan(temp_storage).ExclusiveSum(deg, pref);
        __syncthreads();

        val.prefix_sum = pref[0];
        node_buf[threadIdx.x] = val;
        __syncthreads();

        if (threadIdx.x == 0) {
          buf_size     = task_size;
          total_degree = node_buf[task_size - 1].prefix_sum
                       + node_buf[task_size - 1].degree;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < total_degree; i += BLOCK_SIZE) {
          int low = 0, high = buf_size - 1;
          while (low <= high) {
            int mid   = (low + high) >> 1;
            int start = node_buf[mid].prefix_sum;
            int end   = start + node_buf[mid].degree;
            if      (i >= end)   low  = mid + 1;
            else if (i <  start) high = mid - 1;
            else {
              int source   = node_buf[mid].node_id;
              int neighbor = __ldg(nlist + nidx[source] + (i - start));
              unsigned int it = __ldg(iteration_list + neighbor);
              if (!(it & 0x40000000u)) {
                warpMin = min(warpMin,
                              atomicSub(&degree_list[neighbor], 1) - 1);
              }
              break;
            }
          }
        }
        warpMin = warpReduceMin(warpMin);
        if (lane == 0 && warpMin < 0x7FFFFFFF) atomicMin(&g_minDegree, warpMin);
        __syncthreads();
      }
    }

    grid.sync();

    if (grid.thread_rank() == 0) {
      worker       += remove_size;
      remove_size   = 0;
      theta         = g_minDegree;
      atomicExch(&g_minDegree, 0x7FFFFFFF);
      iteration     = iteration + 1 + FuzzyNumber;
      cursor_remove = 0;
      iter_count++;

#ifdef DYNAMIC_THETA
      // ─── Online θ controller ─────────────────────────────────────
      // Every CTRL_K iterations, check removal rate. If we removed at least
      // CTRL_RATE_THRESHOLD·V vertices since the last checkpoint, bump
      // FuzzyNumber by CTRL_STEP (capped at CTRL_CAP). Monotone non-decreasing.
      if (CTRL_K > 0 && (iter_count % CTRL_K) == 0 && iter_count >= CTRL_K) {
          int delta = worker - last_remove_size;
          last_remove_size = worker;
          if (delta >= (int)(CTRL_RATE_THRESHOLD * (float)g_nodes)) {
              int new_fz = FuzzyNumber + CTRL_STEP;
              if (CTRL_CAP > 0 && new_fz > CTRL_CAP) new_fz = CTRL_CAP;
              if (new_fz > FuzzyNumber) {
                  FuzzyNumber = new_fz;
                  if (bump_count < BUMP_LOG_MAX) {
                      bump_iter [bump_count] = iter_count;
                      bump_theta[bump_count] = new_fz;
                      bump_count = bump_count + 1;
                  }
              }
          }
      }
#endif
    }
    grid.sync();
  } while (worker != nodes);
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
    iter_count++;
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
  iter_count++;
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
  iter_count++;
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
  iter_count++;
}
grid.sync();
} while (worker != nodes);

// ── Post-loop: Initialise posscol2 for runLarge ──────────
for (int i = tid; i < edges / WS + 1; i += threads) posscol2[i] = -1;
}
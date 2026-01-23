#include "globals.cuh"
#include <cuda.h>
#include <cooperative_groups.h>
#include <stdio.h>
namespace cg = cooperative_groups;

static __device__ unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

__global__
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
      unsigned int v_priority=priority[v];
      // if(((v_priority >> 30) & 0x1)!=1)printf("vertex[%d]:%d-%d\n",v,int(v_priority& 0x3FFFFFFF),v_priority);

      cond = (degv >= WS);
      if (cond) {
        wl[atomicAdd(&wlsize, 1)] = v;
      } else {
        active = 0;
        pos = beg;
        for (int i = beg; i < end; i++) {
          const int nei = nlist[i];
          const int degn = nidx[nei + 1] - nidx[nei];
          const unsigned int priority_n=priority[nei];
          // unsigned int hash_v = random_vals_d[v];
          // unsigned int hash_nei = random_vals_d[nei];
          unsigned int hash_v = hash(v);
          unsigned int hash_nei = hash(nei);

          // if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
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
      unsigned int wvpriority=priority[wv];
      int wpos = wbeg;
      for (int i = wbeg + lane; __any_sync(Warp, i < wend); i += WS) {
        int wnei;
        bool prio = false;
        if (i < wend) {
          wnei = nlist[i];
          const int wdegn = nidx[wnei + 1] - nidx[wnei];
          // prio = ((wdegv < wdegn) || ((wdegv == wdegn) && (hash(wv) < hash(wnei))) || ((wdegv == wdegn) && (hash(wv) == hash(wnei)) && (wv < wnei)));
          unsigned int wnpriority = priority[wnei];
          // unsigned int hash_wv = random_vals_d[wv];
          // unsigned int hash_wnei = random_vals_d[wnei];
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
  // if (maxrange >= Mask) {printf("too many active neighbors\n"); asm("trap;");}

  for (int i = thread; i < edges / WS + 1; i += threads) posscol2[i] = -1;
}


__global__ 
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
              wdone &= neidone; //consolidated below
              if (neidone) {
                const int neicol = neidata;
                if (neicol < WS) {
                  wpcol &= ~((unsigned int)MSB >> neicol); //consolidated below
                } else {
                  if ((wmincol <= neicol) && (neicol < wmaxcol) && ((posscol2[woffs + neicol / WS] << (neicol % WS)) < 0)) {
                    atomicAnd((int*)&posscol2[woffs + neicol / WS], ~((unsigned int)MSB >> (neicol % WS)));
                  }
                }
              } else {
                const int neimincol = neidata & Mask;
                const int neimaxcol = neimincol + neirange;
                if ((neimincol <= wmincol) && (neimaxcol >= wmincol)) wshortcut = false; //consolidated below
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


__global__ 
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

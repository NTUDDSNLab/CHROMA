#include "chroma_utils.cuh"
#include "globals.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr unsigned long long kNoReduction =
    std::numeric_limits<unsigned long long>::max();

struct GPUTimer {
    cudaEvent_t beg, end;
    GPUTimer() { cudaEventCreate(&beg); cudaEventCreate(&end); }
    ~GPUTimer() { cudaEventDestroy(beg); cudaEventDestroy(end); }
    void start() { cudaEventRecord(beg, 0); }
    float stop() {
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, beg, end);
        return 0.001f * ms;
    }
};

void check_cuda_or_exit(cudaError_t err, const char* message)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int count_colors_from_device(const ECLgraph& g,
                             const int* color_d,
                             int* color_host_buffer)
{
    check_cuda_or_exit(
        cudaMemcpy(color_host_buffer, color_d, g.nodes * sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy colors from device during post-processing");

    int max_color = -1;
    for (int v = 0; v < g.nodes; ++v) {
        max_color = std::max(max_color, color_host_buffer[v]);
    }
    return max_color + 1;
}

// ════════════════════════════════════════════════════════════════════════════
//  Heuristic 1 — find colour pair (i,j) that never appears in the worklist's
//  edges, swap j-coloured worklist vertices to colour i, swap hic vertices
//  to j. One swap reduces hic by 1.
// ════════════════════════════════════════════════════════════════════════════

__global__ void reduce1CollectWorklist(const int nodes,
                                       const int* const __restrict__ nidx,
                                       const int* const __restrict__ nlist,
                                       const int* const __restrict__ color,
                                       const int hic,
                                       int* const __restrict__ wl,
                                       int* const __restrict__ wlsize)
{
    const int global_thread = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane = threadIdx.x % WS;
    const int warp = global_thread / WS;
    const int warps = (gridDim.x * blockDim.x) / WS;

    for (int v = warp; v < nodes; v += warps) {
        const int beg = nidx[v];
        const int end = nidx[v + 1];
        bool adjacent_to_hic = false;

        for (int i = beg + lane; i < end; i += WS) {
            if (color[nlist[i]] == hic) {
                adjacent_to_hic = true;
                break;
            }
        }

        if (__any_sync(Warp, adjacent_to_hic) && (lane == 0)) {
            wl[atomicAdd(wlsize, 1)] = v;
        }
    }
}

__global__ void reduce1MarkPairs(const int* const __restrict__ nidx,
                                 const int* const __restrict__ nlist,
                                 const int* const __restrict__ color,
                                 const int* const __restrict__ wl,
                                 const int* const __restrict__ wlsize,
                                 const int hic,
                                 unsigned int* const __restrict__ pair_bits)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int idx = from; idx < *wlsize; idx += incr) {
        const int v = wl[idx];
        const int vcol = color[v];
        if ((vcol < 0) || (vcol >= hic)) continue;

        const size_t row_base = static_cast<size_t>(vcol) * static_cast<size_t>(hic);
        for (int i = nidx[v]; i < nidx[v + 1]; ++i) {
            const int ncol = color[nlist[i]];
            if ((ncol < 0) || (ncol >= hic)) continue;

            const size_t bit_index = row_base + static_cast<size_t>(ncol);
            atomicOr(&pair_bits[bit_index >> 5], 1u << (bit_index & 31));
        }
    }
}

__global__ void reduce1FindGap(const int hic,
                               const unsigned int* const __restrict__ pair_bits,
                               unsigned long long* const __restrict__ min_reduce)
{
    const size_t total_pairs = static_cast<size_t>(hic) * static_cast<size_t>(hic);
    const size_t from = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t incr = gridDim.x * blockDim.x;

    for (size_t ind = from; ind < total_pairs; ind += incr) {
        const int i = static_cast<int>(ind % static_cast<size_t>(hic));
        const int j = static_cast<int>(ind / static_cast<size_t>(hic));
        if (i == j) continue;

        const unsigned int word = pair_bits[ind >> 5];
        if (((word >> (ind & 31)) & 1u) != 0u) continue;

        const unsigned long long candidate =
            (static_cast<unsigned long long>(max(i, j)) << 32) |
            static_cast<unsigned int>(min(i, j));
        atomicMin(min_reduce, candidate);
    }
}

__global__ void reduce1RecolorWorklist(int* const __restrict__ color,
                                       const int* const __restrict__ wl,
                                       const int* const __restrict__ wlsize,
                                       const int from_color,
                                       const int to_color)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int i = from; i < *wlsize; i += incr) {
        const int v = wl[i];
        if (color[v] == from_color) {
            color[v] = to_color;
        }
    }
}

__global__ void reduce1RecolorHighest(const int nodes,
                                      int* const __restrict__ color,
                                      const int hic,
                                      const int new_color)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        if (color[v] == hic) {
            color[v] = new_color;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Heuristic 2 — Kempe-chain-style swap.
//  Union all hic-coloured vertices that share a neighbour into hic-sets;
//  for each set, mark which colours each c-coloured neighbour's neighbours
//  use; then for each hic vertex find a colour j (j < min(32,hic)) whose
//  posscolor still has a free bit and swap that c with the free colour.
//  Lifted from ECL-GC-ColorReduction_12.cu (Alabandi & Burtscher 2022),
//  with the __device__ wlsize/minReduce globals replaced by passed pointers.
// ════════════════════════════════════════════════════════════════════════════

__host__ __device__ int representative(const int idx, int* const comp)
{
    int curr = comp[idx];
    if (curr != idx) {
        int next, prev = idx;
        while (curr > (next = comp[curr])) {
            comp[prev] = next;
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

__global__ void reduce2InitState(const int nodes,
                                 const int* const __restrict__ color,
                                 int* const __restrict__ hic,
                                 int* const __restrict__ nstate)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) nstate[v] = -1;
    for (int v = from; v < nodes; v += incr) atomicMax(hic, color[v]);
}

__global__ void reduce2CollectHic(const int nodes,
                                  const int* const __restrict__ color,
                                  int* const __restrict__ wl,
                                  int* const __restrict__ wlsize,
                                  const int* const __restrict__ hic,
                                  int* const __restrict__ nstate)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        if (color[v] == *hic) {
            nstate[v] = v;
            wl[atomicAdd(wlsize, 1)] = v;
        }
    }
}

__global__ void reduce2ComputeUnion(const int nodes,
                                    const int* const __restrict__ nidx,
                                    const int* const __restrict__ nlist,
                                    int* const __restrict__ nstate,
                                    const int* const __restrict__ color,
                                    const int* const __restrict__ hic)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        int prevst = -1;
        bool first = true;
        for (int i = nidx[v]; i < nidx[v + 1]; ++i) {
            const int nei = nlist[i];
            if (color[nei] != *hic) continue;

            int neist = representative(nei, nstate);
            if (first) {
                first = false;
                prevst = neist;
            } else {
                bool repeat;
                do {
                    repeat = false;
                    if (neist != prevst) {
                        int ret;
                        if (prevst < neist) {
                            if ((ret = atomicCAS(&nstate[neist], neist, prevst)) != neist) {
                                neist = ret;
                                repeat = true;
                            }
                        } else {
                            if ((ret = atomicCAS(&nstate[prevst], prevst, neist)) != prevst) {
                                prevst = ret;
                                repeat = true;
                            }
                        }
                    }
                } while (repeat);
            }
        }
    }
}

__global__ void reduce2InitHicSet(const int nodes,
                                  const int* const __restrict__ nidx,
                                  const int* const __restrict__ nlist,
                                  int* const __restrict__ nstate,
                                  const int* const __restrict__ color,
                                  const int* const __restrict__ hic,
                                  int* const __restrict__ hicSet,
                                  int* const __restrict__ posscolor)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        hicSet[v] = -1;
        for (int j = nidx[v]; j < nidx[v + 1]; ++j) {
            const int nlj = nlist[j];
            if (color[nlj] == *hic) {
                hicSet[v] = representative(nlj, nstate);
                break;
            }
        }
    }

    const int mask = (*hic >= 32) ? 0 : (-1 << *hic);
    for (int i = from; i < nodes * 32; i += incr) {
        posscolor[i] = (1 << (i % 32)) | mask;
    }
}

__global__ void reduce2MarkColors(const int* const __restrict__ nidx,
                                  const int* const __restrict__ nlist,
                                  int* const __restrict__ nstate,
                                  const int* const __restrict__ color,
                                  int* const __restrict__ posscolor,
                                  const int* const __restrict__ wl,
                                  const int* const __restrict__ wlsize)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int i = from; i < *wlsize; i += incr) {
        const int v = wl[i];
        const int state = representative(v, nstate);
        for (int j = nidx[v]; j < nidx[v + 1]; ++j) {
            const int nlj = nlist[j];
            const int ncol = color[nlj];
            if (ncol < 32) {
                int poc = posscolor[state * 32 + ncol];
                for (int k = nidx[nlj]; k < nidx[nlj + 1]; ++k) {
                    const int nlk = nlist[k];
                    const int ncolk = color[nlk];
                    if (ncolk < 32) {
                        poc |= 1 << ncolk;
                    }
                }
                atomicOr(&posscolor[state * 32 + ncol], poc);
            }
        }
    }
}

__global__ void reduce2CrossSetConflict(const int nodes,
                                        const int* const __restrict__ nidx,
                                        const int* const __restrict__ nlist,
                                        const int* const __restrict__ color,
                                        int* const __restrict__ posscolor,
                                        const int* const __restrict__ hicSet)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        if (hicSet[v] == -1) continue;
        const int vcol = color[v];
        for (int j = nidx[v]; j < nidx[v + 1]; ++j) {
            const int nlj = nlist[j];
            if (hicSet[nlj] == -1) continue;
            if (hicSet[v] == hicSet[nlj]) continue;

            const int ncol = color[nlj];
            int poc1 = posscolor[hicSet[nlj] * 32 + ncol];
            int poc2 = posscolor[hicSet[v]   * 32 + vcol];
            // Push the limitation onto whichever side has more zeros free
            // (so it loses the fewest options).
            if (__popc(poc1) <= __popc(poc2)) {
                atomicOr(&posscolor[hicSet[nlj] * 32 + ncol], ~poc1 & ~poc2);
            } else {
                atomicOr(&posscolor[hicSet[v]   * 32 + vcol], ~poc1 & ~poc2);
            }
        }
    }
}

__global__ void reduce2RecolorH2(const int nodes,
                                 const int* const __restrict__ nidx,
                                 const int* const __restrict__ nlist,
                                 int* const __restrict__ color,
                                 int* const __restrict__ posscolor,
                                 const int* const __restrict__ wl,
                                 const int* const __restrict__ wlsize,
                                 int* const __restrict__ nstate,
                                 const int* const __restrict__ hic)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int i = from; i < *wlsize; i += incr) {
        const int vr      = wl[i];
        const int vrstate = representative(vr, nstate);
        const int beg     = nidx[vr];
        const int end     = nidx[vr + 1];
        const int minidx  = (32 < *hic) ? 32 : *hic;

        for (int j = 0; j < minidx; ++j) {
            int poc = posscolor[vrstate * 32 + j];
            int pos = __ffs(~poc);     // first 0-bit (1-indexed)
            if (pos == 0) continue;

            // Found a free colour. Swap j-coloured neighbours of vr to
            // (pos-1), then assign vr → j.
            for (int v = beg; v < end; ++v) {
                const int nlv  = nlist[v];
                const int ncol = color[nlv];
                if (ncol == j) {
                    color[nlv] = pos - 1;
                }
            }
            color[vr] = j;
            break;
        }
    }
}

}  // namespace

// ───────────────────────────────────────────────────────────────────────────
//  Heuristic 1 driver
// ───────────────────────────────────────────────────────────────────────────
static ColorReductionStats run_reduce_h1(int blocks,
                                         const ECLgraph& g,
                                         DevPtr& d,
                                         int* color_host_buffer,
                                         int colors_before)
{
    ColorReductionStats stats{};
    stats.colors_before = colors_before;
    stats.colors_after  = colors_before;

    const int hic = colors_before - 1;
    if ((g.nodes == 0) || (hic <= 0)) return stats;

    size_t free_mem = 0, total_mem = 0;
    check_cuda_or_exit(cudaMemGetInfo(&free_mem, &total_mem),
                       "Failed to query GPU memory before H1");

    const size_t total_pairs = static_cast<size_t>(hic) * static_cast<size_t>(hic);
    const size_t pair_words  = (total_pairs + 31) / 32;
    const size_t pair_bytes  = pair_words * sizeof(unsigned int);
    const size_t temp_bytes  = pair_bytes + sizeof(int) + sizeof(unsigned long long);

    if (temp_bytes > (free_mem / 2)) {
        printf("post reduction status: skipped (insufficient free GPU memory for heuristic 1)\n");
        return stats;
    }

    int* wlsize_d = nullptr;
    unsigned int* pair_bits_d = nullptr;
    unsigned long long* min_reduce_d = nullptr;

    if (cudaMalloc(&wlsize_d, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&pair_bits_d, pair_bytes) != cudaSuccess ||
        cudaMalloc(&min_reduce_d, sizeof(unsigned long long)) != cudaSuccess) {
        cudaFree(wlsize_d);
        cudaFree(pair_bits_d);
        cudaFree(min_reduce_d);
        printf("post reduction status: skipped (temporary allocation failed for heuristic 1)\n");
        return stats;
    }

    const int zero = 0;
    check_cuda_or_exit(cudaMemcpy(wlsize_d, &zero, sizeof(int), cudaMemcpyHostToDevice),
                       "Failed to initialize H1 wlsize");
    check_cuda_or_exit(cudaMemset(pair_bits_d, 0, pair_bytes),
                       "Failed to clear H1 pair matrix");
    check_cuda_or_exit(cudaMemcpy(min_reduce_d, &kNoReduction, sizeof(unsigned long long),
                                  cudaMemcpyHostToDevice),
                       "Failed to initialize H1 min_reduce");

    GPUTimer timer;
    timer.start();
    stats.attempted = true;

    reduce1CollectWorklist<<<blocks, ThreadsPerBlock>>>(
        g.nodes, d.nidx_d, d.nlist_d, d.color_d, hic, d.wl_d, wlsize_d);
    check_cuda_or_exit(cudaDeviceSynchronize(), "Failed after reduce1CollectWorklist");

    reduce1MarkPairs<<<blocks, ThreadsPerBlock>>>(
        d.nidx_d, d.nlist_d, d.color_d, d.wl_d, wlsize_d, hic, pair_bits_d);
    check_cuda_or_exit(cudaDeviceSynchronize(), "Failed after reduce1MarkPairs");

    reduce1FindGap<<<blocks, ThreadsPerBlock>>>(hic, pair_bits_d, min_reduce_d);
    check_cuda_or_exit(cudaDeviceSynchronize(), "Failed after reduce1FindGap");

    unsigned long long min_reduce = kNoReduction;
    check_cuda_or_exit(cudaMemcpy(&min_reduce, min_reduce_d, sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost),
                       "Failed to copy H1 candidate");

    if (min_reduce != kNoReduction) {
        const int from_color = static_cast<int>(min_reduce & 0xffffffffu);
        const int to_color   = static_cast<int>(min_reduce >> 32);

        reduce1RecolorWorklist<<<blocks, ThreadsPerBlock>>>(
            d.color_d, d.wl_d, wlsize_d, from_color, to_color);
        check_cuda_or_exit(cudaDeviceSynchronize(), "Failed after reduce1RecolorWorklist");

        reduce1RecolorHighest<<<blocks, ThreadsPerBlock>>>(
            g.nodes, d.color_d, hic, from_color);
        check_cuda_or_exit(cudaDeviceSynchronize(), "Failed after reduce1RecolorHighest");
    }

    stats.runtime_sec = timer.stop();

    check_cuda_or_exit(cudaFree(wlsize_d),     "Failed to free H1 wlsize_d");
    check_cuda_or_exit(cudaFree(pair_bits_d),  "Failed to free H1 pair_bits_d");
    check_cuda_or_exit(cudaFree(min_reduce_d), "Failed to free H1 min_reduce_d");

    if (min_reduce == kNoReduction) {
        printf("post reduction status: H1 found no reducible pair\n");
        return stats;
    }

    stats.colors_after = count_colors_from_device(g, d.color_d, color_host_buffer);
    stats.applied = stats.colors_after < stats.colors_before;
    if (!stats.applied) {
        printf("post reduction status: H1 ran but color count did not improve\n");
    }
    return stats;
}

// ───────────────────────────────────────────────────────────────────────────
//  Heuristic 2 driver
// ───────────────────────────────────────────────────────────────────────────
static ColorReductionStats run_reduce_h2(int blocks,
                                         const ECLgraph& g,
                                         DevPtr& d,
                                         int* color_host_buffer,
                                         int colors_before)
{
    ColorReductionStats stats{};
    stats.colors_before = colors_before;
    stats.colors_after  = colors_before;

    if (g.nodes == 0 || colors_before <= 1) return stats;

    // Memory check: posscolor is the dominant temporary (32 ints / vertex).
    size_t free_mem = 0, total_mem = 0;
    check_cuda_or_exit(cudaMemGetInfo(&free_mem, &total_mem),
                       "Failed to query GPU memory before H2");
    const size_t bytes_posscolor = (size_t)g.nodes * 32u * sizeof(int);
    const size_t bytes_nstate    = (size_t)g.nodes * sizeof(int);
    const size_t bytes_hicSet    = (size_t)g.nodes * sizeof(int);
    const size_t temp_bytes      = bytes_posscolor + bytes_nstate + bytes_hicSet
                                 + 2 * sizeof(int);
    if (temp_bytes > (free_mem / 2)) {
        printf("post reduction status: skipped (insufficient free GPU memory for heuristic 2; "
               "needs %.1f MB, %.1f MB free)\n",
               temp_bytes / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
        return stats;
    }

    int *hic_d = nullptr, *wlsize_d = nullptr;
    int *nstate_d = nullptr, *hicSet_d = nullptr, *posscolor_d = nullptr;
    if (cudaMalloc(&hic_d,       sizeof(int))       != cudaSuccess ||
        cudaMalloc(&wlsize_d,    sizeof(int))       != cudaSuccess ||
        cudaMalloc(&nstate_d,    bytes_nstate)      != cudaSuccess ||
        cudaMalloc(&hicSet_d,    bytes_hicSet)      != cudaSuccess ||
        cudaMalloc(&posscolor_d, bytes_posscolor)   != cudaSuccess) {
        cudaFree(hic_d); cudaFree(wlsize_d);
        cudaFree(nstate_d); cudaFree(hicSet_d); cudaFree(posscolor_d);
        printf("post reduction status: skipped (temporary allocation failed for heuristic 2)\n");
        return stats;
    }

    int hic_init = INT_MIN;
    int zero = 0;
    check_cuda_or_exit(cudaMemcpy(hic_d,    &hic_init, sizeof(int), cudaMemcpyHostToDevice),
                       "Failed to initialize H2 hic");
    check_cuda_or_exit(cudaMemcpy(wlsize_d, &zero,     sizeof(int), cudaMemcpyHostToDevice),
                       "Failed to initialize H2 wlsize");

    GPUTimer timer;
    timer.start();
    stats.attempted = true;

    reduce2InitState<<<blocks, ThreadsPerBlock>>>(
        g.nodes, d.color_d, hic_d, nstate_d);

    reduce2CollectHic<<<blocks, ThreadsPerBlock>>>(
        g.nodes, d.color_d, d.wl_d, wlsize_d, hic_d, nstate_d);

    reduce2ComputeUnion<<<blocks, ThreadsPerBlock>>>(
        g.nodes, d.nidx_d, d.nlist_d, nstate_d, d.color_d, hic_d);

    reduce2InitHicSet<<<blocks, ThreadsPerBlock>>>(
        g.nodes, d.nidx_d, d.nlist_d, nstate_d, d.color_d, hic_d, hicSet_d, posscolor_d);

    reduce2MarkColors<<<blocks, ThreadsPerBlock>>>(
        d.nidx_d, d.nlist_d, nstate_d, d.color_d, posscolor_d, d.wl_d, wlsize_d);

    reduce2CrossSetConflict<<<blocks, ThreadsPerBlock>>>(
        g.nodes, d.nidx_d, d.nlist_d, d.color_d, posscolor_d, hicSet_d);

    reduce2RecolorH2<<<blocks, ThreadsPerBlock>>>(
        g.nodes, d.nidx_d, d.nlist_d, d.color_d, posscolor_d, d.wl_d, wlsize_d, nstate_d, hic_d);

    check_cuda_or_exit(cudaDeviceSynchronize(), "Failed after H2 pipeline");
    stats.runtime_sec = timer.stop();

    cudaFree(hic_d); cudaFree(wlsize_d);
    cudaFree(nstate_d); cudaFree(hicSet_d); cudaFree(posscolor_d);

    stats.colors_after = count_colors_from_device(g, d.color_d, color_host_buffer);
    stats.applied = stats.colors_after < stats.colors_before;
    if (!stats.applied) {
        printf("post reduction status: H2 ran but color count did not improve\n");
    }
    return stats;
}

// ───────────────────────────────────────────────────────────────────────────
//  Public dispatcher (used by CHROMA.cu after every CA run)
// ───────────────────────────────────────────────────────────────────────────
ColorReductionStats run_post_color_reduction(int blocks,
                                             const ECLgraph& g,
                                             DevPtr& d,
                                             int* color_host_buffer,
                                             bool enabled)
{
    ColorReductionStats stats{};
    stats.colors_before = count_colors_from_device(g, d.color_d, color_host_buffer);
    stats.colors_after  = stats.colors_before;

    if (!enabled) {
        // Reduction disabled (--no-reduce); leave colors as PA+CA produced them.
        return stats;
    }

    // ECL-GC's dispatch: dense graphs benefit from H1 (cheaper, h*h work);
    // sparse graphs benefit from H2 (Kempe swap that may move more work).
    const float avg_deg = (g.nodes > 0)
                          ? static_cast<float>(g.edges) / static_cast<float>(g.nodes)
                          : 0.0f;
    if (avg_deg > 10.0f) {
        return run_reduce_h1(blocks, g, d, color_host_buffer, stats.colors_before);
    } else {
        return run_reduce_h2(blocks, g, d, color_host_buffer, stats.colors_before);
    }
}

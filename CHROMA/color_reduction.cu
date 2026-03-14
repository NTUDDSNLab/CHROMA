#include "chroma_utils.cuh"
#include "globals.cuh"

#include <algorithm>
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

}  // namespace

ColorReductionStats run_post_color_reduction(int blocks,
                                             const ECLgraph& g,
                                             DevPtr& d,
                                             int* color_host_buffer)
{
    ColorReductionStats stats{};
    stats.colors_before = count_colors_from_device(g, d.color_d, color_host_buffer);
    stats.colors_after = stats.colors_before;

    const int hic = stats.colors_before - 1;
    if ((g.nodes == 0) || (hic <= 0)) {
        return stats;
    }

    size_t free_mem = 0;
    size_t total_mem = 0;
    check_cuda_or_exit(cudaMemGetInfo(&free_mem, &total_mem),
                       "Failed to query GPU memory before post-processing");

    const size_t total_pairs = static_cast<size_t>(hic) * static_cast<size_t>(hic);
    const size_t pair_words = (total_pairs + 31) / 32;
    const size_t pair_bytes = pair_words * sizeof(unsigned int);
    const size_t temp_bytes = pair_bytes + sizeof(int) + sizeof(unsigned long long);

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
                       "Failed to initialize post-reduction worklist size");
    check_cuda_or_exit(cudaMemset(pair_bits_d, 0, pair_bytes),
                       "Failed to clear post-reduction pair matrix");
    check_cuda_or_exit(cudaMemcpy(min_reduce_d, &kNoReduction, sizeof(unsigned long long),
                                  cudaMemcpyHostToDevice),
                       "Failed to initialize post-reduction reduction target");

    GPUTimer timer;
    timer.start();
    stats.attempted = true;

    reduce1CollectWorklist<<<blocks, ThreadsPerBlock>>>(
        g.nodes, d.nidx_d, d.nlist_d, d.color_d, hic, d.wl_d, wlsize_d);
    check_cuda_or_exit(cudaDeviceSynchronize(),
                       "Failed after reduce1CollectWorklist");

    reduce1MarkPairs<<<blocks, ThreadsPerBlock>>>(
        d.nidx_d, d.nlist_d, d.color_d, d.wl_d, wlsize_d, hic, pair_bits_d);
    check_cuda_or_exit(cudaDeviceSynchronize(),
                       "Failed after reduce1MarkPairs");

    reduce1FindGap<<<blocks, ThreadsPerBlock>>>(
        hic, pair_bits_d, min_reduce_d);
    check_cuda_or_exit(cudaDeviceSynchronize(),
                       "Failed after reduce1FindGap");

    unsigned long long min_reduce = kNoReduction;
    check_cuda_or_exit(cudaMemcpy(&min_reduce, min_reduce_d, sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost),
                       "Failed to copy post-reduction candidate");

    if (min_reduce != kNoReduction) {
        const int from_color = static_cast<int>(min_reduce & 0xffffffffu);
        const int to_color = static_cast<int>(min_reduce >> 32);

        reduce1RecolorWorklist<<<blocks, ThreadsPerBlock>>>(
            d.color_d, d.wl_d, wlsize_d, from_color, to_color);
        check_cuda_or_exit(cudaDeviceSynchronize(),
                           "Failed after reduce1RecolorWorklist");

        reduce1RecolorHighest<<<blocks, ThreadsPerBlock>>>(
            g.nodes, d.color_d, hic, from_color);
        check_cuda_or_exit(cudaDeviceSynchronize(),
                           "Failed after reduce1RecolorHighest");
    }

    stats.runtime_sec = timer.stop();

    check_cuda_or_exit(cudaFree(wlsize_d), "Failed to free post-reduction worklist size");
    check_cuda_or_exit(cudaFree(pair_bits_d), "Failed to free post-reduction pair matrix");
    check_cuda_or_exit(cudaFree(min_reduce_d), "Failed to free post-reduction candidate");

    if (min_reduce == kNoReduction) {
        printf("post reduction status: no reducible pair found\n");
        return stats;
    }

    stats.colors_after = count_colors_from_device(g, d.color_d, color_host_buffer);
    stats.applied = stats.colors_after < stats.colors_before;

    if (!stats.applied) {
        printf("post reduction status: heuristic 1 ran but color count did not improve\n");
    }

    return stats;
}

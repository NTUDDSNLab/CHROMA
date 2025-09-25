// =============================================================================
//  SDL + Greedy  (CPU, min-heap version, 無 CUDA)
//  build : g++ -O3 -std=c++17 sdl_heap.cpp -o sdl_heap
// =============================================================================
#include "ECLgraph.h"
#include <vector>
#include <queue>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
using clk = std::chrono::high_resolution_clock;
// -----------------------------------------------------------------------------
//  Smallest-Last peeling with min-heap
// -----------------------------------------------------------------------------
static void peel_smallest_last_heap(const ECLgraph& g,
                                    std::vector<unsigned int>& deg,
                                    std::vector<unsigned int>& iter_out)
{
    const int N = g.nodes;
    iter_out.assign(N, 0);
    std::vector<char> removed(N, 0);

    using Node = std::pair<unsigned int,int>;               // (deg , v)
    auto cmp = [](const Node& a, const Node& b){return a > b;};
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> pq(cmp);

    for (int v = 0; v < N; ++v) pq.emplace(deg[v], v);

    for (int itr = 0; itr < N; ++itr)
    {
        int v;
        // int v; unsigned int d;
        do {                                 // 取尚未移除且度數最新的
            // d = pq.top().first;
            pq.top().first;
            v = pq.top().second;
            pq.pop();
        } while (removed[v]);                // 若過期 (已移/度數降) 就丟掉

        removed[v] = 1;
        iter_out[v] = itr + 1;

        for (int e = g.nindex[v]; e < g.nindex[v+1]; ++e) {
            int nei = g.nlist[e];
            if (removed[nei]) continue;
            --deg[nei];
            pq.emplace(deg[nei], nei);       // push 新鍵值，舊的留待之後丟
        }
    }
}

// -----------------------------------------------------------------------------
//  Greedy coloring (reverse SDL)
// -----------------------------------------------------------------------------
static int greedy_color(const ECLgraph& g,
                        const std::vector<unsigned int>& iter,
                        std::vector<int>& color_out)
{
    const int N = g.nodes;
    std::vector<int> order(N);
    for (int v = 0; v < N; ++v) order[iter[v]-1] = v;

    color_out.assign(N, -1);
    int maxColor = -1;
    std::vector<char> forbid;

    for (int i = N-1; i >= 0; --i) {
        int v = order[i];
        forbid.assign(maxColor + 2, 0);

        for (int e = g.nindex[v]; e < g.nindex[v+1]; ++e) {
            int nei = g.nlist[e], c = color_out[nei];
            if (c >= 0 && c < (int)forbid.size()) forbid[c] = 1;
        }
        int c = 0;
        while (c < (int)forbid.size() && forbid[c]) ++c;
        color_out[v] = c;
        maxColor = std::max(maxColor, c);
    }
    return maxColor + 1;
}

// -----------------------------------------------------------------------------
//  main
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("USAGE: %s <graph.egr>\n", argv[0]);
        return EXIT_FAILURE;
    }

    ECLgraph g = readECLgraph(argv[1]);
    printf("nodes: %d  edges: %d  avg_deg: %.2f\n",
           g.nodes, g.edges, 1.0 * g.edges / g.nodes);

    // 1. initial degree
    std::vector<unsigned int> degree(g.nodes);
    for (int v = 0; v < g.nodes; ++v) {
        degree[v] = g.nindex[v+1] - g.nindex[v];
    }
    auto t0 = clk::now();
    // 2. SDL peeling by heap
    std::vector<unsigned int> iteration;
    peel_smallest_last_heap(g, degree, iteration);
    auto t1 = clk::now();
    // 3. greedy coloring
    std::vector<int> color;
    int colors_used = greedy_color(g, iteration, color);
    auto t2 = clk::now();
    // 4. verification
    for (int v = 0; v < g.nodes; ++v)
        for (int e = g.nindex[v]; e < g.nindex[v+1]; ++e)
            if (color[v]==color[g.nlist[e]] && v!=g.nlist[e]) {
                printf("ERROR: edge (%d,%d) same color %d\n",
                       v, g.nlist[e], color[v]);
                return EXIT_FAILURE;
            }
    double pa_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ca_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    // printf("CA  time: %.3f ms\n", ca_ms);
    // printf("PA  time: %.3f ms\n", pa_ms);
    printf("runtime:    %.6f ms\n", ca_ms+pa_ms);

    printf("colors used: %d\n", colors_used);

    const int SHOW = 16;
    std::vector<int> hist(SHOW,0); int cum=0;
    for (int c : color) if (c<SHOW) ++hist[c];
    for (int i=0;i<std::min(SHOW,colors_used);++i){
        cum += hist[i];
        printf("col %2d: %6d (%5.1f%%)\n",
               i, hist[i], 100.0*cum/g.nodes);
    }
    freeECLgraph(g);
    return 0;
}

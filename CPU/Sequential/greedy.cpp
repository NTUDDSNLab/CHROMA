// =============================================================================
//  Simple Greedy Graph Coloring  (CPU, no preprocessing)
//  author : 吳建賢  (2025-06-18)
//  build  : g++ -O3 -std=c++17 greedy_cpu.cpp ECLgraph.cpp -o greedy_cpu
//
//  Two traversal orders:
//    • NATURAL   —— 0,1,2,…      (Original input order)
//    • DEG_DESC  —— Sort vertices by degree descending then color (Common improvement)
//  Switch via `ORDER_TYPE` macro in main(), default DEG_DESC.
// =============================================================================
#include "ECLgraph.h"
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#ifndef ORDER_TYPE
#define ORDER_TYPE  DEG_DESC        // NATURAL or DEG_DESC
#endif

using clk = std::chrono::high_resolution_clock;

// -----------------------------------------------------------------------------
//  Greedy coloring  (Returns number of colors)
// -----------------------------------------------------------------------------
static int greedy_color(const ECLgraph& g,
                        const std::vector<int>& order,
                        std::vector<int>& color_out)
{
    const int N = g.nodes;
    color_out.assign(N, -1);
    int maxColor = -1;

    // work array: Set of colors used by neighbors, dynamic length
    std::vector<int> used;

    for (int idx = 0; idx < N; ++idx) {
        int v = order[idx];

        used.assign(maxColor + 2, 0);
        for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
            int c = color_out[g.nlist[e]];
            if (c >= 0 && c < (int)used.size()) used[c] = 1;
        }
        int c = 0;
        while (c < (int)used.size() && used[c]) ++c;

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

    // -------- Build traversal order (order) --------
    std::vector<int> order(g.nodes);
    for (int v = 0; v < g.nodes; ++v) order[v] = v;

#if ORDER_TYPE == DEG_DESC
    std::sort(order.begin(), order.end(),
              [&](int a, int b) {
                  int da = g.nindex[a+1] - g.nindex[a];
                  int db = g.nindex[b+1] - g.nindex[b];
                  return da > db;            // Higher degree first
              });
    printf("order  : degree-descending greedy\n");
#else
    printf("order  : natural (0…N-1) greedy\n");
#endif

    // -------- Execute and time --------
    std::vector<int> color;
    auto t0 = clk::now();
    int colors_used = greedy_color(g, order, color);
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("runtime:    %.6f ms\n", ms);
    printf("colors used: %d\n", colors_used);

    // -------- Verify --------
    for (int v = 0; v < g.nodes; ++v)
        for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e)
            if (color[v] == color[g.nlist[e]] && v != g.nlist[e]) {
                printf("ERROR: edge (%d,%d) same color %d\n",
                       v, g.nlist[e], color[v]);
                return EXIT_FAILURE;
            }
    printf("verify ok\n");

    // -------- Top 16 color statistics --------
    const int SHOW = 16; std::vector<int> hist(SHOW, 0); int cum = 0;
    for (int c : color) if (c < SHOW) ++hist[c];
    for (int i = 0; i < std::min(SHOW, colors_used); ++i) {
        cum += hist[i];
        printf("col %2d: %6d (%5.1f%%)\n",
               i, hist[i], 100.0 * cum / g.nodes);
    }

    freeECLgraph(g);
    return 0;
}

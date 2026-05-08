// CPU JP-SL^M (Mutala 1983): heap-based smallest-last peel.
// Lifted from CPU/Sequential/SDL.cpp; same semantics.
#include "cpu_sdl.h"
#include <algorithm>
#include <queue>
#include <utility>

void cpu_sdl_peel(const ECLgraph& g, std::vector<unsigned int>& iter_out)
{
    const int N = g.nodes;
    std::vector<unsigned int> deg(N);
    for (int v = 0; v < N; ++v) deg[v] = g.nindex[v + 1] - g.nindex[v];

    iter_out.assign(N, 0);
    std::vector<char> removed(N, 0);

    using Node = std::pair<unsigned int, int>;          // (deg, v); ties → smaller v wins
    auto cmp = [](const Node& a, const Node& b) { return a > b; };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> pq(cmp);
    for (int v = 0; v < N; ++v) pq.emplace(deg[v], v);

    for (int itr = 0; itr < N; ++itr) {
        int v;
        do {
            v = pq.top().second;
            pq.pop();
        } while (removed[v]);

        removed[v]  = 1;
        iter_out[v] = itr + 1;

        for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
            int u = g.nlist[e];
            if (removed[u]) continue;
            --deg[u];
            pq.emplace(deg[u], u);
        }
    }
}

int cpu_sdl_greedy_color(const ECLgraph& g,
                         const std::vector<unsigned int>& iter,
                         std::vector<int>& color_out)
{
    const int N = g.nodes;
    std::vector<int> order(N);
    for (int v = 0; v < N; ++v) order[iter[v] - 1] = v;

    color_out.assign(N, -1);
    int max_color = -1;
    std::vector<char> forbid;
    for (int i = N - 1; i >= 0; --i) {
        int v = order[i];
        forbid.assign(max_color + 2, 0);
        for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
            int c = color_out[g.nlist[e]];
            if (c >= 0 && c < (int)forbid.size()) forbid[c] = 1;
        }
        int c = 0;
        while (c < (int)forbid.size() && forbid[c]) ++c;
        color_out[v] = c;
        max_color = std::max(max_color, c);
    }
    return max_color + 1;
}

#include "ECLgraph.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using clk = std::chrono::high_resolution_clock;

class DSaturColoring {
public:
    explicit DSaturColoring(const ECLgraph& graph)
        : g(graph),
          color(g.nodes, -1),
          saturation_upper(g.nodes, 0),
          uncolored_degree(g.nodes, 0),
          heap(g.nodes, -1),
          position(g.nodes, -1),
          color_mark(1, 0)
    {
        for (int v = 0; v < g.nodes; ++v) {
            uncolored_degree[v] = g.nindex[v + 1] - g.nindex[v];
            heap[v] = v;
            position[v] = v;
        }
        build_heap();
    }

    int run()
    {
        while (!heap.empty()) {
            const int u = select_next_vertex();
            const int chosen_color = smallest_available_color(u);

            color[u] = chosen_color;
            max_color = std::max(max_color, chosen_color);
            remove_top();

            for (int e = g.nindex[u]; e < g.nindex[u + 1]; ++e) {
                const int v = g.nlist[e];
                if (color[v] != -1) continue;

                saturation_upper[v] += 1;
                uncolored_degree[v] -= 1;
                sift_up(position[v]);
            }
        }

        return max_color + 1;
    }

    const std::vector<int>& colors() const
    {
        return color;
    }

private:
    const ECLgraph& g;
    std::vector<int> color;
    std::vector<int> saturation_upper;
    std::vector<int> uncolored_degree;
    std::vector<int> heap;
    std::vector<int> position;
    std::vector<uint32_t> color_mark;
    uint32_t mark_token = 1;
    int max_color = -1;

    bool higher_priority(int lhs, int rhs) const
    {
        if (saturation_upper[lhs] != saturation_upper[rhs]) {
            return saturation_upper[lhs] > saturation_upper[rhs];
        }
        if (uncolored_degree[lhs] != uncolored_degree[rhs]) {
            return uncolored_degree[lhs] > uncolored_degree[rhs];
        }
        return lhs < rhs;
    }

    void swap_heap_nodes(const int i, const int j)
    {
        std::swap(heap[i], heap[j]);
        position[heap[i]] = i;
        position[heap[j]] = j;
    }

    void sift_up(int idx)
    {
        while (idx > 0) {
            const int parent = (idx - 1) / 2;
            if (!higher_priority(heap[idx], heap[parent])) break;
            swap_heap_nodes(idx, parent);
            idx = parent;
        }
    }

    void sift_down(int idx)
    {
        const int n = static_cast<int>(heap.size());
        while (true) {
            int best = idx;
            const int left = idx * 2 + 1;
            const int right = left + 1;

            if (left < n && higher_priority(heap[left], heap[best])) best = left;
            if (right < n && higher_priority(heap[right], heap[best])) best = right;
            if (best == idx) break;

            swap_heap_nodes(idx, best);
            idx = best;
        }
    }

    void build_heap()
    {
        for (int idx = static_cast<int>(heap.size()) / 2 - 1; idx >= 0; --idx) {
            sift_down(idx);
        }
    }

    void ensure_color_mark_size(const int color_id)
    {
        if (color_id >= static_cast<int>(color_mark.size())) {
            color_mark.resize(color_id + 1, 0);
        }
    }

    uint32_t next_mark_token()
    {
        if (mark_token == std::numeric_limits<uint32_t>::max()) {
            std::fill(color_mark.begin(), color_mark.end(), 0);
            mark_token = 1;
        } else {
            ++mark_token;
        }
        return mark_token;
    }

    int exact_saturation(const int v)
    {
        if (max_color >= 0) ensure_color_mark_size(max_color);
        const uint32_t token = next_mark_token();
        int saturation = 0;

        for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
            const int neighbor = g.nlist[e];
            const int neighbor_color = color[neighbor];
            if (neighbor_color < 0) continue;

            ensure_color_mark_size(neighbor_color);
            if (color_mark[neighbor_color] != token) {
                color_mark[neighbor_color] = token;
                ++saturation;
            }
        }

        return saturation;
    }

    int smallest_available_color(const int v)
    {
        ensure_color_mark_size(max_color + 1);
        const uint32_t token = next_mark_token();

        for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
            const int neighbor_color = color[g.nlist[e]];
            if (neighbor_color < 0) continue;

            ensure_color_mark_size(neighbor_color);
            color_mark[neighbor_color] = token;
        }

        int chosen_color = 0;
        while (chosen_color <= max_color && color_mark[chosen_color] == token) {
            ++chosen_color;
        }
        return chosen_color;
    }

    int select_next_vertex()
    {
        while (true) {
            const int v = heap.front();
            const int exact = exact_saturation(v);

            if (exact == saturation_upper[v]) {
                return v;
            }

            saturation_upper[v] = exact;
            sift_down(0);
        }
    }

    void remove_top()
    {
        const int top = heap.front();
        position[top] = -1;

        if (heap.size() == 1) {
            heap.pop_back();
            return;
        }

        heap[0] = heap.back();
        position[heap[0]] = 0;
        heap.pop_back();
        sift_down(0);
    }
};

static bool verify_coloring(const ECLgraph& g, const std::vector<int>& color)
{
    for (int v = 0; v < g.nodes; ++v) {
        if (color[v] < 0) {
            std::printf("ERROR: vertex %d is uncolored\n", v);
            return false;
        }

        for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
            const int neighbor = g.nlist[e];
            if (v == neighbor) {
                std::printf("ERROR: self-loop at vertex %d\n", v);
                return false;
            }
            if (color[v] == color[neighbor]) {
                std::printf("ERROR: edge (%d,%d) same color %d\n",
                            v, neighbor, color[v]);
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::printf("USAGE: %s <graph.egr>\n", argv[0]);
        return EXIT_FAILURE;
    }

    ECLgraph g = readECLgraph(argv[1]);
    std::printf("nodes: %d  edges: %d  avg_deg: %.2f\n",
                g.nodes, g.edges, 1.0 * g.edges / g.nodes);
    std::printf("order  : DSatur (CSR)\n");

    DSaturColoring solver(g);

    const auto start = clk::now();
    const int colors_used = solver.run();
    const auto end = clk::now();

    const double runtime_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::printf("runtime:    %.6f ms\n", runtime_ms);
    std::printf("colors used: %d\n", colors_used);

    const bool ok = verify_coloring(g, solver.colors());
    std::printf("%s\n", ok ? "result verification passed"
                            : "result verification FAILED");

    freeECLgraph(g);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

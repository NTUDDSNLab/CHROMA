// lib/io/graph_features.cpp
#include "graph_features.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

GraphFeatures compute_graph_features(const ECLgraph& g)
{
    GraphFeatures out{};
    out.V = static_cast<double>(g.nodes);
    out.E = static_cast<double>(g.edges);

    if (g.nodes <= 0) {
        for (int i = 0; i < 9; ++i) out.as_array[i] = 0.0;
        out.as_array[0] = out.V;
        out.as_array[1] = out.E;
        return out;
    }

    // Pass 1: degree array, mean, min, max, Σ(deg-d)²
    std::vector<int64_t> deg(g.nodes);
    int64_t sum_deg = 0;
    int64_t min_d   = INT64_MAX;
    int64_t max_d   = 0;
    for (int v = 0; v < g.nodes; ++v) {
        int64_t dv = static_cast<int64_t>(g.nindex[v + 1] - g.nindex[v]);
        deg[v]   = dv;
        sum_deg += dv;
        if (dv < min_d) min_d = dv;
        if (dv > max_d) max_d = dv;
    }
    const double n      = static_cast<double>(g.nodes);
    const double m_dir  = static_cast<double>(sum_deg);   // == g.edges in well-formed .egr
    const double d_mean = m_dir / n;

    long double var_acc = 0.0L;
    for (int v = 0; v < g.nodes; ++v) {
        long double diff = static_cast<long double>(deg[v]) - d_mean;
        var_acc += diff * diff;
    }
    const double s = std::sqrt(static_cast<double>(var_acc / n));
    const double R = (d_mean > 0.0) ? (static_cast<double>(max_d - min_d) / d_mean) : 0.0;

    // Pass 2: sort for Gini sorted-rank form
    std::vector<double> sorted_d(g.nodes);
    for (int v = 0; v < g.nodes; ++v) sorted_d[v] = static_cast<double>(deg[v]);
    std::sort(sorted_d.begin(), sorted_d.end());

    long double gini_num = 0.0L;
    for (int i = 0; i < g.nodes; ++i) {
        long double coeff = static_cast<long double>(2 * (i + 1) - g.nodes - 1);
        gini_num += coeff * sorted_d[i];
    }
    const double GI = (sum_deg > 0)
                      ? static_cast<double>(gini_num / (n * static_cast<long double>(sum_deg)))
                      : 0.0;

    // Pass 3: entropy
    long double H_acc = 0.0L;
    if (g.nodes > 1 && sum_deg > 0) {
        for (int v = 0; v < g.nodes; ++v) {
            if (deg[v] == 0) continue;
            long double p = static_cast<long double>(deg[v]) / m_dir;
            H_acc += p * std::log2(static_cast<double>(p));
        }
    }
    const double H_er = (g.nodes > 1)
                        ? static_cast<double>(-H_acc / std::log2(n))
                        : 0.0;

    // Pass 4: degeneracy (k-core max) via bucket-based peeling — Batagelj & Zaversnik 2003.
    // O(V + E). Repeatedly remove the minimum-degree vertex; track the maximum
    // degree seen at removal time. That maximum is the degeneracy.
    double kcore = 0.0;
    {
        std::vector<int64_t> peel_deg = deg;          // copy so above passes are unaffected
        const int64_t max_d_init = max_d;
        // bucket[d] is a flat list of vertices currently with degree d.
        // pos[v] is v's index in bucket[peel_deg[v]] for O(1) removal.
        std::vector<std::vector<int>> bucket(max_d_init + 2);
        std::vector<int> pos(g.nodes, 0);
        for (int v = 0; v < g.nodes; ++v) {
            bucket[peel_deg[v]].push_back(v);
            pos[v] = static_cast<int>(bucket[peel_deg[v]].size()) - 1;
        }
        std::vector<char> removed(g.nodes, 0);
        int64_t cur_d = 0;
        int64_t degeneracy = 0;
        for (int iter = 0; iter < g.nodes; ++iter) {
            while (cur_d <= max_d_init && bucket[cur_d].empty()) ++cur_d;
            if (cur_d > max_d_init) break;
            const int v = bucket[cur_d].back();
            bucket[cur_d].pop_back();
            removed[v] = 1;
            if (cur_d > degeneracy) degeneracy = cur_d;
            for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
                const int u = g.nlist[e];
                if (removed[u]) continue;
                const int64_t old = peel_deg[u];
                if (old <= cur_d) continue;            // already at-or-below current peel level
                // Remove u from bucket[old]
                auto& ob = bucket[old];
                const int p_u = pos[u];
                const int last = ob.back();
                ob[p_u] = last;
                pos[last] = p_u;
                ob.pop_back();
                peel_deg[u] = old - 1;
                bucket[old - 1].push_back(u);
                pos[u] = static_cast<int>(bucket[old - 1].size()) - 1;
                if (old - 1 < cur_d) cur_d = old - 1;  // demoted below current scan: rewind
            }
        }
        kcore = static_cast<double>(degeneracy);
    }

    // Pass 5: assortativity = Pearson correlation of (deg(u), deg(v)) over directed edges
    // Newman 2002 §III, eq. (4). All-constant degrees → undefined; return 0.
    double assort = 0.0;
    if (g.edges > 0 && d_mean > 0.0 && (max_d > min_d)) {
        long double sum_x  = 0.0L, sum_y = 0.0L;
        long double sum_xx = 0.0L, sum_yy = 0.0L, sum_xy = 0.0L;
        for (int v = 0; v < g.nodes; ++v) {
            const long double dv = static_cast<long double>(deg[v]);
            for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
                const long double du = static_cast<long double>(deg[g.nlist[e]]);
                sum_x  += dv;
                sum_y  += du;
                sum_xx += dv * dv;
                sum_yy += du * du;
                sum_xy += dv * du;
            }
        }
        const long double M     = static_cast<long double>(g.edges);
        const long double mean_x = sum_x / M;
        const long double mean_y = sum_y / M;
        const long double var_x  = sum_xx / M - mean_x * mean_x;
        const long double var_y  = sum_yy / M - mean_y * mean_y;
        const long double cov    = sum_xy / M - mean_x * mean_y;
        if (var_x > 0.0L && var_y > 0.0L) {
            assort = static_cast<double>(cov / std::sqrt(static_cast<double>(var_x * var_y)));
        }
    }

    out.d     = d_mean;
    out.s     = s;
    out.R     = R;
    out.GI    = GI;
    out.H_er  = H_er;
    out.kcore = kcore;
    out.assort= assort;
    out.as_array[0] = out.V;
    out.as_array[1] = out.E;
    out.as_array[2] = out.d;
    out.as_array[3] = out.s;
    out.as_array[4] = out.R;
    out.as_array[5] = out.GI;
    out.as_array[6] = out.H_er;
    out.as_array[7] = out.kcore;
    out.as_array[8] = out.assort;
    return out;
}

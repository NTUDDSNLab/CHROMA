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
        for (int i = 0; i < 7; ++i) out.as_array[i] = 0.0;
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

    out.d   = d_mean;
    out.s   = s;
    out.R   = R;
    out.GI  = GI;
    out.H_er= H_er;
    out.as_array[0] = out.V;
    out.as_array[1] = out.E;
    out.as_array[2] = out.d;
    out.as_array[3] = out.s;
    out.as_array[4] = out.R;
    out.as_array[5] = out.GI;
    out.as_array[6] = out.H_er;
    return out;
}

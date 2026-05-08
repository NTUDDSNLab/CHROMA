// consistency_metric.cpp
// Computes:
//   * Consistency Ratio C/T per CHROMA paper Eq. (1)-(3)
//     T = n(n-1) ordered pairs, C = #ordered pairs with strict sign-match
//     in (P_ref - P_ref)(P_test - P_test) > 0  (any tie => not concordant)
//   * Adjacent-pair tie ratio for each priority list:
//     #directed-half-edges (u,v) with P[u] == P[v] / total directed-half-edges
//
// USAGE: consistency_metric <graph.egr> <ref.bin> <test.bin>
// Output: a single JSON line on stdout
//
// build:
//   g++ -O3 -std=c++17 -I../lib/io consistency_metric.cpp -o consistency_metric

#include "ECLgraph.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using clk = std::chrono::high_resolution_clock;

class Fenwick {
public:
    explicit Fenwick(size_t n) : tree_(n + 1, 0), n_(n) {}
    void update(size_t i) {
        for (size_t k = i + 1; k <= n_; k += k & (-k)) ++tree_[k];
    }
    uint64_t prefix(int64_t i) const {  // sum of [0..i], or 0 if i<0
        uint64_t s = 0;
        for (int64_t k = i + 1; k > 0; k -= k & (-k)) s += tree_[k];
        return s;
    }
private:
    std::vector<uint32_t> tree_;
    size_t n_;
};

static std::vector<unsigned int> read_priority(const char* path, size_t n) {
    std::vector<unsigned int> p(n);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
    if (fread(p.data(), sizeof(unsigned int), n, f) != n) {
        fprintf(stderr, "ERROR: short read on %s\n", path); exit(1);
    }
    fclose(f);
    return p;
}

// Dense rank assignment via sort: rank_out[v] in [0, K), K = #distinct values
static size_t densify(const std::vector<unsigned int>& p,
                      std::vector<uint32_t>& rank_out) {
    const size_t N = p.size();
    rank_out.assign(N, 0);
    std::vector<uint32_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(),
              [&](uint32_t a, uint32_t b) { return p[a] < p[b]; });
    uint32_t r = 0;
    rank_out[idx[0]] = 0;
    for (size_t i = 1; i < N; ++i) {
        if (p[idx[i]] != p[idx[i - 1]]) ++r;
        rank_out[idx[i]] = r;
    }
    return size_t(r) + 1;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr,
                "USAGE: %s <graph.egr> <ref_priority.bin> <test_priority.bin>\n",
                argv[0]);
        return 1;
    }
    const char* graph_path = argv[1];
    const char* ref_path   = argv[2];
    const char* test_path  = argv[3];

    auto t0 = clk::now();
    ECLgraph g = readECLgraph(graph_path);
    const int  N = g.nodes;
    const long long M = g.edges;
    auto p_ref  = read_priority(ref_path,  N);
    auto p_test = read_priority(test_path, N);
    auto t_load = clk::now();

    // ---------- Tie ratio over directed half-edges ----------
    uint64_t tied_ref = 0, tied_test = 0;
    for (int v = 0; v < N; ++v) {
        for (int e = g.nindex[v]; e < g.nindex[v + 1]; ++e) {
            int u = g.nlist[e];
            if (p_ref [v] == p_ref [u]) ++tied_ref;
            if (p_test[v] == p_test[u]) ++tied_test;
        }
    }
    double tie_ratio_ref  = (M > 0) ? double(tied_ref ) / double(M) : 0.0;
    double tie_ratio_test = (M > 0) ? double(tied_test) / double(M) : 0.0;
    auto t_tie = clk::now();

    // ---------- Consistency ratio ----------
    // Concordant pair (i,j), i!=j: (P_ref[i] - P_ref[j])(P_test[i] - P_test[j]) > 0
    // Strategy:
    //   1. Densify P_test ranks.
    //   2. Sort vertex indices by P_ref ascending.
    //   3. Process equal-P_ref groups in order. For each vertex in current group,
    //      query Fenwick tree for #inserted-ranks strictly less than rank_test[v];
    //      this counts ordered pairs (j, v) where P_ref[j] < P_ref[v] AND
    //      P_test[j] < P_test[v]. Then insert all group vertices.
    //   4. Sum is the count of unordered strictly-concordant pairs.
    //   5. Multiply by 2 for ordered C, divide by T = N(N-1).
    std::vector<uint32_t> rank_test;
    size_t K_test = densify(p_test, rank_test);

    std::vector<uint32_t> rank_ref;
    size_t K_ref = densify(p_ref, rank_ref);

    std::vector<uint32_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(),
              [&](uint32_t a, uint32_t b) { return p_ref[a] < p_ref[b]; });

    Fenwick bit(K_test);
    uint64_t concordant_unord = 0;
    int i = 0;
    while (i < N) {
        int j = i;
        const unsigned int g_val = p_ref[idx[i]];
        while (j < N && p_ref[idx[j]] == g_val) ++j;
        // Query: count inserted ranks strictly less than rank_test[v]
        for (int k = i; k < j; ++k) {
            uint32_t r = rank_test[idx[k]];
            if (r > 0) concordant_unord += bit.prefix(int64_t(r) - 1);
        }
        // Insert
        for (int k = i; k < j; ++k) bit.update(rank_test[idx[k]]);
        i = j;
    }

    long double T_ord = (long double)N * (long double)(N - 1);
    long double C_ord = (long double)concordant_unord * 2.0L;
    double consistency = (N > 1) ? double(C_ord / T_ord) : 0.0;
    auto t_done = clk::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // JSON line
    printf("{\"graph\":\"%s\",\"nodes\":%d,\"edges_directed\":%lld,"
           "\"consistency_ratio\":%.10f,"
           "\"tie_ratio_ref\":%.10f,\"tie_ratio_test\":%.10f,"
           "\"distinct_ref\":%zu,\"distinct_test\":%zu,"
           "\"tied_edges_ref\":%llu,\"tied_edges_test\":%llu,"
           "\"concordant_unord_pairs\":%llu,"
           "\"load_ms\":%.1f,\"tie_ms\":%.1f,\"consist_ms\":%.1f}\n",
           graph_path, N, M,
           consistency, tie_ratio_ref, tie_ratio_test,
           K_ref, K_test,
           (unsigned long long)tied_ref, (unsigned long long)tied_test,
           (unsigned long long)concordant_unord,
           ms(t0, t_load), ms(t_load, t_tie), ms(t_tie, t_done));

    freeECLgraph(g);
    return 0;
}

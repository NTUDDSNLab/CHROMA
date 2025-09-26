// Lightweight pluggable graph partitioners for CHROMA_RGP
// Provides: round_robin, random (balanced), LDG (streaming), and METIS (if available)

#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cctype>
#include <limits>
#include <cmath>

#include "ECLgraph.h"

#ifdef HAVE_METIS
#include <metis.h>
#endif

namespace partitioning {

enum class Method { Metis, RoundRobin, Random, LDG };

inline std::string to_lower(std::string s) {
  for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

inline bool parse_method(const std::string &name, Method &out) {
  const std::string nm = to_lower(name);
  if (nm == "metis") { out = Method::Metis; return true; }
  if (nm == "round_robin" || nm == "rr") { out = Method::RoundRobin; return true; }
  if (nm == "random" || nm == "rand") { out = Method::Random; return true; }
  if (nm == "ldg") { out = Method::LDG; return true; }
  return false;
}

inline bool compute_partition_round_robin(const ECLgraph &g, int nParts, std::vector<int> &part) {
  part.resize(g.nodes);
  for (int v = 0; v < g.nodes; ++v) part[v] = (nParts > 0) ? (v % nParts) : 0;
  return true;
}

inline bool compute_partition_random_balanced(const ECLgraph &g, int nParts, std::vector<int> &part) {
  if (nParts <= 0) return false;
  part.resize(g.nodes);
  std::vector<int> order(g.nodes);
  for (int v = 0; v < g.nodes; ++v) order[v] = v;
  std::mt19937 rng(42);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<int> counts(nParts, 0);
  const int base = g.nodes / nParts;
  int rem = g.nodes % nParts;
  std::vector<int> caps(nParts, base);
  for (int p = 0; p < nParts && rem > 0; ++p, --rem) caps[p]++;
  int idx = 0;
  for (int p = 0; p < nParts; ++p) {
    for (int c = 0; c < caps[p]; ++c) {
      int v = order[idx++];
      part[v] = p;
      counts[p]++;
    }
  }
  return true;
}

// Linear Deterministic Greedy (LDG) streaming partitioning
inline bool compute_partition_ldg(const ECLgraph &g, int nParts, std::vector<int> &part) {
  if (nParts <= 0) return false;
  part.assign(g.nodes, -1);
  std::vector<int> counts(nParts, 0);
  const int maxCap = (int)std::ceil((double)g.nodes / (double)nParts); // enforce near-perfect balance
  const double lambda = 1.0; // balance penalty weight

  for (int v = 0; v < g.nodes; ++v) {
    std::vector<int> neighCount(nParts, 0);
    for (int j = g.nindex[v]; j < g.nindex[v + 1]; ++j) {
      int u = g.nlist[j];
      int pu = (u >= 0 && u < g.nodes) ? part[u] : -1;
      if (pu >= 0) neighCount[pu]++;
    }
    int bestP = 0;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (int p = 0; p < nParts; ++p) {
      double bal = (counts[p] < maxCap) ? (counts[p] / (double)maxCap) : 1.0;
      double score = (double)neighCount[p] - lambda * bal;
      // prefer less filled on ties
      if (score > bestScore || (std::abs(score - bestScore) < 1e-12 && counts[p] < counts[bestP])) {
        bestScore = score;
        bestP = p;
      }
    }
    // If chosen is at capacity, place in smallest count below capacity
    if (counts[bestP] >= maxCap) {
      int candidate = -1;
      for (int p = 0; p < nParts; ++p) {
        if (counts[p] < maxCap && (candidate < 0 || counts[p] < counts[candidate])) candidate = p;
      }
      if (candidate >= 0) bestP = candidate; else {
        // all at capacity, pick smallest (should not happen if maxCap is ceil(N/k))
        bestP = static_cast<int>(std::min_element(counts.begin(), counts.end()) - counts.begin());
      }
    }
    part[v] = bestP;
    counts[bestP]++;
  }
  return true;
}

inline bool compute_partition(const ECLgraph &g,
                              int nParts,
                              Method method,
                              std::vector<int> &part,
                              std::string *err = nullptr) {
  if (g.nodes < 1 || nParts < 1) {
    if (err) *err = "Invalid graph or number of parts";
    return false;
  }
  if (nParts == 1) {
    part.assign(g.nodes, 0);
    return true;
  }

  switch (method) {
    case Method::RoundRobin:
      return compute_partition_round_robin(g, nParts, part);
    case Method::Random:
      return compute_partition_random_balanced(g, nParts, part);
    case Method::LDG:
      return compute_partition_ldg(g, nParts, part);
    case Method::Metis:
#ifdef HAVE_METIS
      {
        idx_t nVertices = g.nodes;
        idx_t ncon = 1;
        idx_t objval = 0;
        std::vector<idx_t> partIdx(nVertices, 0);
        std::vector<real_t> tpwgts((size_t)nParts * ncon, 1.0 / (real_t)nParts);
        std::vector<real_t> ubvec(ncon, 1.05);
        int ret = METIS_PartGraphKway(
          &nVertices,
          &ncon,
          (idx_t*)g.nindex,
          (idx_t*)g.nlist,
          nullptr, nullptr, nullptr,
          (idx_t*)&nParts,
          tpwgts.data(),
          ubvec.data(),
          nullptr,
          &objval,
          partIdx.data()
        );
        if (ret != METIS_OK) {
          if (err) *err = "METIS_PartGraphKway returned error code " + std::to_string(ret);
          return false;
        }
        part.resize(g.nodes);
        for (int i = 0; i < g.nodes; ++i) part[i] = (int)partIdx[i];
        return true;
      }
#else
      if (err) *err = "METIS not available (rebuild with -DHAVE_METIS and link METIS)";
      return false;
#endif
  }
  if (err) *err = "Unknown method";
  return false;
}

} // namespace partitioning

// lib/io/graph_features.h
#pragma once
#include "ECLgraph.h"

// Seven features matched to model/features.py (FEATURE_NAMES). Order matters
// — must agree with the order produced by model/features.py:compute_features
// and consumed by score(double input[]) emitted by m2cgen.
//
//   index | name | meaning
//   ------+------+----------------------------------------------------------
//     0   |  V   | nodes
//     1   |  E   | directed edges (== g.edges)
//     2   |  d   | mean degree (= E / V)
//     3   |  s   | population stdev of degrees
//     4   |  R   | (max_deg - min_deg) / d ; 0 if d == 0
//     5   |  GI  | Gini index, sorted-rank form
//     6   | H_er | (-Σ pᵢ log₂ pᵢ) / log₂ V, pᵢ = deg(v) / E
struct GraphFeatures {
    double V, E, d, s, R, GI, H_er;
    double as_array[7];
};

GraphFeatures compute_graph_features(const ECLgraph& g);

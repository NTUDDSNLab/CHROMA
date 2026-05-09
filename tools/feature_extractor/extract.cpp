// tools/feature_extractor/extract.cpp
//
// USAGE:  extract <graph.egr>
//
// Reads the .egr, computes the 7 GraphFeatures, prints them as a
// single-line JSON object on stdout. Used by the Python ↔ C++ parity
// test (tests/test_features_cpp.py).
#include "graph_features.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "USAGE: %s <graph.egr>\n", argv[0]);
        return 2;
    }
    ECLgraph g = readECLgraph(argv[1]);
    GraphFeatures f = compute_graph_features(g);
    printf("{\"V\":%.17g,\"E\":%.17g,\"d\":%.17g,\"s\":%.17g,"
           "\"R\":%.17g,\"GI\":%.17g,\"H_er\":%.17g}\n",
           f.V, f.E, f.d, f.s, f.R, f.GI, f.H_er);
    freeECLgraph(g);
    return 0;
}

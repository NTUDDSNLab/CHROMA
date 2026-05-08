#pragma once
#include "ECLgraph.h"
#include <vector>

// CPU JP-SL^M peel: heap-based smallest-last, one vertex per iter.
// iter_out[v] in [1..N] (each unique). Higher iter_out = peeled later =
// colored first (matches the convention used by CHROMA's iteration_list).
void cpu_sdl_peel(const ECLgraph& g, std::vector<unsigned int>& iter_out);

// Greedy color in REVERSE peel order, returns # colors used.
// color_out[v] = assigned color (0-indexed).
int cpu_sdl_greedy_color(const ECLgraph& g,
                         const std::vector<unsigned int>& iter,
                         std::vector<int>& color_out);

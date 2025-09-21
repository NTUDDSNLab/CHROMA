// ============================================================================
//  greedy_kokkos.cpp  –  Vertex-Based (VB_VIT) parallel greedy coloring (CPU)
//
//  build : g++ -O3 -fopenmp -std=c++17 greedy_kokkos.cpp ECLgraph.cpp        \
//          -I${KOKKOS_ROOT}/include -I${KOKKOS_KERNELS_ROOT}/include         \
//          -L${KOKKOS_ROOT}/lib -lkokkos -lkokkosalgorithms -lkokkoskernels
// ============================================================================
#include "ECLgraph.h"
#include <Kokkos_Core.hpp>
#include <KokkosGraph_Distance1Color.hpp>

using Ordinal   = int;                      // 頂點 (row) index type
using SizeType  = int;                      // rowmap offset type
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("USAGE: %s <graph.egr>\n", argv[0]);
    return EXIT_FAILURE;
  }

  // ---- 讀檔 ----------------------------------------------------------------
  ECLgraph g = readECLgraph(argv[1]);
  printf("nodes: %d  edges: %d  avg_deg: %.2f\n",
         g.nodes, g.edges, 1.0 * g.edges / g.nodes);

  // ---- Kokkos 初始化 --------------------------------------------------------
  Kokkos::initialize(argc, argv);
  {
    // 1) 將 ECLgraph → Kokkos::View (CSR)
    Kokkos::View<SizeType*, MemSpace> rowmap_d("rowmap", g.nodes + 1);
    Kokkos::View<Ordinal*,  MemSpace> colinds_d("colinds", g.edges);

    auto rowmap_h  = Kokkos::create_mirror_view(rowmap_d);
    auto colinds_h = Kokkos::create_mirror_view(colinds_d);

    for (int i = 0; i <= g.nodes; ++i) rowmap_h(i)  = g.nindex[i];
    for (int i = 0; i <  g.edges; ++i) colinds_h(i) = g.nlist[i];

    Kokkos::deep_copy(rowmap_d,  rowmap_h);
    Kokkos::deep_copy(colinds_d, colinds_h);

    // 2) 建立 handle，指定 VB_VIT
    using Handle = KokkosGraph::Experimental::KokkosKernelsHandle<
                     Ordinal, Ordinal, SizeType,
                     ExecSpace, MemSpace, MemSpace>;

    Handle handle;
    handle.create_graph_coloring_handle(KokkosGraph::COLORING_VB);   // VB_VIT

    // 3) 執行 coloring
    auto t0 = ExecSpace().impl_static_fence();   // 記時前 barrier
    KokkosGraph::Experimental::graph_color(
        &handle, g.nodes, g.nodes, rowmap_d, colinds_d);             // :contentReference[oaicite:0]{index=0}
    auto t1 = ExecSpace().impl_static_fence();

    // 取結果
    auto d_colors   = handle.get_graph_coloring_handle()->get_vertex_colors();
    int  numColors  = handle.get_graph_coloring_handle()->get_num_colors();

    auto h_colors =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_colors);

    double ms =
      Kokkos::duration<double, std::milli>(t1 - t0);   // 簡單 CPU 計時
    printf("runtime: %.3f ms   colors used: %d\n", ms, numColors);

    // ---- 驗證 --------------------------------------------------------------
    bool ok = true;
    for (int v = 0; v < g.nodes && ok; ++v)
      for (int e = g.nindex[v]; e < g.nindex[v + 1] && ok; ++e)
        if (h_colors(v) == h_colors(g.nlist[e]) && v != g.nlist[e])
          ok = false;
    printf("%s\n", ok ? "verify ok" : "verify FAILED");

    handle.destroy_graph_coloring_handle();
  }
  Kokkos::finalize();
  freeECLgraph(g);
  return 0;
}

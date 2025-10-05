/* 

This program is used to convert the SNAP dataset to build multiple partitons with METIS.
Output format is SNAP format.
    # Undirected graph: <dataset_name> + "_<partition_number.txt>"
    # Nodes: <number_of_nodes> Edges: <number_of_edges>
    # FromNodeID    ToNodeID
    # <FromNodeID>    <ToNodeID>
    # ...
    # ...

The partition name of each partition is <dataset_name> + "_<partition_number.txt>".
E.g. facebook_1.txt, facebook_2.txt, ...

Usage:
    ./snap2partition <dataset_name> <partition_number> <output_dir>

Example:
    ./snap2partition facebook 10 output_dir

*/

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <sstream>

#include "metis.h"
#include "ECLgraph.h"
#include "graph.h"

namespace fs = std::filesystem;

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <dataset> <num_partitions> <output_dir>\n";
}

static fs::path resolve_dataset_path(const std::string& arg) {
    fs::path p(arg);
    // If an explicit file path exists, use it directly
    if (fs::exists(p) && fs::is_regular_file(p)) return p;
    // Try relative to Datasets/SNAP
    fs::path snap_dir = fs::path("Datasets") / "SNAP";
    if (p.extension().empty()) {
        fs::path candidate = snap_dir / (p.string() + ".txt");
        if (fs::exists(candidate)) return candidate;
        // fallback: name without extension under SNAP
        candidate = snap_dir / p;
        if (fs::exists(candidate)) return candidate;
    } else {
        fs::path candidate = snap_dir / p;
        if (fs::exists(candidate)) return candidate;
    }
    return p; // return original (will error later if not found)
}

static std::string stem_basename(const fs::path& p) {
    return p.stem().string();
}

// Read SNAP .txt and build a compact 0..N-1 ECLgraph (undirected by default).
static bool read_snap_to_ecl(const fs::path& path, ECLgraph& g, bool undirected = true) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open SNAP file: " << path << "\n";
        return false;
    }
    std::vector<std::pair<int,int>> edges;
    edges.reserve(1 << 20);
    std::unordered_map<int,int> id2c;
    id2c.reserve(1 << 16);
    std::string line;
    auto get_cid = [&id2c](int id) -> int {
        auto it = id2c.find(id);
        if (it != id2c.end()) return it->second;
        int cid = (int)id2c.size();
        id2c.emplace(id, cid);
        return cid;
    };
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        int u, v;
        std::istringstream iss(line);
        if (!(iss >> u >> v)) continue;
        int cu = get_cid(u);
        int cv = get_cid(v);
        if (cu == cv) continue;
        edges.emplace_back(cu, cv);
        if (undirected) edges.emplace_back(cv, cu);
    }
    const int N = (int)id2c.size();
    std::vector<std::vector<int>> adj(N);
    for (auto &e : edges) adj[e.first].push_back(e.second);
    size_t m = 0;
    for (int i = 0; i < N; ++i) {
        auto &nbrs = adj[i];
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        m += nbrs.size();
    }
    g.nodes = N;
    g.edges = (int)m;
    g.nindex = (int*)malloc(sizeof(int) * (N + 1));
    g.nlist  = (int*)malloc(sizeof(int) * m);
    g.eweight = (int*)malloc(sizeof(int) * m);
    size_t off = 0;
    g.nindex[0] = 0;
    for (int i = 0; i < N; ++i) {
        auto &nbrs = adj[i];
        for (int v : nbrs) {
            g.nlist[off] = v;
            g.eweight[off] = 0;
            ++off;
        }
        g.nindex[i+1] = (int)off;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        usage(argv[0]);
        return 1;
    }

    const std::string dataset_arg = argv[1];
    const int nParts = std::max(0, std::atoi(argv[2]));
    const fs::path out_dir(argv[3]);

    if (nParts <= 0) {
        std::cerr << "num_partitions must be > 0\n";
        return 1;
    }

    const fs::path ds_path = resolve_dataset_path(dataset_arg);
    if (!fs::exists(ds_path)) {
        std::cerr << "Dataset not found: " << ds_path << "\n";
        return 1;
    }

    try { fs::create_directories(out_dir); } catch (const std::exception& e) {
        std::cerr << "Failed to create output dir: " << out_dir << ": " << e.what() << "\n";
        return 1;
    }

    // Build compact ECLgraph from SNAP .txt (undirected)
    ECLgraph g{};
    if (!read_snap_to_ecl(ds_path, g, /*undirected=*/true)) {
        return 1;
    }

    // Prepare METIS inputs
    idx_t nvtxs = g.nodes;
    idx_t ncon = 1;
    idx_t nparts = nParts;
    idx_t objval = 0;
    std::vector<idx_t> part(nvtxs, 0);

    // METIS expects idx_t pointers
    idx_t* xadj = reinterpret_cast<idx_t*>(g.nindex);
    idx_t* adjncy = reinterpret_cast<idx_t*>(g.nlist);

    int metis_ret = METIS_PartGraphKway(
        &nvtxs, &ncon,
        xadj, adjncy,
        /*vwgt*/ nullptr, /*vsize*/ nullptr, /*adjwgt*/ nullptr,
        &nparts,
        /*tpwgts*/ nullptr,
        /*ubvec*/ nullptr,
        /*options*/ nullptr,
        &objval,
        part.data()
    );
    if (metis_ret != METIS_OK) {
        std::cerr << "METIS_PartGraphKway failed with code " << metis_ret << "\n";
        freeECLgraph(g);
        return 2;
    }

    // Build per-part mappings and write SNAP files
    const std::string base = stem_basename(ds_path);

    // Build list of nodes per part and mapping global->local
    std::vector<std::vector<int>> nodes_in_part(nParts);
    nodes_in_part.reserve(nParts);
    for (int v = 0; v < g.nodes; ++v) {
        int p = part[v];
        if (p < 0 || p >= nParts) continue;
        nodes_in_part[p].push_back(v);
    }

    // Precompute global->local maps
    std::vector<std::unordered_map<int,int>> g2l(nParts);
    for (int p = 0; p < nParts; ++p) {
        auto& map = g2l[p];
        map.reserve(nodes_in_part[p].size());
        int lid = 0;
        for (int v : nodes_in_part[p]) map.emplace(v, lid++);
    }

    for (int p = 0; p < nParts; ++p) {
        const auto& vertices = nodes_in_part[p];
        const auto& map = g2l[p];
        const int local_nodes = static_cast<int>(vertices.size());

        // Collect unique undirected edges inside the partition
        // Policy: since adjacency is symmetric, only output (u,v) where local_u < local_v
        std::vector<std::pair<int,int>> edges;
        edges.reserve(local_nodes * 2); // heuristic
        for (int gv : vertices) {
            const int lv = map.at(gv);
            for (int j = g.nindex[gv]; j < g.nindex[gv + 1]; ++j) {
                int gu = g.nlist[j];
                if (part[gu] != p) continue;
                auto it = map.find(gu);
                if (it == map.end()) continue;
                int lu = it->second;
                if (lv < lu) edges.emplace_back(lv, lu);
            }
        }

        // Optional: sort edges for stable output and remove accidental dups
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

        // Compose filename and write SNAP-format file (0-based local IDs)
        fs::path out_path = out_dir / (base + "_" + std::to_string(p + 1) + ".txt");
        std::ofstream ofs(out_path);
        if (!ofs) {
            std::cerr << "Failed to open output file: " << out_path << "\n";
            freeECLgraph(g);
            return 3;
        }
        ofs << "# Undirected graph: " << out_path.filename().string() << "\n";
        ofs << "# Nodes: " << local_nodes << " Edges: " << edges.size() << "\n";
        ofs << "# FromNodeID\tToNodeID\n";
        for (const auto& e : edges) {
            ofs << e.first << "\t" << e.second << "\n";
        }
        ofs.close();
        std::cout << "Wrote partition " << (p+1) << "/" << nParts << ": "
                  << out_path << " (nodes=" << local_nodes
                  << ", edges=" << edges.size() << ")\n";
    }

    freeECLgraph(g);
    return 0;
}

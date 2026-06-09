// CPM over-merge driver (verification only; NOT built by CMake).
//
// Reads an undirected edge list and runs the GVE bridge `runGveLeiden` with the
// modularity or CPM objective, writing per-node community ids. Used by
// cpm_overmerge.py to prove CPM fixes the modularity resolution-limit over-merge
// on a fragmented entity-resolution graph, independent of the engine/wheel.
//
// Build inside the container (from the engine root):
//   g++ -std=c++17 -O3 -fopenmp -DTYPE=float \
//       -Iextension/algo/src/include \
//       extension/algo/src/function/gve_leiden.cpp \
//       verification/leiden/cpm_overmerge_driver.cpp -o /tmp/cpm_overmerge
//
// Usage: cpm_overmerge <edges.csv> <numNodes> <mod|cpm> <param> <out.csv>
//   edges.csv : one "src,dst" undirected edge per line (0-indexed, no header)
//   param     : resolution (mod) or gamma (cpm)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "function/gve_leiden.h"

using namespace lbug::algo_extension;

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <edges.csv> <numNodes> <mod|cpm> <param> <out.csv>\n", argv[0]);
        return 2;
    }
    const char* edgePath = argv[1];
    const uint64_t numNodes = strtoull(argv[2], nullptr, 10);
    const bool useCpm = (strcmp(argv[3], "cpm") == 0);
    const double param = atof(argv[4]);
    const char* outPath = argv[5];

    FILE* f = fopen(edgePath, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", edgePath);
        return 2;
    }
    std::vector<GveEdge> edges;
    long a, b;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%ld,%ld", &a, &b) == 2) {
            edges.push_back(GveEdge{(uint64_t)a, (uint64_t)b, 1.0f});
        }
    }
    fclose(f);

    // resolution for modularity, gamma for cpm.
    const double resolution = useCpm ? 1.0 : param;
    const double gamma = useCpm ? param : 1.0;
    auto comm = runGveLeiden(numNodes, edges, resolution, 4, useCpm, gamma);

    FILE* o = fopen(outPath, "w");
    fprintf(o, "node,community\n");
    for (uint64_t i = 0; i < numNodes; ++i) {
        fprintf(o, "%llu,%llu\n", (unsigned long long)i, (unsigned long long)comm[i]);
    }
    fclose(o);
    fprintf(stderr, "%s param=%g numNodes=%llu edges=%zu -> %s\n", argv[3], param,
        (unsigned long long)numNodes, edges.size(), outPath);
    return 0;
}

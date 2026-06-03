// Isolated validation driver for the GVE bridge (NOT built by CMake; used only
// for ad-hoc verification that runGveLeiden reproduces the standalone GVE-Leiden
// result on the fragmented ER match graph, independent of the engine).
//
// Build (inside a container with the GVE headers + bridge staged):
//   g++ -std=c++17 -O3 -fopenmp -DTYPE=float -I<inc> gve_leiden.cpp gve_bridge_driver.cpp -o t
// Run: ./t er.mtx out.csv   (reads 1-indexed MTX, writes record_id,community)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "function/gve_leiden.h"

using namespace lbug::algo_extension;

int main(int argc, char** argv) {
    const char* mtx = argc > 1 ? argv[1] : "/root/work/er.mtx";
    const char* outp = argc > 2 ? argv[2] : "/root/work/membership_bridge.csv";
    FILE* f = fopen(mtx, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", mtx);
        return 2;
    }
    char line[512];
    if (!fgets(line, sizeof line, f)) {
        return 2; // banner
    }
    long n = 0, m = 0, nnz = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '%') {
            continue;
        }
        sscanf(line, "%ld %ld %ld", &n, &m, &nnz);
        break;
    }
    std::vector<GveEdge> edges;
    edges.reserve(nnz);
    long a, b;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%ld %ld", &a, &b) == 2) {
            edges.push_back(GveEdge{(uint64_t)(a - 1), (uint64_t)(b - 1), 1.0f});
        }
    }
    fclose(f);
    fprintf(stderr, "numNodes=%ld edges=%zu\n", n, edges.size());

    auto comm = runGveLeiden((uint64_t)n, edges, 1.0, 8);

    FILE* o = fopen(outp, "w");
    fprintf(o, "record_id,community\n");
    for (long i = 0; i < n; ++i) {
        fprintf(o, "%ld,%llu\n", i, (unsigned long long)comm[i]);
    }
    fclose(o);
    fprintf(stderr, "wrote %s (%zu records)\n", outp, comm.size());
    return 0;
}

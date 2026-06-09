// GVE-Leiden bridge translation unit.
//
// This file is the ONLY place that includes the vendored GVE-Leiden headers
// (github.com/puzzlef/leiden-communities-openmp, MIT). Those headers declare
// `using namespace std;` at global scope and define macros (LOG, ASSERT, WRITE,
// PRINT, ...) plus a global `struct None`. Including them anywhere that also
// pulls in the engine's templated headers causes name collisions, so the GVE
// surface is fully contained here and exposed via the GVE-free `gve_leiden.h`.
//
// Compiled with -fopenmp (see CMakeLists). GVE parallelizes the local-move /
// aggregation phases with OpenMP; we cap the thread count so it does not
// oversubscribe alongside the engine's own task scheduler.

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>

// GVE headers (global namespace). Keep this include block first and isolated.
#include "function/gve_leiden.h"
#include "gve/main.hxx" // NOLINT: pulls Graph.hxx, update.hxx, leiden.hxx, etc.

namespace lbug {
namespace algo_extension {

std::vector<uint64_t> runGveLeiden(uint64_t numNodes, const std::vector<GveEdge>& edges,
    double resolution, int maxThreads, bool useCpm, double gamma) {
    using K = uint32_t; // GVE vertex key type
    using V = float;    // edge weight type (matches -DTYPE=float in the verified harness)

    std::vector<uint64_t> result(numNodes, 0);
    if (numNodes == 0) {
        return result;
    }
    // GVE keys are uint32_t; the engine offsets are uint64_t. Guard the cast.
    if (numNodes > static_cast<uint64_t>(std::numeric_limits<K>::max())) {
        throw std::runtime_error(
            "LEIDEN: projected graph has more nodes than the GVE-Leiden backend supports "
            "(uint32 vertex id limit).");
    }

    // Build an undirected graph: every node becomes a vertex so isolated nodes
    // are preserved as their own singleton community, then each undirected edge
    // is added in both directions (explicit symmetrization, matching the
    // verified standalone gate which symmetrizes the match graph).
    DiGraph<K, None, V> x;
    x.respan(static_cast<size_t>(numNodes));
    for (uint64_t u = 0; u < numNodes; ++u) {
        x.addVertex(static_cast<K>(u));
    }
    for (const auto& e : edges) {
        const auto u = static_cast<K>(e.src);
        const auto v = static_cast<K>(e.dst);
        x.addEdge(u, v, e.weight);
        if (u != v) {
            x.addEdge(v, u, e.weight);
        }
    }
    x.update();

    // Constrain GVE's OpenMP usage. leidenStaticOmp reads omp_get_max_threads()
    // internally, so we set it for the duration of the call and restore after.
    const int savedThreads = omp_get_max_threads();
    if (maxThreads > 0) {
        omp_set_num_threads(maxThreads);
    }

    // LeidenOptions(repeat, resolution, tolerance, aggregationTolerance,
    // toleranceDrop, maxIterations, maxPasses, gamma) — single run. The CPM
    // objective ignores `resolution` and uses `gamma`; modularity ignores
    // `gamma`. We pass both and let the compile-time objective pick.
    LeidenOptions opts(1, resolution);
    opts.gamma = gamma;
    const LeidenResult<K> a = useCpm ? leidenStaticOmp<true>(x, opts)
                                     : leidenStaticOmp<false>(x, opts);

    omp_set_num_threads(savedThreads);

    // a.membership is span-sized (== numNodes here, since vertices are 0..numNodes-1).
    const size_t msz = a.membership.size();
    for (uint64_t u = 0; u < numNodes; ++u) {
        result[u] = (u < msz) ? static_cast<uint64_t>(a.membership[u]) : u;
    }
    return result;
}

} // namespace algo_extension
} // namespace lbug

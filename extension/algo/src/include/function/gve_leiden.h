#pragma once

#include <cstdint>
#include <vector>

namespace lbug {
namespace algo_extension {

// Thin, GVE-isolating bridge to the vendored GVE-Leiden implementation
// (github.com/puzzlef/leiden-communities-openmp, MIT). The GVE headers pull in
// `using namespace std;` at global scope and define common macros (LOG, ASSERT,
// WRITE, ...) and a global `struct None`. To keep that out of the engine's
// heavily templated headers, ALL GVE includes live in `gve_leiden.cpp`; this
// header exposes only plain std/scalar types so `leiden.cpp` never sees GVE.

/// A single undirected edge of the projected graph, given by engine node
/// offsets `src`/`dst` and an edge `weight`. The bridge symmetrizes internally,
/// so each logical undirected edge should be supplied exactly once.
struct GveEdge {
    uint64_t src;
    uint64_t dst;
    float weight;
};

/// Run GVE-Leiden over a projected graph and return per-node community ids.
///
/// The returned vector has length `numNodes`; entry `i` is the community id of
/// the node with engine offset `i`. Isolated nodes (no incident edge) come back
/// as their own singleton community, so callers must pass `numNodes` covering
/// every node offset, not just those that appear in `edges`.
///
/// \param numNodes total number of projected-graph nodes (offsets 0..numNodes-1).
///        Must not exceed UINT32_MAX (GVE keys are uint32_t); the caller is
///        expected to have validated this.
/// \param edges undirected edges; self-loops and duplicates are tolerated.
/// \param resolution modularity resolution parameter (>0). 1.0 is the default.
/// \param maxThreads cap on OpenMP threads GVE may use (<=0 means "engine default").
/// \returns community ids, one per node offset, as uint64 (widened from GVE's uint32).
std::vector<uint64_t> runGveLeiden(uint64_t numNodes, const std::vector<GveEdge>& edges,
    double resolution, int maxThreads);

} // namespace algo_extension
} // namespace lbug

#pragma once

#include "common/copy_constructors.h"
#include "common/types/types.h"
#include "function/gds/gds_object_manager.h"

namespace lbug {
namespace algo_extension {

// Edge weight type for the Louvain/Leiden in-memory CSR.
//
// This is `double` (not an integer) so the in-memory graph can hold fractional edge weights, e.g.
// a Splink match probability in [0, 1] passed via `weightProperty`. When no weight property is
// supplied, every edge defaults to `DEFAULT_WEIGHT` (1.0) and the algorithms behave exactly as the
// historical unweighted integer implementation did. See `louvain.cpp` / `leiden.cpp`.
//
// Negative weights are rejected at ingestion (modularity is ill-defined for them); callers should
// pass a non-negative weight such as a match probability or a positive transform of it.
using weight_t = double;
constexpr weight_t DEFAULT_WEIGHT = 1.0;

struct Neighbor {
    common::offset_t neighbor;
    weight_t weight;

    Neighbor(const common::offset_t neighbor, const weight_t weight)
        : neighbor{neighbor}, weight{weight} {}
};

// CSR-like in-memory representation of an undirected weighted graph. Insert nodes in sequence
// by first calling `initNextNode()` and then insert all its neighbors using `insertNbr()`.
// Undirected edges should be explicitly inserted twice.
// Note: modifying the in-memory graph is NOT thread-safe.
struct InMemGraph {
    function::vector_t<common::offset_t> csrOffsets;
    function::vector_t<Neighbor> csrEdges;
    common::offset_t numNodes = 0;
    common::offset_t numEdges = 0;

    InMemGraph(const common::offset_t numNodes, storage::MemoryManager* mm);
    DELETE_BOTH_COPY(InMemGraph);
    ~InMemGraph() = default;

    // Re-initializes to an empty graph. Reuses allocations if `numNodes` <= `this->numNodes`.
    void reinit(const common::offset_t numNodes);

    // Initializes the next node in the sequence to prepare for edges insertions for the node.
    void initNextNode();

    // Inserts a neighbor of the last initialized node.
    void insertNbr(const common::offset_t to, const weight_t weight = DEFAULT_WEIGHT);
};

} // namespace algo_extension
} // namespace lbug

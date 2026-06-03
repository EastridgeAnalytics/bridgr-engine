#include "binder/binder.h"
#include "common/exception/runtime.h"
#include "common/string_utils.h"
#include "common/types/types.h"
#include "function/algo_function.h"
#include "function/config/leiden_config.h"
#include "function/config/louvain_config.h"
#include "function/config/max_iterations_config.h"
#include "function/gds/gds_utils.h"
#include "function/gds/gds_vertex_compute.h"
#include "function/gve_leiden.h"
#include "function/table/bind_input.h"
#include "main/client_context.h"
#include "processor/execution_context.h"
#include "transaction/transaction.h"

#include <cstdint>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace lbug::binder;
using namespace lbug::common;
using namespace lbug::processor;
using namespace lbug::storage;
using namespace lbug::graph;
using namespace lbug::function;

// Leiden community detection: Traag, Waltman & van Eck, Scientific Reports 9, 5233 (2019).
//
// The community-detection core is the vendored GVE-Leiden implementation
// (github.com/puzzlef/leiden-communities-openmp, MIT), reached through the
// GVE-isolating bridge in `function/gve_leiden.h`. The engine's own bespoke
// Leiden over-merged disconnected components on fragmented entity-resolution
// match graphs (pairwise-F1 collapsed to ~0.03 with one giant false community);
// GVE-Leiden is correct on the same data (F1 ~= 0.83, precision ~= 1.0, all
// communities internally connected and pure).
//
// This file keeps the existing `CALL LEIDEN('graph', ...)` signature and output
// schema (node internal id + INT64 `community_id`); only the algorithm body
// changed: it enumerates the projected graph, builds an undirected edge list,
// runs GVE-Leiden, and writes the per-node community id back.

namespace lbug {
namespace algo_extension {
namespace {

struct LeidenOptionalParams final : public MaxIterationOptionalParams {
    OptionalParam<MaxPhases> maxPhases;
    OptionalParam<Resolution> resolution;

    explicit LeidenOptionalParams(const expression_vector& optionalParams);

    LeidenOptionalParams(OptionalParam<MaxIterations> maxIterations,
        OptionalParam<MaxPhases> maxPhases, OptionalParam<Resolution> resolution)
        : MaxIterationOptionalParams{maxIterations}, maxPhases{std::move(maxPhases)},
          resolution{std::move(resolution)} {}

    void evaluateParams(main::ClientContext* context) override {
        MaxIterationOptionalParams::evaluateParams(context);
        maxPhases.evaluateParam(context);
        resolution.evaluateParam(context);
    }

    std::unique_ptr<function::OptionalParams> copy() override {
        return std::make_unique<LeidenOptionalParams>(maxIterations, maxPhases, resolution);
    }
};

LeidenOptionalParams::LeidenOptionalParams(const expression_vector& optionalParams)
    : MaxIterationOptionalParams{constructMaxIterationParam(optionalParams)} {
    for (auto& optionalParam : optionalParams) {
        auto paramName = StringUtils::getLower(optionalParam->getAlias());
        if (paramName == MaxPhases::NAME) {
            maxPhases = function::OptionalParam<MaxPhases>(optionalParam);
        } else if (paramName == Resolution::NAME) {
            resolution = function::OptionalParam<Resolution>(optionalParam);
        } else if (paramName == MaxIterations::NAME) {
            continue;
        } else {
            throw BinderException{"Unknown optional parameter: " + optionalParam->getAlias()};
        }
    }
}

struct LeidenBindData final : public GDSBindData {
    LeidenBindData(expression_vector columns, graph::NativeGraphEntry graphEntry,
        std::shared_ptr<Expression> nodeOutput,
        std::unique_ptr<LeidenOptionalParams> optionalParams)
        : GDSBindData{std::move(columns), std::move(graphEntry), expression_vector{nodeOutput}} {
        this->optionalParams = std::move(optionalParams);
    }

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<LeidenBindData>(*this);
    }
};

// Per-node community assignment, indexed by node offset.
struct FinalResults {
    vector<uint64_t> communities;

    explicit FinalResults(const offset_t numNodes) { communities.resize(numNodes); }
};

// Writes (node internal id, community_id) rows. Unchanged output path: one row
// per node, community id widened to the engine's UINT64 result vector while the
// bound column type stays INT64 (see bindFunc), matching the prior schema.
class WriteResultsVC final : public GDSResultVertexCompute {
public:
    WriteResultsVC(MemoryManager* mm, GDSFuncSharedState* sharedState, FinalResults& finalResults)
        : GDSResultVertexCompute{mm, sharedState}, finalResults{finalResults} {
        nodeIDVector = createVector(LogicalType::INTERNAL_ID());
        communityIDVector = createVector(LogicalType::UINT64());
    }

    void beginOnTableInternal(table_id_t /*tableID*/) override {}

    void vertexCompute(const offset_t startOffset, const offset_t endOffset,
        const table_id_t tableID) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            const auto nodeID = nodeID_t{i, tableID};
            nodeIDVector->setValue<nodeID_t>(0, nodeID);
            communityIDVector->setValue<uint64_t>(0, finalResults.communities[i]);
            localFT->append(vectors);
        }
    }

    unique_ptr<VertexCompute> copy() override {
        return std::make_unique<WriteResultsVC>(mm, sharedState, finalResults);
    }

private:
    FinalResults& finalResults;
    unique_ptr<ValueVector> nodeIDVector;
    unique_ptr<ValueVector> communityIDVector;
};

// Enumerate the projected graph's undirected edges exactly once each.
//
// The engine may store a relationship in a single direction, so a node's full
// neighborhood requires scanning both forward and backward (the same traversal
// the previous implementation used to build its undirected CSR). We canonicalize
// each pair as (min, max) and dedupe, so every undirected edge is emitted once
// regardless of which storage direction it lives in. GVE-Leiden symmetrizes the
// resulting edge list internally.
std::vector<GveEdge> collectUndirectedEdges(const table_id_t tableId, const offset_t numNodes,
    Graph* graph) {
    const auto nbrTables = graph->getRelInfos(tableId);
    const auto nbrInfo = nbrTables[0];
    DASSERT(nbrInfo.srcTableID == nbrInfo.dstTableID);
    const auto scanState = graph->prepareRelScan(*nbrInfo.relGroupEntry, nbrInfo.relTableID,
        nbrInfo.dstTableID, {}, false /*randomLookup*/);

    std::vector<GveEdge> edges;
    // Dedupe canonical undirected pairs. Node offsets fit in 32 bits here (the
    // bridge enforces numNodes <= UINT32_MAX), so pack (min<<32 | max) as the key.
    std::unordered_set<uint64_t> seen;

    const auto addPair = [&](offset_t a, offset_t b) {
        const offset_t lo = a < b ? a : b;
        const offset_t hi = a < b ? b : a;
        const uint64_t key = (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
        if (seen.insert(key).second) {
            edges.push_back(GveEdge{static_cast<uint64_t>(a), static_cast<uint64_t>(b), 1.0f});
        }
    };

    for (auto nodeId = 0u; nodeId < numNodes; ++nodeId) {
        const nodeID_t srcNodeId = {nodeId, tableId};
        for (auto chunk : graph->scanFwd(srcNodeId, *scanState)) {
            chunk.forEach([&](auto neighbors, auto, auto i) { addPair(nodeId, neighbors[i].offset); });
        }
        for (auto chunk : graph->scanBwd(srcNodeId, *scanState)) {
            chunk.forEach([&](auto neighbors, auto, auto i) { addPair(nodeId, neighbors[i].offset); });
        }
    }
    return edges;
}

static common::offset_t tableFunc(const TableFuncInput& input, TableFuncOutput&) {
    const auto clientContext = input.context->clientContext;
    const auto transaction = transaction::Transaction::Get(*clientContext);
    auto sharedState = input.sharedState->ptrCast<GDSFuncSharedState>();
    auto mm = MemoryManager::Get(*clientContext);
    const auto graph = sharedState->graph.get();
    DASSERT(graph->getNodeTableIDs().size() == 1);
    const auto tableID = graph->getNodeTableIDs()[0];
    const auto numNodes = graph->getMaxOffset(transaction, tableID);

    auto leidenBindData = input.bindData->constPtrCast<LeidenBindData>();
    auto& config = leidenBindData->optionalParams->constCast<LeidenOptionalParams>();
    const double resolution = config.resolution.getParamVal();

    // GVE keys are uint32_t; surface a clear error before traversing if too large.
    if (numNodes > static_cast<offset_t>(UINT32_MAX)) {
        throw RuntimeException("LEIDEN: projected graph has more than 2^32 nodes, which the "
                               "GVE-Leiden backend does not support.");
    }

    FinalResults finalResults(numNodes);

    // Build the undirected edge list and run GVE-Leiden. Isolated nodes are not
    // in the edge list but are covered because runGveLeiden receives numNodes and
    // returns a community id for every offset (singletons for isolated nodes).
    const auto edges = collectUndirectedEdges(tableID, numNodes, graph);
    // Cap GVE's OpenMP threads to the engine's configured parallelism so it does
    // not oversubscribe alongside the engine's task scheduler.
    const int maxThreads = static_cast<int>(clientContext->getMaxNumThreadForExec());
    finalResults.communities = runGveLeiden(numNodes, edges, resolution, maxThreads);

    const auto parallelCompute = make_unique<WriteResultsVC>(mm, sharedState, finalResults);
    GDSUtils::runVertexCompute(input.context, GDSDensityState::DENSE, graph, *parallelCompute);

    sharedState->factorizedTablePool.mergeLocalTables();
    return 0;
}

static constexpr char LEIDEN_COMM_COLUMN_NAME[] = "community_id";

static std::unique_ptr<TableFuncBindData> bindFunc(main::ClientContext* context,
    const TableFuncBindInput* input) {
    const auto graphName = input->getLiteralVal<std::string>(0);
    auto graphEntry = GDSFunction::bindGraphEntry(*context, graphName);
    if (graphEntry.nodeInfos.size() != 1) {
        throw RuntimeException("Leiden only supports operations on one node table.");
    }
    if (graphEntry.relInfos.size() != 1) {
        throw RuntimeException("Leiden only supports operations on one edge table.");
    }
    expression_vector columns;
    auto nodeOutput = GDSFunction::bindNodeOutput(*input, graphEntry.getNodeEntries());
    columns.push_back(nodeOutput->constPtrCast<NodeExpression>()->getInternalID());
    columns.push_back(input->binder->createVariable(LEIDEN_COMM_COLUMN_NAME, LogicalType::INT64()));
    return std::make_unique<LeidenBindData>(std::move(columns), std::move(graphEntry), nodeOutput,
        std::make_unique<LeidenOptionalParams>(input->optionalParamsLegacy));
}

} // anonymous namespace

function_set LeidenFunction::getFunctionSet() {
    function_set result;
    auto func = std::make_unique<TableFunction>(name, std::vector{LogicalTypeID::ANY});
    func->bindFunc = bindFunc;
    func->tableFunc = tableFunc;
    func->initSharedStateFunc = GDSFunction::initSharedState;
    func->initLocalStateFunc = TableFunction::initEmptyLocalState;
    func->canParallelFunc = [] { return false; };
    func->getLogicalPlanFunc = GDSFunction::getLogicalPlan;
    func->getPhysicalPlanFunc = GDSFunction::getPhysicalPlan;
    result.push_back(std::move(func));
    return result;
}

} // namespace algo_extension
} // namespace lbug

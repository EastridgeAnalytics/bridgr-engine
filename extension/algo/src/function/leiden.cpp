#include "binder/binder.h"
#include "common/exception/runtime.h"
#include "common/in_mem_gds_utils.h"
#include "common/in_mem_graph.h"
#include "common/string_utils.h"
#include "common/task_system/progress_bar.h"
#include "common/types/types.h"
#include "function/algo_function.h"
#include "function/config/leiden_config.h"
#include "function/config/louvain_config.h"
#include "function/config/max_iterations_config.h"
#include "function/gds/gds_utils.h"
#include "function/gds/gds_vertex_compute.h"
#include "function/table/bind_input.h"
#include "processor/execution_context.h"
#include "transaction/transaction.h"

using namespace std;
using namespace lbug::binder;
using namespace lbug::common;
using namespace lbug::processor;
using namespace lbug::storage;
using namespace lbug::graph;
using namespace lbug::function;

// Leiden community detection: Traag, Waltman & van Eck, Scientific Reports 9, 5233 (2019).
// Extends Louvain with a refinement step that guarantees internally connected communities.
// The parallel move phase follows the same Grappolo-style approach as Louvain.

namespace lbug {
namespace algo_extension {
namespace {

constexpr double THRESHOLD = 1e-6;
constexpr offset_t UNASSIGNED_COMM = numeric_limits<offset_t>::max();

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

struct CommInfo {
    std::atomic<offset_t> size;
    std::atomic<weight_t> degree;

    CommInfo() : size{0}, degree(0) {}
    CommInfo(const CommInfo& other) {
        size.store(other.size.load());
        degree.store(other.degree.load());
    }
    CommInfo& operator=(const CommInfo& other) {
        if (this != &other) {
            size.store(other.size.load());
            degree.store(other.degree.load());
        }
        return *this;
    }
};

struct PhaseState {
    InMemGraph graph;
    AtomicObjectArray<offset_t> acceptedComm;
    AtomicObjectArray<offset_t> currComm;
    AtomicObjectArray<offset_t> nextComm;
    ObjectArray<CommInfo> currCommInfos;
    ObjectArray<CommInfo> nextCommInfos;
    AtomicObjectArray<weight_t> nodeWeightedDegrees;
    AtomicObjectArray<weight_t> selfCommWeights;
    weight_t totalWeight = 0;
    double modularityConstant = 0.0;

    PhaseState(const offset_t numNodes, MemoryManager* mm, ExecutionContext* context)
        : graph{InMemGraph(numNodes, mm)} {
        reinit(numNodes, mm, context);
    }
    DELETE_BOTH_COPY(PhaseState);

    void reinit(offset_t numNodes, MemoryManager* mm, ExecutionContext* context);

    void startNewIter(MemoryManager* mm, ExecutionContext* context);

    void initNextNode(const offset_t nodeId) {
        graph.initNextNode();
        currCommInfos.getUnsafe(nodeId).size.store(1, memory_order_relaxed);
        currCommInfos.getUnsafe(nodeId).degree.store(0, memory_order_relaxed);
        acceptedComm.set(nodeId, nodeId, memory_order_relaxed);
        currComm.set(nodeId, nodeId, memory_order_relaxed);
    }

    void insertNbr(const offset_t from, const offset_t to, const weight_t weight = DEFAULT_WEIGHT) {
        graph.insertNbr(to, weight);
        nodeWeightedDegrees.fetchAdd(from, weight, memory_order_relaxed);
        currCommInfos.getUnsafe(from).degree.fetch_add(weight, memory_order_relaxed);
        totalWeight += weight;
    }

    void finalize() { graph.initNextNode(); }
};

class ResetPhaseStateVC final : public InMemParallelCompute {
public:
    explicit ResetPhaseStateVC(PhaseState& state) : state{state} {}
    ~ResetPhaseStateVC() override = default;

    void parallelCompute(const offset_t startOffset, const offset_t endOffset,
        const std::optional<table_id_t>&) override {
        for (auto nodeId = startOffset; nodeId < endOffset; ++nodeId) {
            state.nodeWeightedDegrees.set(nodeId, 0, memory_order_relaxed);
            state.currCommInfos.set(nodeId, CommInfo());
            state.acceptedComm.set(nodeId, UNASSIGNED_COMM, memory_order_relaxed);
            state.currComm.set(nodeId, UNASSIGNED_COMM, memory_order_relaxed);
            state.nextComm.set(nodeId, UNASSIGNED_COMM, memory_order_relaxed);
        }
    }

    std::unique_ptr<InMemParallelCompute> copy() override {
        return std::make_unique<ResetPhaseStateVC>(state);
    }

private:
    PhaseState& state;
};

class StartNewIterVC final : public InMemParallelCompute {
public:
    explicit StartNewIterVC(PhaseState& state) : state{state} {}
    ~StartNewIterVC() override = default;

    void parallelCompute(const offset_t startOffset, const offset_t endOffset,
        const std::optional<table_id_t>&) override {
        for (auto nodeId = startOffset; nodeId < endOffset; ++nodeId) {
            state.selfCommWeights.set(nodeId, 0, memory_order_relaxed);
            state.nextCommInfos.set(nodeId, CommInfo());
        }
    }

    std::unique_ptr<InMemParallelCompute> copy() override {
        return std::make_unique<StartNewIterVC>(state);
    }

private:
    PhaseState& state;
};

void PhaseState::reinit(const offset_t numNodes, MemoryManager* mm, ExecutionContext* context) {
    totalWeight = 0;
    graph.reinit(numNodes);
    nodeWeightedDegrees.reallocate(numNodes, mm);
    currCommInfos.reallocate(numNodes, mm);
    acceptedComm.reallocate(numNodes, mm);
    currComm.reallocate(numNodes, mm);
    nextComm.reallocate(numNodes, mm);

    ResetPhaseStateVC resetPhaseStateVC(*this);
    InMemGDSUtils::runParallelCompute(resetPhaseStateVC, numNodes, context);
}

void PhaseState::startNewIter(MemoryManager* mm, ExecutionContext* context) {
    selfCommWeights.reallocate(graph.numNodes, mm);
    nextCommInfos.reallocate(graph.numNodes, mm);

    StartNewIterVC startNewIterVC(*this);
    InMemGDSUtils::runParallelCompute(startNewIterVC, graph.numNodes, context);

    modularityConstant = 1.0 / totalWeight;
}

struct FinalResults {
    vector<offset_t> communities;

    explicit FinalResults(const offset_t numNodes) { communities.resize(numNodes); }
};

class SaveCommAssignmentsVC final : public InMemParallelCompute {
public:
    explicit SaveCommAssignmentsVC(const offset_t phaseId, FinalResults& finalResults,
        PhaseState& state)
        : phaseId{phaseId}, finalResults{finalResults}, state{state} {}
    ~SaveCommAssignmentsVC() override = default;

    void parallelCompute(const offset_t startOffset, const offset_t endOffset,
        const std::optional<table_id_t>&) override {
        if (phaseId == 0) {
            for (auto nodeId = startOffset; nodeId < endOffset; ++nodeId) {
                finalResults.communities[nodeId] =
                    state.acceptedComm.get(nodeId, memory_order_relaxed);
            }
        } else {
            for (auto nodeId = startOffset; nodeId < endOffset; ++nodeId) {
                const auto prevCommunity = finalResults.communities[nodeId];
                if (prevCommunity == UNASSIGNED_COMM) {
                    continue;
                }
                const auto newCommunity =
                    state.acceptedComm.get(prevCommunity, memory_order_relaxed);
                finalResults.communities[nodeId] = newCommunity;
            }
        }
    }

    std::unique_ptr<InMemParallelCompute> copy() override {
        return std::make_unique<SaveCommAssignmentsVC>(phaseId, finalResults, state);
    }

private:
    offset_t phaseId;
    FinalResults& finalResults;
    PhaseState& state;
};

class RunIterationVC final : public InMemParallelCompute {
public:
    explicit RunIterationVC(PhaseState& state, double resolution)
        : state{state}, resolution{resolution} {}
    ~RunIterationVC() override = default;

    void parallelCompute(const offset_t startOffset, const offset_t endOffset,
        const std::optional<table_id_t>&) override {
        vector<weight_t> intraCommWeights;
        unordered_map<offset_t, offset_t> commToWeightsIndex;
        for (auto nodeId = startOffset; nodeId < endOffset; ++nodeId) {
            const auto startCSROffset = state.graph.csrOffsets[nodeId];
            const auto endCSROffset = state.graph.csrOffsets[nodeId + 1];
            offset_t targetCommId = UNASSIGNED_COMM;
            if (startCSROffset != endCSROffset) {
                commToWeightsIndex.clear();
                intraCommWeights.clear();
                const weight_t selfLoopWeight = computeIntraCommWeights(nodeId, startCSROffset,
                    endCSROffset, intraCommWeights, commToWeightsIndex);
                targetCommId = findPotentialNewComm(nodeId, selfLoopWeight, intraCommWeights,
                    commToWeightsIndex);
                state.selfCommWeights.set(nodeId, intraCommWeights[0], memory_order_relaxed);
            }
            state.nextComm.set(nodeId, targetCommId, memory_order_relaxed);

            const auto currCommId = state.currComm.get(nodeId, memory_order_relaxed);
            if (targetCommId != currCommId && targetCommId != UNASSIGNED_COMM) {
                const auto nodeDegree = state.nodeWeightedDegrees.get(nodeId, memory_order_relaxed);
                state.nextCommInfos.getUnsafe(targetCommId).degree.fetch_add(nodeDegree);
                state.nextCommInfos.getUnsafe(targetCommId).size.fetch_add(1);
                state.nextCommInfos.getUnsafe(currCommId).degree.fetch_sub(nodeDegree);
                state.nextCommInfos.getUnsafe(currCommId).size.fetch_sub(1);
            }
        }
    }

    weight_t computeIntraCommWeights(const offset_t nodeId, const offset_t startCSROffset,
        const offset_t endCSROffset, vector<weight_t>& intraCommWeights,
        unordered_map<offset_t, offset_t>& commToWeightsIndex) const {
        weight_t selfLoopWeight = 0;
        const auto currComm = state.currComm.get(nodeId, memory_order_relaxed);
        commToWeightsIndex[currComm] = 0;
        intraCommWeights.push_back(0);
        offset_t nextIndex = 1;
        for (auto offset = startCSROffset; offset < endCSROffset; offset++) {
            auto nbrEntry = state.graph.csrEdges[offset];
            if (nbrEntry.neighbor == nodeId) {
                selfLoopWeight += nbrEntry.weight;
            }
            auto nbrCommId = state.currComm.get(nbrEntry.neighbor, memory_order_relaxed);
            if (!commToWeightsIndex.contains(nbrCommId)) {
                commToWeightsIndex[nbrCommId] = nextIndex;
                nextIndex++;
                intraCommWeights.push_back(nbrEntry.weight);
            } else {
                intraCommWeights[commToWeightsIndex[nbrCommId]] += nbrEntry.weight;
            }
        }
        return selfLoopWeight;
    }

    offset_t findPotentialNewComm(const offset_t nodeId, const weight_t selfLoopWeight,
        const vector<weight_t>& intraCommWeights,
        unordered_map<offset_t, offset_t> commToWeightsIndex) const {
        const auto currComm = state.currComm.get(nodeId, memory_order_relaxed);
        const auto degree =
            static_cast<double>(state.nodeWeightedDegrees.get(nodeId, memory_order_relaxed));
        auto newComm = currComm;
        double newCommModGain = 0.0;
        const auto prevIntraCommWeights = static_cast<double>(intraCommWeights[0] - selfLoopWeight);
        const auto prevWeightedDegrees =
            static_cast<double>(
                state.currCommInfos.getUnsafe(currComm).degree.load(memory_order_relaxed)) -
            degree;
        for (auto [nbrCommId, weightIndex] : commToWeightsIndex) {
            if (currComm != nbrCommId) {
                const auto newIntraCommWeights = static_cast<double>(intraCommWeights[weightIndex]);
                const auto newWeightedDegrees = static_cast<double>(
                    state.currCommInfos.getUnsafe(nbrCommId).degree.load(memory_order_relaxed));
                const auto changeIntraWeights = 2 * (newIntraCommWeights - prevIntraCommWeights);
                const auto changeSumWeightedDegrees = 2 * degree * state.modularityConstant *
                                                      (newWeightedDegrees - prevWeightedDegrees);
                const auto modGain =
                    changeIntraWeights - resolution * changeSumWeightedDegrees;
                if (modGain > newCommModGain || ((newCommModGain - modGain) < THRESHOLD &&
                                                    modGain != 0 && (nbrCommId < newComm))) {
                    newCommModGain = modGain;
                    newComm = nbrCommId;
                }
            }
        }
        if (state.currCommInfos.getUnsafe(newComm).size.load(memory_order_relaxed) == 1 &&
            state.currCommInfos.getUnsafe(currComm).size.load(memory_order_relaxed) == 1 &&
            newComm > currComm) {
            newComm = currComm;
        }
        return newComm;
    }

    std::unique_ptr<InMemParallelCompute> copy() override {
        return std::make_unique<RunIterationVC>(state, resolution);
    }

private:
    PhaseState& state;
    double resolution;
};

class ComputeModularityVC final : public InMemParallelCompute {
public:
    ComputeModularityVC(PhaseState& state, std::atomic<weight_t>& sumIntraWeights,
        std::atomic<weight_t>& sumWeightedDegrees)
        : state{state}, sumIntraWeights{sumIntraWeights}, sumWeightedDegrees{sumWeightedDegrees} {}
    ~ComputeModularityVC() override = default;

    void parallelCompute(const offset_t startOffset, const offset_t endOffset,
        const std::optional<table_id_t>&) override {
        weight_t sumIntraLocal = 0;
        weight_t sumTotalLocal = 0;
        for (auto nodeId = startOffset; nodeId < endOffset; ++nodeId) {
            sumIntraLocal += state.selfCommWeights.get(nodeId, memory_order_relaxed);
            const auto degree =
                state.currCommInfos.getUnsafe(nodeId).degree.load(memory_order_relaxed);
            sumTotalLocal += degree * degree;
        }
        sumIntraWeights.fetch_add(sumIntraLocal);
        sumWeightedDegrees.fetch_add(sumTotalLocal);
    }

    std::unique_ptr<InMemParallelCompute> copy() override {
        return std::make_unique<ComputeModularityVC>(state, sumIntraWeights, sumWeightedDegrees);
    }

private:
    PhaseState& state;
    std::atomic<weight_t>& sumIntraWeights;
    std::atomic<weight_t>& sumWeightedDegrees;
};

class UpdateCommInfosVC final : public InMemParallelCompute {
public:
    explicit UpdateCommInfosVC(PhaseState& state) : state{state} {}
    ~UpdateCommInfosVC() override = default;

    void parallelCompute(const offset_t startOffset, const offset_t endOffset,
        const std::optional<table_id_t>&) override {
        for (auto nodeId = startOffset; nodeId < endOffset; ++nodeId) {
            const offset_t size =
                state.nextCommInfos.getUnsafe(nodeId).size.load(memory_order_relaxed);
            const weight_t degree =
                state.nextCommInfos.getUnsafe(nodeId).degree.load(memory_order_relaxed);
            state.currCommInfos.getUnsafe(nodeId).size.fetch_add(size, memory_order_relaxed);
            state.currCommInfos.getUnsafe(nodeId).degree.fetch_add(degree, memory_order_relaxed);
        }
    }

    std::unique_ptr<InMemParallelCompute> copy() override {
        return std::make_unique<UpdateCommInfosVC>(state);
    }

private:
    PhaseState& state;
};

class WriteResultsVC final : public GDSResultVertexCompute {
public:
    WriteResultsVC(MemoryManager* mm, GDSFuncSharedState* sharedState, FinalResults& finalResults)
        : GDSResultVertexCompute{mm, sharedState}, finalResults{finalResults} {
        nodeIDVector = createVector(LogicalType::INTERNAL_ID());
        componentIDVector = createVector(LogicalType::UINT64());
    }

    void beginOnTableInternal(table_id_t /*tableID*/) override {}

    void vertexCompute(const offset_t startOffset, const offset_t endOffset,
        const table_id_t tableID) override {
        for (auto i = startOffset; i < endOffset; ++i) {
            const auto nodeID = nodeID_t{i, tableID};
            nodeIDVector->setValue<nodeID_t>(0, nodeID);
            componentIDVector->setValue<uint64_t>(0, finalResults.communities[i]);
            localFT->append(vectors);
        }
    }

    unique_ptr<VertexCompute> copy() override {
        return std::make_unique<WriteResultsVC>(mm, sharedState, finalResults);
    }

private:
    FinalResults& finalResults;
    unique_ptr<ValueVector> nodeIDVector;
    unique_ptr<ValueVector> componentIDVector;
};

void initInMemoryGraph(const table_id_t tableId, const offset_t numNodes, Graph* graph,
    PhaseState& state) {
    const auto nbrTables = graph->getRelInfos(tableId);
    const auto nbrInfo = nbrTables[0];
    DASSERT(nbrInfo.srcTableID == nbrInfo.dstTableID);
    const auto scanState = graph->prepareRelScan(*nbrInfo.relGroupEntry, nbrInfo.relTableID,
        nbrInfo.dstTableID, {}, false /*randomLookup*/);

    for (auto nodeId = 0u; nodeId < numNodes; ++nodeId) {
        state.initNextNode(nodeId);
        const nodeID_t nextNodeId = {nodeId, tableId};
        for (auto chunk : graph->scanFwd(nextNodeId, *scanState)) {
            chunk.forEach([&](auto neighbors, auto, auto i) {
                auto nbrId = neighbors[i].offset;
                state.insertNbr(nodeId, nbrId);
            });
        }
        for (auto chunk : graph->scanBwd(nextNodeId, *scanState)) {
            chunk.forEach([&](auto neighbors, auto, auto i) {
                auto nbrId = neighbors[i].offset;
                if (nbrId != nodeId) {
                    state.insertNbr(nodeId, nbrId);
                }
            });
        }
    }
    state.finalize();
}

offset_t renumberCommunities(PhaseState& state) {
    unordered_map<offset_t, offset_t> map;
    offset_t nextCommId = 0;
    for (auto nodeId = 0LU; nodeId < state.graph.numNodes; ++nodeId) {
        auto commId = state.acceptedComm.get(nodeId, memory_order_relaxed);
        if (commId == UNASSIGNED_COMM) {
            continue;
        }
        if (!map.contains(commId)) {
            map.insert(make_pair(commId, nextCommId));
            nextCommId++;
        }
        state.acceptedComm.set(nodeId, map.at(commId), memory_order_relaxed);
    }
    return nextCommId;
}

// The Leiden refinement step: ensure every community is internally connected.
// For each community, find connected components via BFS within the community's
// induced subgraph. If a community is disconnected, split it.
void refineCommunities(PhaseState& state) {
    const auto numNodes = state.graph.numNodes;

    unordered_map<offset_t, vector<offset_t>> commNodes;
    for (offset_t i = 0; i < numNodes; i++) {
        auto comm = state.acceptedComm.get(i, memory_order_relaxed);
        if (comm != UNASSIGNED_COMM) {
            commNodes[comm].push_back(i);
        }
    }

    // Generation-based visited tracking avoids resetting between communities.
    vector<offset_t> visitGen(numNodes, 0);
    vector<offset_t> bfsQueue;
    offset_t generation = 1;
    offset_t nextNewComm = numNodes;

    for (auto& [commId, nodes] : commNodes) {
        if (nodes.size() <= 1) {
            generation++;
            continue;
        }

        bool firstComponent = true;
        for (auto start : nodes) {
            if (visitGen[start] == generation) {
                continue;
            }

            bfsQueue.clear();
            bfsQueue.push_back(start);
            visitGen[start] = generation;
            offset_t qHead = 0;

            while (qHead < bfsQueue.size()) {
                auto u = bfsQueue[qHead++];
                const auto csrBegin = state.graph.csrOffsets[u];
                const auto csrEnd = state.graph.csrOffsets[u + 1];
                for (auto off = csrBegin; off < csrEnd; off++) {
                    auto nbr = state.graph.csrEdges[off].neighbor;
                    if (visitGen[nbr] != generation &&
                        state.acceptedComm.get(nbr, memory_order_relaxed) == commId) {
                        visitGen[nbr] = generation;
                        bfsQueue.push_back(nbr);
                    }
                }
            }

            if (firstComponent) {
                firstComponent = false;
            } else {
                for (offset_t j = 0; j < bfsQueue.size(); j++) {
                    state.acceptedComm.set(bfsQueue[j], nextNewComm, memory_order_relaxed);
                }
                nextNewComm++;
            }
        }
        generation++;
    }
}

void aggregateCommunities(const offset_t newCommCount, PhaseState& state, MemoryManager* mm,
    ExecutionContext* context) {
    vector_t<unordered_map<offset_t, weight_t>> commWeights(mm);
    commWeights.resize(newCommCount);
    for (auto nodeId = 0u; nodeId < state.graph.numNodes; nodeId++) {
        const auto beginCSROffset = state.graph.csrOffsets[nodeId];
        const auto endCSROffset = state.graph.csrOffsets[nodeId + 1];
        auto commId = state.acceptedComm.get(nodeId, memory_order_relaxed);
        for (auto offset = beginCSROffset; offset < endCSROffset; ++offset) {
            const auto nbr = state.graph.csrEdges[offset];
            auto nbrCommId = state.acceptedComm.get(nbr.neighbor, memory_order_relaxed);
            if (commId >= nbrCommId) {
                commWeights[commId][nbrCommId] += nbr.weight;
                if (commId != nbrCommId) {
                    commWeights[nbrCommId][commId] += nbr.weight;
                }
            }
        }
    }
    state.reinit(newCommCount, mm, context);
    for (auto nodeId = 0u; nodeId < newCommCount; nodeId++) {
        state.initNextNode(nodeId);
        for (auto [nbrId, weight] : commWeights[nodeId]) {
            state.insertNbr(nodeId, nbrId, weight);
        }
    }
    state.finalize();
}

static common::offset_t tableFunc(const TableFuncInput& input, TableFuncOutput&) {
    const auto clientContext = input.context->clientContext;
    const auto transaction = transaction::Transaction::Get(*clientContext);
    auto sharedState = input.sharedState->ptrCast<GDSFuncSharedState>();
    auto mm = MemoryManager::Get(*clientContext);
    const auto graph = sharedState->graph.get();
    auto maxOffsetMap = graph->getMaxOffsetMap(transaction);
    DASSERT(graph->getNodeTableIDs().size() == 1);
    const auto tableID = graph->getNodeTableIDs()[0];
    const auto origNumNodes = graph->getMaxOffset(transaction, tableID);

    auto leidenBindData = input.bindData->constPtrCast<LeidenBindData>();
    auto& config = leidenBindData->optionalParams->constCast<LeidenOptionalParams>();
    const double resolution = config.resolution.getParamVal();

    auto progressBar = ProgressBar::Get(*clientContext);
    const auto steps = config.maxPhases.getParamVal() * config.maxIterations.getParamVal();

    FinalResults finalResults(origNumNodes);
    PhaseState state(origNumNodes, mm, input.context);

    initInMemoryGraph(tableID, origNumNodes, graph, state);

    for (auto phase = 0u; phase < config.maxPhases.getParamVal(); ++phase) {
        double oldMod = -1;

        for (auto iter = 0u; iter < config.maxIterations.getParamVal(); ++iter) {
            double progress = static_cast<double>((phase + 1) * (iter + 1)) / steps;

            state.startNewIter(mm, input.context);

            RunIterationVC runIteration(state, resolution);
            InMemGDSUtils::runParallelCompute(runIteration, state.graph.numNodes, input.context);

            progressBar->updateProgress(input.context->queryID, progress * 0.5);

            std::atomic<weight_t> sumIntraWeights{0};
            std::atomic<weight_t> sumWeightedDegrees{0};
            ComputeModularityVC newModularityVC(state, sumIntraWeights, sumWeightedDegrees);
            InMemGDSUtils::runParallelCompute(newModularityVC, state.graph.numNodes, input.context);
            const double currMod =
                sumIntraWeights.load() * state.modularityConstant -
                (sumWeightedDegrees.load() * state.modularityConstant * state.modularityConstant);

            if (currMod - oldMod < THRESHOLD) {
                break;
            }

            oldMod = currMod;
            UpdateCommInfosVC updateCommInfosVC(state);
            InMemGDSUtils::runParallelCompute(updateCommInfosVC, state.graph.numNodes,
                input.context);

            std::swap(state.acceptedComm, state.currComm);
            std::swap(state.currComm, state.nextComm);

            progressBar->updateProgress(input.context->queryID, progress);
        }

        // Leiden refinement: split any disconnected communities.
        refineCommunities(state);

        const auto oldCommCount = state.graph.numNodes;
        const auto newCommCount = renumberCommunities(state);

        SaveCommAssignmentsVC setFinalComms(phase, finalResults, state);
        InMemGDSUtils::runParallelCompute(setFinalComms, origNumNodes, input.context);

        if (oldCommCount == newCommCount) {
            break;
        }

        aggregateCommunities(newCommCount, state, mm, input.context);
    }

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

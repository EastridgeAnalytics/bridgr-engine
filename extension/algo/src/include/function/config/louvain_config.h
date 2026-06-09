#pragma once

#include <string>

#include "common/exception/binder.h"
#include "common/types/types.h"
#include "function/gds/gds.h"

namespace lbug {
namespace function {

// The maximum number of phases in which the graph is clustered and then aggregated.
struct MaxPhases {
    static constexpr const char* NAME = "maxphases";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::INT64;
    static constexpr int64_t DEFAULT_VALUE = 20;

    static void validate(int64_t maxPhases) {
        if (maxPhases < 0) {
            throw common::BinderException{"maxphases must be a positive integer."};
        }
    }
};

struct LouvainConfig final : public GDSConfig {
    uint64_t maxIterations = 20;
    uint64_t maxPhases = MaxPhases::DEFAULT_VALUE;
    // Optional named edge property to use as the edge weight. Empty means unweighted
    // (every edge has weight 1.0). The `WeightProperty` config struct is shared with
    // spanning forest (see `spanning_forest_config.h`).
    std::string weightProperty;

    LouvainConfig() = default;
};

} // namespace function
} // namespace lbug

#pragma once

#include <string>

#include "common/exception/binder.h"
#include "common/types/types.h"
#include "function/gds/gds.h"
#include <format>

namespace lbug {
namespace function {

struct Resolution {
    static constexpr const char* NAME = "resolution";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 1.0;

    static void validate(double resolution) {
        if (resolution <= 0) {
            throw common::BinderException{"resolution must be a positive number."};
        }
    }
};

// Community-detection objective. "modularity" (default) is the historical
// behavior; "cpm" (Constant Potts Model) is resolution-limit-free, so it does
// not over-merge weakly-connected communities on fragmented graphs. CPM uses the
// `gamma` parameter instead of `resolution`.
struct Objective {
    static constexpr const char* NAME = "objective";
    static constexpr const char* MODULARITY = "modularity";
    static constexpr const char* CPM = "cpm";
    static constexpr const char* DEFAULT_VALUE = MODULARITY;
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::STRING;

    static void validate(std::string objective) {
        if (objective != MODULARITY && objective != CPM) {
            throw common::BinderException(std::format(
                "objective argument expects '{}' or '{}'. Got: {}", MODULARITY, CPM, objective));
        }
    }
};

// CPM resolution / density threshold (only used when objective := 'cpm'). A
// community is retained only if its internal edge density exceeds gamma, so
// larger gamma yields smaller, denser communities.
struct Gamma {
    static constexpr const char* NAME = "gamma";
    static constexpr common::LogicalTypeID TYPE = common::LogicalTypeID::DOUBLE;
    static constexpr double DEFAULT_VALUE = 1.0;

    static void validate(double gamma) {
        if (gamma <= 0) {
            throw common::BinderException{"gamma must be a positive number."};
        }
    }
};

struct LeidenConfig final : public GDSConfig {
    uint64_t maxIterations = 20;
    uint64_t maxPhases = 20;
    double resolution = Resolution::DEFAULT_VALUE;
    std::string objective = Objective::DEFAULT_VALUE;
    double gamma = Gamma::DEFAULT_VALUE;

    LeidenConfig() = default;
};

} // namespace function
} // namespace lbug

#pragma once

#include <string>

#include "common/exception/binder.h"
#include "common/types/types.h"
#include "function/gds/gds.h"

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

struct LeidenConfig final : public GDSConfig {
    uint64_t maxIterations = 20;
    uint64_t maxPhases = 20;
    double resolution = Resolution::DEFAULT_VALUE;

    LeidenConfig() = default;
};

} // namespace function
} // namespace lbug

//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "ovms_calculator_base.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../logging.hpp"
#include "graph_side_packets.hpp"

namespace ovms {

void OvmsCalculatorBase::SetCalculatorMetricsContract(mediapipe::CalculatorContract* cc) {
    if (cc->InputSidePackets().HasTag(CALCULATOR_METRICS_TAG)) {
        cc->InputSidePackets().Tag(CALCULATOR_METRICS_TAG).Set<CalculatorMetricsMap>();
    }
}

absl::Status OvmsCalculatorBase::InitializeCalculatorMetrics(mediapipe::CalculatorContext* cc) {
    calculatorMetrics.reset();
    if (!cc->InputSidePackets().HasTag(CALCULATOR_METRICS_TAG)) {
        SPDLOG_LOGGER_INFO(mediapipe_logger,
            "Custom calculator metrics are not configured for node '{}'; the CALCULATOR_METRICS side packet is absent.",
            cc->NodeName());
        return absl::OkStatus();
    }

    const auto& scopes = cc->InputSidePackets().Tag(CALCULATOR_METRICS_TAG).Get<CalculatorMetricsMap>();
    const auto scope = scopes.find(cc->NodeName());
    RET_CHECK(scope != scopes.end()) << "Missing metrics scope for node: " << cc->NodeName();
    RET_CHECK(scope->second != nullptr) << "Null metrics scope for node: " << cc->NodeName();
    calculatorMetrics = scope->second;
    return absl::OkStatus();
}

}  // namespace ovms

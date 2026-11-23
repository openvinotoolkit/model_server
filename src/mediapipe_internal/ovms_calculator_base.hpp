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
#pragma once

#include <memory>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_contract.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../metrics/calculator_metrics.hpp"

namespace ovms {

class OvmsCalculatorBase : public mediapipe::CalculatorBase {
protected:
    std::shared_ptr<CalculatorMetrics> calculatorMetrics;

    static void SetCalculatorMetricsContract(mediapipe::CalculatorContract* cc);
    absl::Status InitializeCalculatorMetrics(mediapipe::CalculatorContext* cc);
};

}  // namespace ovms

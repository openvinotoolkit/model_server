//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <stdexcept>

#include <openvino/openvino.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
namespace mediapipe {

class NegativeCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        for (const std::string& tag : cc->Inputs().GetTags()) {
            cc->Inputs().Tag(tag).Set<ov::Tensor>();
        }
        for (const std::string& tag : cc->Outputs().GetTags()) {
            cc->Outputs().Tag(tag).Set<ov::Tensor>();
        }
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        return absl::InvalidArgumentError("Returned error");
    }

    absl::Status Process(CalculatorContext* cc) final {
        return absl::InvalidArgumentError("Returned error");
    }
};
REGISTER_CALCULATOR(NegativeCalculator);
}  // namespace mediapipe

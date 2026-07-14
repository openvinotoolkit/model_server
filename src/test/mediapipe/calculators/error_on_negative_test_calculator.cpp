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
#include <cstring>

#include <openvino/openvino.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop

namespace mediapipe {

class ErrorOnNegativeTestCalculator : public CalculatorBase {
public:
    static absl::Status GetContract(CalculatorContract* cc) {
        cc->Inputs().Index(0).Set<ov::Tensor>();
        cc->Outputs().Index(0).Set<ov::Tensor>();
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }
    absl::Status Close(CalculatorContext* cc) final { return absl::OkStatus(); }
    absl::Status Process(CalculatorContext* cc) final {
        ov::Tensor input = cc->Inputs().Index(0).Get<ov::Tensor>();
        if (static_cast<float*>(input.data())[0] < 0.0f) {
            return absl::InvalidArgumentError("Negative input value");
        }
        ov::Tensor output(input.get_element_type(), input.get_shape());
        std::memcpy(output.data(), input.data(), input.get_byte_size());
        cc->Outputs().Index(0).Add(new ov::Tensor(output), cc->InputTimestamp());
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(ErrorOnNegativeTestCalculator);
}  // namespace mediapipe

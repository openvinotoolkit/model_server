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
#include <openvino/openvino.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop

namespace mediapipe {

/*
    Adds 1 to all bytes in input tensor.
    // TODO: Make it dynamic with multiple inputs/outputs
*/
class DummyCalculator : public CalculatorBase {

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "DummyCalculator::GetContract";
        cc->Inputs().Index(0).Set<ov::Tensor>();
        cc->Outputs().Index(0).Set<ov::Tensor>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "DummyCalculator::Open";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        ov::Tensor input = cc->Inputs().Index(0).Get<ov::Tensor>();
        for (size_t i = 0; i < input.get_byte_size() / 4; i++) {
            ((float*)(input.data()))[i] += 1.0;
        }
        ov::Tensor output(input.get_element_type(), input.get_shape());
        cc->Outputs().Index(0).Add(new ov::Tensor(output), cc->InputTimestamp());

        LOG(INFO) << "DummyCalculator::Process";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(DummyCalculator);
}  // namespace mediapipe

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
#include "src/mediapipe_calculators/streaming_test_calculator_options.pb.h"

namespace mediapipe {

/*
    Adds 1 to all bytes in input tensor.
    Behavior depends on options.kind
*/
class StreamingTestCalculator : public CalculatorBase {
    int cycle_iteration = 0;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "StreamingTestCalculator::GetContract";
        cc->Inputs().Index(0).Set<ov::Tensor>();
        cc->Outputs().Index(0).Set<ov::Tensor>();
        if (cc->Options<StreamingTestCalculatorOptions>().kind() == "cycle") {
            cc->Inputs().Index(1).Set<ov::Tensor>();   // signal
            cc->Outputs().Index(1).Set<ov::Tensor>();  // signal
        }
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingTestCalculator::Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "StreamingTestCalculator::Open";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        if (cc->Options<StreamingTestCalculatorOptions>().kind() == "cycle") {
            LOG(INFO) << "StreamingTestCalculator::Process Cycle";
            if (++cycle_iteration > 3) {
                return absl::OkStatus();
            }
            ov::Tensor input = cc->Inputs().Index(0).IsEmpty() ? cc->Inputs().Index(1).Get<ov::Tensor>() : cc->Inputs().Index(0).Get<ov::Tensor>();
            ov::Tensor output1(input.get_element_type(), input.get_shape());
            ov::Tensor output2(input.get_element_type(), input.get_shape());
            for (size_t i = 0; i < input.get_byte_size() / 4; i++) {
                ((float*)(output1.data()))[i] = ((float*)(input.data()))[i] + 1.0f;
                ((float*)(output2.data()))[i] = ((float*)(input.data()))[i] + 1.0f;
            }
            cc->Outputs().Index(0).Add(new ov::Tensor(output1), Timestamp(cycle_iteration));  // TODO: ?
            cc->Outputs().Index(1).Add(new ov::Tensor(output2), Timestamp(cycle_iteration));  // TODO: ?
        } else {
            LOG(INFO) << "StreamingTestCalculator::Process Default";
            ov::Tensor input = cc->Inputs().Index(0).Get<ov::Tensor>();
            ov::Tensor output(input.get_element_type(), input.get_shape());
            for (size_t i = 0; i < input.get_byte_size() / 4; i++) {
                ((float*)(output.data()))[i] = ((float*)(input.data()))[i] + 1.0f;
            }
            cc->Outputs().Index(0).Add(new ov::Tensor(output), cc->InputTimestamp());
        }
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(StreamingTestCalculator);
}  // namespace mediapipe

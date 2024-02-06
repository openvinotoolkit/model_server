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
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <openvino/openvino.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#include "src/test/mediapipe/calculators/ovmscalculator.pb.h"
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
namespace mediapipe {
using std::endl;

class ExceptionDuringCloseCalculator : public CalculatorBase {
    std::unordered_map<std::string, std::string> outputNameToTag;

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
        LOG(INFO) << "Throwing exception from ExceptionDuringCloseCalculator";
        throw std::runtime_error("Throwing exception from ExceptionDuringCloseCalculator");
        return absl::OkStatus();
    }
    absl::Status Open(CalculatorContext* cc) final {
        for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId(); ++id) {
            if (!cc->Inputs().Get(id).Header().IsEmpty()) {
                cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
            }
        }
        if (cc->OutputSidePackets().NumEntries() != 0) {
            for (CollectionItemId id = cc->InputSidePackets().BeginId(); id < cc->InputSidePackets().EndId(); ++id) {
                cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
            }
        }
        cc->SetOffset(TimestampDiff(0));
        const auto& options = cc->Options<OVMSCalculatorOptions>();
        for (const auto& [key, value] : options.tag_to_output_tensor_names()) {
            outputNameToTag[value] = key;
        }
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        ov::Shape ovShape{1, 10};
        ov::Tensor* output = new ov::Tensor(ov::element::Type_t::f32, ovShape);
        cc->Outputs().Tag(outputNameToTag.at("a")).Add(output, cc->InputTimestamp());
        return absl::OkStatus();
    }
};
#pragma GCC diagnostic pop
REGISTER_CALCULATOR(ExceptionDuringCloseCalculator);
}  // namespace mediapipe

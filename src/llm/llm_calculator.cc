//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop

#include <memory>

#include <continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>

constexpr size_t BATCH_SIZE = 1;

namespace mediapipe {

class LLMCalculator : public CalculatorBase {
    ov::Core core;
    ov::InferRequest tokenizer, detokenizer, llm;
    std::shared_ptr<ContinuousBatchingPipeline> cbPipe = nullptr;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open start";
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process start";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(LLMCalculator);
}  // namespace mediapipe

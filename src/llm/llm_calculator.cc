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

#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "src/kfserving_api/grpc_predict_v2.pb.h"

#include <openvino/openvino.hpp>
#include <continuous_batching_pipeline.hpp>
#include <scheduler_config.hpp>
#include <tokenizer.hpp>
#include <generation_config.hpp>

using KFSRequest = inference::ModelInferRequest;
using KFSResponse = inference::ModelInferResponse;

constexpr size_t BATCH_SIZE = 1;

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string prompt) {
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t> tokens) {
    constexpr size_t BATCH_SIZE = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

std::string detokenize(ov::InferRequest& detokenizer, ov::Tensor tokens) {
    detokenizer.set_input_tensor(tokens);
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}


namespace mediapipe {

class LLMCalculator : public CalculatorBase {
    ov::Core core;
    ov::InferRequest tokenizer, detokenizer, llm;

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

#define RETURN_EXECUTION_FAILED_STATUS() \
    return absl::Status(absl::StatusCode::kInternal, "Error occurred during graph execution")

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process start";
       
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(LLMCalculator);
}  // namespace mediapipe

//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include "openvino/op/constant.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/multiply.hpp"
#include "../ovinferrequestsqueue.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "src/embeddings/embeddings_calculator_ov.pb.h"

#include <openvino/genai/tokenizer.hpp>

namespace ovms {

class EmbeddingsModel {
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiledModel;
    std::unique_ptr<OVInferRequestsQueue> inferRequestsQueue;

public:
    void prepareInferenceRequestsQueue(const uint32_t& numberOfParallelInferRequests);
    OVInferRequestsQueue& getInferRequestsQueue() {
        return *inferRequestsQueue;
    }
    EmbeddingsModel(const std::filesystem::path& model_dir,
        const std::string& target_device,
        const ov::AnyMap& properties);
};

class EmbeddingsServable {
    std::shared_ptr<ov::genai::Tokenizer> tokenizer;
    std::shared_ptr<EmbeddingsModel> embeddings;
    int64_t pad_token;
    std::optional<uint32_t> maxModelLength;

public:
    EmbeddingsServable(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string graphPath);
    OVInferRequestsQueue& getEmbeddingsInferRequestsQueue() {
        return embeddings->getInferRequestsQueue();
    }
    ov::genai::Tokenizer& getTokenizer() {
        return *tokenizer;
    }
    int64_t& getPadToken() {
        return pad_token;
    }
    std::optional<uint32_t>& getMaxModelLength() {
        return maxModelLength;
    }
};

using EmbeddingsServableMap = std::unordered_map<std::string, std::shared_ptr<EmbeddingsServable>>;
}  // namespace ovms

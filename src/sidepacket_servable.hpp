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
#include "ovinferrequestsqueue.hpp"

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

#include <openvino/genai/tokenizer.hpp>

namespace ovms {

struct SidepacketServable {
    std::shared_ptr<ov::genai::Tokenizer> tokenizer;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiledModel;
    std::unique_ptr<OVInferRequestsQueue> inferRequestsQueue;
    std::optional<int64_t> pad_token;
    std::optional<int64_t> eos_token;
    std::optional<int64_t> bos_token;
    std::optional<int64_t> sep_token;
    std::optional<uint32_t> maxModelLength;
    std::filesystem::path parsedModelsPath;

public:
    SidepacketServable(const std::string& modelDir, const std::string& targetDevice, const std::string& pluginConfig, const std::string& graphPath);
    OVInferRequestsQueue& getInferRequestsQueue() {
        return *inferRequestsQueue;
    }
    ov::genai::Tokenizer& getTokenizer() {
        return *tokenizer;
    }
    const std::optional<int64_t> getPadToken() {
        return pad_token;
    }
    const std::optional<int64_t> getEosToken() {
        return eos_token;
    }
    const std::optional<int64_t> getBosToken() {
        return bos_token;
    }
    const std::optional<int64_t> getSepToken() {
        return sep_token;
    }
    const std::optional<uint32_t> getMaxModelLength() {
        return maxModelLength;
    }
    const size_t getNumberOfModelInputs() {
        return compiledModel.inputs().size();
    }
};

using RerankServableMap = std::unordered_map<std::string, std::shared_ptr<SidepacketServable>>;
using EmbeddingsServableMap = std::unordered_map<std::string, std::shared_ptr<SidepacketServable>>;
}  // namespace ovms

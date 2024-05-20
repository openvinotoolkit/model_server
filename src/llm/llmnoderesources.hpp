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
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#include <scheduler_config.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop

namespace ovms {
class Status;
class LLMExecutor;

struct LLMNodeResources {
public:
    LLMNodeResources(const LLMNodeResources&) = delete;
    LLMNodeResources& operator=(LLMNodeResources&) = delete;

    LLMNodeResources();
    static Status createLLMNodeResources(std::shared_ptr<LLMNodeResources>& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNode, std::string graphPath);

    std::shared_ptr<ContinuousBatchingPipeline> cbPipe = nullptr;
    std::string workspacePath;

    void initiateGeneration();

private:
    // LLM Executor launches generation loop thread upon constrution and stops it when destroyed.
    std::unique_ptr<LLMExecutor> llmExecutor;
    static std::unordered_map<std::string, std::string> prepareLLMNodeInitializeArguments(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string basePath);
};
using LLMNodeResourcesMap = std::unordered_map<std::string, std::shared_ptr<LLMNodeResources>>;
}  // namespace ovms

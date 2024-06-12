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

#include <map>
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

#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>

#include "src/python/utils.hpp"

namespace ovms {
class Status;
class LLMExecutorWrapper;

using plugin_config_t = std::map<std::string, ov::Any>;

struct TextProcessor {
    std::string bosToken = "";
    std::string eosToken = "";
    std::unique_ptr<PyObjectWrapper<py::object>> chatTemplate = nullptr;
};

struct LLMNodeResources {
public:
    std::shared_ptr<ContinuousBatchingPipeline> cbPipe = nullptr;
    std::string modelsPath;
    std::string device;
    plugin_config_t pluginConfig;
    SchedulerConfig schedulerConfig;
    TextProcessor textProcessor;

    static Status createLLMNodeResources(std::shared_ptr<LLMNodeResources>& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNode, std::string graphPath);

    LLMNodeResources(const LLMNodeResources&) = delete;
    LLMNodeResources& operator=(LLMNodeResources&) = delete;
    LLMNodeResources() = default;

    void initiateGeneration();

    void notifyExecutorThread();

private:
    std::unique_ptr<LLMExecutorWrapper> llmExecutorWrapper;
    static std::unordered_map<std::string, std::string> prepareLLMNodeInitializeArguments(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string basePath);
};
using LLMNodeResourcesMap = std::unordered_map<std::string, std::shared_ptr<LLMNodeResources>>;
}  // namespace ovms

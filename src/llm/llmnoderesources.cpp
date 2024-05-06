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
#include "llmnoderesources.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "../logging.hpp"
#include "../status.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop

#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "src/python/llm_calculator.pb.h"

namespace ovms {

LLMNodeResources::LLMNodeResources() {
}

std::unordered_map<std::string, std::string> LLMNodeResources::prepareLLMNodeInitializeArguments(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string basePath) {
    std::unordered_map<std::string, std::string> LLMArguments;
    return LLMArguments;
}

Status LLMNodeResources::createLLMNodeResources(std::shared_ptr<LLMNodeResources>& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string graphPath) {
    mediapipe::LLMCalculatorOptions nodeOptions;
    graphNodeConfig.node_options(0).UnpackTo(&nodeOptions);

    nodeResources = std::make_shared<PythonNodeResources>(pythonBackend);
    return StatusCode::OK;
}

}  // namespace ovms

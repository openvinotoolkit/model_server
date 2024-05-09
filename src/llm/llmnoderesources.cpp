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
#include "llmnoderesources.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#include "../logging.hpp"
#include "../status.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop

#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "src/llm/llm_calculator.pb.h"

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
    nodeResources = std::make_shared<LLMNodeResources>();
    auto fsWorkspacePath = std::filesystem::path(nodeOptions.workspace_path());

    std::string basePath;
    if (fsWorkspacePath.is_relative()) {
        basePath = (std::filesystem::path(graphPath) / fsWorkspacePath).string();
    } else {
        basePath = fsWorkspacePath.string();
    }

    nodeResources->workspacePath = basePath;
    if (basePath.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node workspace_path: {} is empty. ", basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }
    if (!std::filesystem::exists(basePath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node workspace_path: {} does not exist. ", basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }
    if (!std::filesystem::is_directory(basePath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node workspace_path: {} is not a directory. ", basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }

    size_t NUM_BLOCKS = 364;
    // Currently harrdcoded, will parametrize in future
    SchedulerConfig default_config{
        .max_num_batched_tokens = 256,
        .num_kv_blocks = NUM_BLOCKS,
        .dynamic_split_fuse = false,
        .max_num_seqs = 256,
        .max_paddings = 256,
    };

    try {
        nodeResources->cbPipe = std::make_unique<ContinuousBatchingPipeline>(basePath, default_config);
    } catch (...) {
        SPDLOG_ERROR("Error during llm node initialization for workspace_path: {}", basePath);
        return StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED;
    }

    return StatusCode::OK;
}

}  // namespace ovms

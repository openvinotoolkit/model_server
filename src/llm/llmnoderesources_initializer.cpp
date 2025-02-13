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
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <spdlog/spdlog.h>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../logging.hpp"
#include "../status.hpp"
#include "llmnoderesources_initializer.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "continuous_batching_pipeline/node_resources_initializer.hpp"

namespace ovms {

Status LLMNodeResourcesInitializer::parseModelsPath(std::string modelsPath, std::string graphPath) {
    auto fsModelsPath = std::filesystem::path(modelsPath);
    if (fsModelsPath.is_relative()) {
        this->basePath = (std::filesystem::path(graphPath) / fsModelsPath).string();
    } else {
        this->basePath = fsModelsPath.string();
    }

    if (this->basePath.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} is empty. ", this->basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }
    if (!std::filesystem::exists(this->basePath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} does not exist. ", this->basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }
    if (!std::filesystem::is_directory(this->basePath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node models_path: {} is not a directory. ", this->basePath);
        return StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST;
    }

    return StatusCode::OK;
}

Status initializeLLMNodeResources(std::shared_ptr<LLMNodeResources>& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string graphPath) {
    mediapipe::LLMCalculatorOptions nodeOptions;
    graphNodeConfig.node_options(0).UnpackTo(&nodeOptions);
    Status status;
    if (nodeOptions.has_models_path()) { // Legacy initialization
        SPDLOG_LOGGER_WARN(modelmanager_logger, "LLM node options contain models_path field. " 
            "This is a legacy initialization method that will not be available in the future. "
            "Consider using pipeline_config field instead.");
        ContinuousBatchingNodeResourcesInitializer cbNodeResourcesInitializer;
        status = cbNodeResourcesInitializer.initializeLegacy(nodeResources, nodeOptions, graphPath);
        if (status != StatusCode::OK) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during LLM node resources initialization: {}", status.string());
            return status;
        }
    } else {
        if(nodeOptions.has_continuous_batching_pipeline_config()) {
            ContinuousBatchingNodeResourcesInitializer cbNodeResourcesInitializer;
            status = cbNodeResourcesInitializer.initialize(nodeResources, nodeOptions, graphPath);
        } /* else if (nodeOptions.has_visual_language_model_pipeline_config()) {
            nodeResources = std::make_unique<VisualLanguageModelNodeResources>(nodeOptions);
        } else if (nodeOptions.has_llm_pipeline_config()) {
            nodeResources = std::make_unique<???NodeResources>(nodeOptions);
        } */ else {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node options do not contain any recognized pipeline configuration.");
            return StatusCode::INTERNAL_ERROR;
        }

        if (status != StatusCode::OK) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error during LLM node resources initialization: {}", status.string());
            return status;
        }
    }
    
    return nodeResources->initialize();
}

}  // namespace ovms

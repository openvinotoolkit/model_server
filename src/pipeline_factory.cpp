//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "pipeline_factory.hpp"

#include "prediction_service_utils.hpp"

namespace ovms {

Status PipelineFactory::createDefinition(const std::string& pipelineName,
    const std::vector<NodeInfo>& nodeInfos,
    const pipeline_connections_t& connections,
    ModelManager& manager) {
    if (definitionExists(pipelineName)) {
        SPDLOG_WARN("Two pipelines with the same name: {} defined in config file. Ignoring the second definition", pipelineName);
        return StatusCode::PIPELINE_DEFINITION_ALREADY_EXIST;
    }
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>(pipelineName, nodeInfos, connections);

    pipelineDefinition->makeSubscriptions(manager);
    Status validationResult = pipelineDefinition->validate(manager);
    if (!validationResult.ok()) {
        SPDLOG_ERROR("Loading pipeline definition: {} failed: {}", pipelineName, validationResult.string());
        return validationResult;
    }

    std::unique_lock lock(definitionsMtx);
    definitions[pipelineName] = std::move(pipelineDefinition);

    SPDLOG_INFO("Loading pipeline definition: {} succeeded", pipelineName);
    return StatusCode::OK;
}

Status PipelineFactory::create(std::unique_ptr<Pipeline>& pipeline,
    const std::string& name,
    const tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response,
    ModelManager& manager) const {
    if (!definitionExists(name)) {
        SPDLOG_INFO("Pipeline with requested name: {} does not exist", name);
        return StatusCode::PIPELINE_DEFINITION_NAME_MISSING;
    }
    std::shared_lock lock(definitionsMtx);
    auto& definition = *definitions.at(name);
    lock.unlock();
    return definition.create(pipeline, request, response, manager);
}
}  // namespace ovms

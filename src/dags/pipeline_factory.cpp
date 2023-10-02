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

#include "../capi_frontend/inferencerequest.hpp"
#include "../capi_frontend/inferenceresponse.hpp"
#include "../logging.hpp"
#include "../model_metric_reporter.hpp"
#include "../modelmanager.hpp"
#include "../prediction_service_utils.hpp"
#include "../status.hpp"
#include "nodeinfo.hpp"
#include "pipeline.hpp"
#include "pipelinedefinition.hpp"

namespace ovms {

bool PipelineFactory::definitionExists(const std::string& name) const {
    std::shared_lock lock(definitionsMtx);
    return definitions.find(name) != definitions.end();
}

PipelineDefinition* PipelineFactory::findDefinitionByName(const std::string& name) const {
    std::shared_lock lock(definitionsMtx);
    auto it = definitions.find(name);
    if (it == std::end(definitions)) {
        return nullptr;
    } else {
        return it->second.get();
    }
}

void PipelineFactory::retireOtherThan(std::set<std::string>&& pipelinesInConfigFile, ModelManager& manager) {
    std::for_each(definitions.begin(),
        definitions.end(),
        [&pipelinesInConfigFile, &manager](auto& nameDefinitionPair) {
            if (pipelinesInConfigFile.find(nameDefinitionPair.second->getName()) == pipelinesInConfigFile.end() && nameDefinitionPair.second->getStateCode() != PipelineDefinitionStateCode::RETIRED) {
                nameDefinitionPair.second->retire(manager);
            }
        });
}

Status PipelineFactory::createDefinition(const std::string& pipelineName,
    const std::vector<NodeInfo>& nodeInfos,
    const pipeline_connections_t& connections,
    ModelManager& manager) {
    if (definitionExists(pipelineName)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "pipeline definition: {} is already created", pipelineName);
        return StatusCode::PIPELINE_DEFINITION_ALREADY_EXIST;
    }
    std::unique_ptr<PipelineDefinition> pipelineDefinition = std::make_unique<PipelineDefinition>(pipelineName, nodeInfos, connections, manager.getMetricRegistry(), &manager.getMetricConfig());

    pipelineDefinition->makeSubscriptions(manager);
    Status validationResult = pipelineDefinition->validate(manager);
    if (!validationResult.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Validation of pipeline definition: {} failed: {}", pipelineName, validationResult.string());
        if (validationResult == StatusCode::PIPELINE_NAME_OCCUPIED) {
            pipelineDefinition->resetSubscriptions(manager);
            return validationResult;
        }
    } else {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Loading pipeline definition: {} succeeded", pipelineName);
    }

    std::unique_lock lock(definitionsMtx);
    definitions[pipelineName] = std::move(pipelineDefinition);

    return validationResult;
}

Status PipelineFactory::reloadDefinition(const std::string& pipelineName,
    const std::vector<NodeInfo>&& nodeInfos,
    const pipeline_connections_t&& connections,
    ModelManager& manager) {
    auto pd = findDefinitionByName(pipelineName);
    if (pd == nullptr) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Requested to reload pipeline definition but it does not exist: {}", pipelineName);
        return StatusCode::UNKNOWN_ERROR;
    }
    return pd->reload(manager, std::move(nodeInfos), std::move(connections));
}

Status PipelineFactory::revalidatePipelines(ModelManager& manager) {
    Status firstErrorStatus = StatusCode::OK;
    for (auto& [name, definition] : definitions) {
        if (definition->getStatus().isRevalidationRequired()) {
            auto validationResult = definition->validate(manager);
            if (!validationResult.ok()) {
                if (firstErrorStatus.ok()) {
                    firstErrorStatus = validationResult;
                }
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Revalidation pipeline definition: {} failed: {}", name, validationResult.string());
            } else {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Revalidation of pipeline: {} succeeded", name);
            }
        }
    }
    return firstErrorStatus;
}

const std::vector<std::string> PipelineFactory::getPipelinesNames() const {
    std::vector<std::string> names;
    std::shared_lock lock(definitionsMtx);
    names.reserve(definitions.size());
    for (auto& [name, definition] : definitions) {
        names.push_back(definition->getName());
    }
    return names;
}

template <typename RequestType, typename ResponseType>
Status PipelineFactory::create(std::unique_ptr<Pipeline>& pipeline, const std::string& name, const RequestType* request, ResponseType* response, ModelManager& manager) const {
    std::shared_lock lock(definitionsMtx);
    if (!definitionExists(name)) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Pipeline with requested name: {} does not exist", name);
        return StatusCode::PIPELINE_DEFINITION_NAME_MISSING;
    }
    auto& definition = *definitions.at(name);
    lock.unlock();
    return definition.create(pipeline, request, response, manager);
}

template Status PipelineFactory::create<::KFSRequest, ::KFSResponse>(std::unique_ptr<Pipeline>& pipeline, const std::string& name, const ::KFSRequest* request, ::KFSResponse* response, ModelManager& manager) const;
template Status PipelineFactory::create<tensorflow::serving::PredictRequest, tensorflow::serving::PredictResponse>(std::unique_ptr<Pipeline>& pipeline, const std::string& name, const tensorflow::serving::PredictRequest* request, tensorflow::serving::PredictResponse* response, ModelManager& manager) const;
template Status PipelineFactory::create<InferenceRequest, InferenceResponse>(std::unique_ptr<Pipeline>& pipeline, const std::string& name, const InferenceRequest* request, InferenceResponse* response, ModelManager& manager) const;
}  // namespace ovms

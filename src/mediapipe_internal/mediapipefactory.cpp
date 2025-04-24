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
#include "mediapipefactory.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6326 6308 6387 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../logging.hpp"
#include "../modelmanager.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386 6326 6246)
#include "mediapipe/framework/deps/registration.h"
#pragma warning(pop)
#include "mediapipegraphdefinition.hpp"

namespace ovms {

static void logRegisteredNames(std::unordered_set<std::string> registrySet, std::string registryName) {
    std::vector<std::string> names(registrySet.begin(), registrySet.end());
    std::sort(names.begin(), names.end());
    auto result = joins(names, ", ");
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Registered {}: {}\n", registryName, result);
}

MediapipeFactory::MediapipeFactory(PythonBackend* pythonBackend) {
    this->pythonBackend = pythonBackend;
    logRegisteredNames(mediapipe::CalculatorBaseRegistry::GetRegisteredNames(), "Calculators");
    logRegisteredNames(mediapipe::SubgraphRegistry::GetRegisteredNames(), "Subgraphs");
    logRegisteredNames(mediapipe::InputStreamHandlerRegistry::GetRegisteredNames(), "InputStreamHandlers");
    logRegisteredNames(mediapipe::OutputStreamHandlerRegistry::GetRegisteredNames(), "OutputStreamHandlers");
}

Status MediapipeFactory::createDefinition(const std::string& pipelineName,
    const MediapipeGraphConfig& config,
    ModelManager& manager) {
    if (definitionExists(pipelineName)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Mediapipe graph definition: {} is already created", pipelineName);
        return StatusCode::PIPELINE_DEFINITION_ALREADY_EXIST;
    }
    std::shared_ptr<MediapipeGraphDefinition> graphDefinition = std::make_shared<MediapipeGraphDefinition>(pipelineName, config, manager.getMetricRegistry(), &manager.getMetricConfig(), pythonBackend);
    auto stat = graphDefinition->validate(manager);
                SPDLOG_ERROR("ER");
    if (stat.getCode() == StatusCode::MEDIAPIPE_GRAPH_NAME_OCCUPIED) {
        return stat;
    }
                SPDLOG_ERROR("ER");
    std::unique_lock lock(definitionsMtx);
                SPDLOG_ERROR("ER");
    definitions.insert({pipelineName, std::move(graphDefinition)});
                SPDLOG_ERROR("ER");
    return stat;
}

bool MediapipeFactory::definitionExists(const std::string& name) const {
    std::shared_lock lock(definitionsMtx);
    return this->definitions.find(name) != this->definitions.end();
}

MediapipeGraphDefinition* MediapipeFactory::findDefinitionByName(const std::string& name) const {
    std::shared_lock lock(definitionsMtx);
    auto it = definitions.find(name);
    if (it == std::end(definitions)) {
        return nullptr;
    } else {
        return it->second.get();
    }
}

Status MediapipeFactory::reloadDefinition(const std::string& name,
    const MediapipeGraphConfig& config,
    ModelManager& manager) {
    auto mgd = findDefinitionByName(name);
    if (mgd == nullptr) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Requested to reload mediapipe graph definition but it does not exist: {}", name);
        return StatusCode::INTERNAL_ERROR;
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Reloading mediapipe graph: {}", name);
    return mgd->reload(manager, config);
}

Status MediapipeFactory::create(std::shared_ptr<MediapipeGraphExecutor>& pipeline,
    const std::string& name,
    ModelManager& manager) const {
    std::shared_lock lock(definitionsMtx);
    if (!definitionExists(name)) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Mediapipe with requested name: {} does not exist", name);
        return StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING;
    }
    auto& definition = *definitions.at(name);
    lock.unlock();
    return definition.create(pipeline);
}

void MediapipeFactory::retireOtherThan(std::set<std::string>&& graphsInConfigFile, ModelManager& manager) {
    std::for_each(definitions.begin(),
        definitions.end(),
        [&graphsInConfigFile, &manager](auto& nameDefinitionPair) {
            if (graphsInConfigFile.find(nameDefinitionPair.second->getName()) == graphsInConfigFile.end() && nameDefinitionPair.second->getStateCode() != PipelineDefinitionStateCode::RETIRED) {
                nameDefinitionPair.second->retire(manager);
            }
        });
}

Status MediapipeFactory::revalidatePipelines(ModelManager&) {
    SPDLOG_LOGGER_WARN(modelmanager_logger, "revalidation of mediapipe graphs not implemented yet");
    return StatusCode::OK;
}

const std::vector<std::string> MediapipeFactory::getMediapipePipelinesNames() const {
    std::vector<std::string> names;
    std::shared_lock lock(definitionsMtx);
    names.reserve(definitions.size());
    for (auto& [name, definition] : definitions) {
        names.push_back(definition->getName());
    }
    return names;
}

MediapipeFactory::~MediapipeFactory() = default;
}  // namespace ovms

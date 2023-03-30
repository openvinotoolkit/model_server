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

#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../modelmanager.hpp"
#include "../status.hpp"
#include "mediapipegraphdefinition.hpp"

namespace ovms {

Status MediapipeFactory::createDefinition(const std::string& pipelineName,
    const MediapipeGraphConfig& config,
    ModelManager& manager) {
    if (definitionExists(pipelineName)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Mediapipe graph definition: {} is already created", pipelineName);
        return StatusCode::PIPELINE_DEFINITION_ALREADY_EXIST;
    }
    std::shared_ptr<MediapipeGraphExecutor> graphDefinition = std::make_shared<MediapipeGraphExecutor>(pipelineName, config, manager.getMetricRegistry(), &manager.getMetricConfig());
    auto stat = graphDefinition->validate(manager);
    auto it = definitions.insert({pipelineName, std::move(graphDefinition)});
    return StatusCode::OK;
}

bool MediapipeFactory::definitionExists(const std::string& name) const {
    // TODO thread safety
    return this->definitions.find(name) != this->definitions.end();
}

MediapipeGraphExecutor* MediapipeFactory::findDefinitionByName(const std::string& name) const {
    return nullptr;  // TODO
}
Status MediapipeFactory::reloadDefinition(const std::string& pipelineName,  // TODO
    const MediapipeGraphConfig& config,
    ModelManager& manager) {
    return StatusCode::OK;
}

Status MediapipeFactory::create(std::shared_ptr<MediapipeGraphExecutor>& pipeline,
    const std::string& name,
    const KFSRequest* request,
    KFSResponse* response,
    ModelManager& manager) const {
    auto it = definitions.find(name);
    if (it == definitions.end()) {
        // TODO log
        return StatusCode::NOT_IMPLEMENTED;
    }
    pipeline = it->second;
    return StatusCode::OK;
}

void MediapipeFactory::retireOtherThan(std::set<std::string>&& pipelinesInConfigFile, ModelManager& manager) {}  // TODO
Status MediapipeFactory::revalidatePipelines(ModelManager&) {                                                    // TODO
    return StatusCode::OK;
}
// const std::vector<std::string> getPipelinesNames() const;
MediapipeFactory::~MediapipeFactory() = default;
}  // namespace ovms

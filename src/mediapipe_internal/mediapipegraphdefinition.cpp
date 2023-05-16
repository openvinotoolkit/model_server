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
#include "mediapipegraphdefinition.hpp"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../deserialization.hpp"
#include "../execution_context.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../metric.hpp"
#include "../modelmanager.hpp"
#include "../serialization.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../tensorinfo.hpp"
#include "../timer.hpp"
#include "../version.hpp"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipegraphexecutor.hpp"

namespace ovms {
MediapipeGraphConfig MediapipeGraphDefinition::MGC;

const tensor_map_t MediapipeGraphDefinition::getInputsInfo() const {
    return this->inputsInfo;
}

const tensor_map_t MediapipeGraphDefinition::getOutputsInfo() const {
    return this->outputsInfo;
}

Status MediapipeGraphDefinition::validateForConfigFileExistence() {
    std::ifstream ifs(this->mgconfig.getGraphPath());
    if (!ifs.is_open()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to open mediapipe graph definition: {}, file: {}\n", this->getName(), this->mgconfig.getGraphPath());
        return StatusCode::FILE_INVALID;
    }
    this->chosenConfig.clear();
    ifs.seekg(0, std::ios::end);
    this->chosenConfig.reserve(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    this->chosenConfig.assign((std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::validateForConfigLoadableness() {
    bool success = ::google::protobuf::TextFormat::ParseFromString(chosenConfig, &this->config);
    if (!success) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Trying to parse mediapipe graph definition: {} failed", this->getName(), this->chosenConfig);
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::validate(ModelManager& manager) {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Started validation of mediapipe: {}", getName());
    ValidationResultNotifier notifier(this->status, this->loadedNotify);
    Status validationResult = validateForConfigFileExistence();
    if (!validationResult.ok()) {
        return validationResult;
    }
    validationResult = validateForConfigLoadableness();
    if (!validationResult.ok()) {
        return validationResult;
    }
    // TODO
    // 3 validate 1<= outputs
    // 4 validate 1<= inputs
    // 5 validate no side_packets?
    ::mediapipe::CalculatorGraphConfig proto;
    auto status = createInputsInfo();
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create inputs info for mediapipe graph definition: {}", getName());
        return status;
    }
    status = createOutputsInfo();
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create outputs info for mediapipe graph definition: {}", getName());
        return status;
    }
    notifier.passed = true;
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Finished validation of mediapipe: {}", getName());
    return StatusCode::OK;
}

MediapipeGraphDefinition::MediapipeGraphDefinition(const std::string name,
    const MediapipeGraphConfig& config,
    MetricRegistry* registry,
    const MetricConfig* metricConfig) :
    name(name),
    status(this->name) {
    mgconfig = config;
}

Status MediapipeGraphDefinition::createInputsInfo() {
    for (auto& name : config.output_stream()) {
        outputsInfo.insert({name, TensorInfo::getUnspecifiedTensorInfo()});
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::createOutputsInfo() {
    for (auto& name : this->config.input_stream()) {
        inputsInfo.insert({name, TensorInfo::getUnspecifiedTensorInfo()});
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::create(std::shared_ptr<MediapipeGraphExecutor>& pipeline, const KFSRequest* request, KFSResponse* response) {
    if (!this->getStatus().isAvailable()) {
        SPDLOG_DEBUG("Failed to execute mediapipe graph: {} since it is not available", getName());
        return StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET;
    }
    pipeline = std::make_shared<MediapipeGraphExecutor>(getName(), std::to_string(getVersion()), this->config);
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::reload(ModelManager& manager, const MediapipeGraphConfig& config) {
    // TODO block creating new unloadGuards
    // TODO thread safety
    this->status.handle(ReloadEvent());
    this->mgconfig = config;
    return validate(manager);
}

void MediapipeGraphDefinition::retire(ModelManager& manager) {
    this->status.handle(RetireEvent());
}
}  // namespace ovms

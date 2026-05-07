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

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../execution_context.hpp"
#include "src/filesystem/filesystem.hpp"
#include "src/metrics/metric.hpp"
#include "../model_metric_reporter.hpp"
#include "../ov_utils.hpp"
#include "../servable_definition_unload_guard.hpp"
#include "../servable_name_checker.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../tensorinfo.hpp"
#include "../timer.hpp"
#include "../version.hpp"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe_utils.hpp"
#include "mediapipegraphexecutor.hpp"
#include "node_initializer.hpp"

namespace ovms {
MediapipeGraphConfig MediapipeGraphDefinition::MGC;

const std::string MediapipeGraphDefinition::SCHEDULER_CLASS_NAME{"Mediapipe"};

MediapipeGraphDefinition::~MediapipeGraphDefinition() = default;

const tensor_map_t MediapipeGraphDefinition::getInputsInfo() const {
    std::shared_lock lock(metadataMtx);
    return this->inputsInfo;
}

const tensor_map_t MediapipeGraphDefinition::getOutputsInfo() const {
    std::shared_lock lock(metadataMtx);
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
    std::stringstream config;
    config << ifs.rdbuf();
    this->mgconfig.setCurrentGraphPbTxtMD5(ovms::FileSystem::getStringMD5(config.str()));
    this->chosenConfig.assign(config.str());
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::validateForConfigLoadableness() {
    if (chosenConfig.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Trying to parse empty mediapipe graph definition: {} failed", this->getName(), this->chosenConfig);
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
    SPDLOG_TRACE("Will try to load pbtxt config: {}", this->chosenConfig);
    bool success = ::google::protobuf::TextFormat::ParseFromString(chosenConfig, &this->config);
    if (!success) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Trying to parse mediapipe graph definition: {} failed", this->getName(), this->chosenConfig);
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::dryInitializeTest() {
    ::mediapipe::CalculatorGraph graph;
    try {
        auto absStatus = graph.Initialize(this->config);
        if (!absStatus.ok()) {
            const std::string absMessage = absStatus.ToString();
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Mediapipe graph: {} initialization failed with message: {}", this->getName(), absMessage);
            return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, std::move(absMessage));
        }
    } catch (std::exception& e) {
        SPDLOG_ERROR("Exception caught whilie trying to initialize MediaPipe graph: {}", e.what());
        return StatusCode::UNKNOWN_ERROR;
    } catch (...) {
        SPDLOG_ERROR("Exception caught whilie trying to initialize MediaPipe graph.");
        return StatusCode::UNKNOWN_ERROR;
    }
    return StatusCode::OK;
}
Status MediapipeGraphDefinition::validate(const ServableNameChecker& checker) {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Started validation of mediapipe: {}", getName());
    if (!this->sidePacketMaps.empty()) {
        SPDLOG_ERROR("Internal Error: MediaPipe definition is in unexpected state.");
        return StatusCode::INTERNAL_ERROR;
    }
    ValidationResultNotifier notifier(this->status, this->loadedNotify);
    if (checker.servableExists(this->getName(), ServableQueryType::Model | ServableQueryType::Pipeline)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Mediapipe graph name: {} is already occupied by model or pipeline.", this->getName());
        return StatusCode::MEDIAPIPE_GRAPH_NAME_OCCUPIED;
    }
    Status validationResult = validateForConfigFileExistence();
    if (!validationResult.ok()) {
        return validationResult;
    }
    validationResult = validateForConfigLoadableness();
    if (!validationResult.ok()) {
        return validationResult;
    }
    std::unique_lock lock(metadataMtx);
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
    status = createInputSidePacketsInfo();
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create input side packets info for mediapipe graph definition: {}", getName());
        return status;
    }
    // Detect what deserialization needs to be performed
    status = this->setStreamTypes();
    if (!status.ok()) {
        return status;
    }
    // here we will not be available if calculator does not exist in OVMS
    status = this->dryInitializeTest();
    if (!status.ok()) {
        return status;
    }

    status = this->initializeNodes();
    if (!status.ok()) {
        return status;
    }

    lock.unlock();
    notifier.passed = true;
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Finished validation of mediapipe: {}", getName());
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Mediapipe: {} inputs: {}", getName(), getTensorMapString(inputsInfo));
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Mediapipe: {} outputs: {}", getName(), getTensorMapString(outputsInfo));
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Mediapipe: {} kfs pass through: {}", getName(), this->passKfsRequestFlag);
    return StatusCode::OK;
}

MediapipeGraphDefinition::MediapipeGraphDefinition(const std::string name,
    const MediapipeGraphConfig& config,
    MetricRegistry* registry,
    const MetricConfig* metricConfig,
    PythonBackend* pythonBackend) :
    SingleVersionServableDefinition(name),
    status(SCHEDULER_CLASS_NAME, getName()),
    pythonBackend(pythonBackend),
    reporter(std::make_unique<MediapipeServableMetricReporter>(metricConfig, registry, name)) {
    mgconfig = config;
    passKfsRequestFlag = false;
}

Status MediapipeGraphDefinition::createInputsInfo() {
    inputsInfo.clear();
    inputNames.clear();
    inputNames.reserve(this->config.input_stream().size());
    for (auto& name : config.input_stream()) {
        std::string streamName = getStreamName(name);
        if (streamName.empty()) {
            SPDLOG_ERROR("Creating Mediapipe graph inputs name failed for: {}", name);
            return StatusCode::MEDIAPIPE_WRONG_INPUT_STREAM_PACKET_NAME;
        }
        const auto [it, success] = inputsInfo.insert({streamName, TensorInfo::getUnspecifiedTensorInfo()});
        if (!success) {
            SPDLOG_ERROR("Creating Mediapipe graph inputs name failed for: {}. Input with the same name already exists.", name);
            return StatusCode::MEDIAPIPE_WRONG_INPUT_STREAM_PACKET_NAME;
        }
        inputNames.emplace_back(std::move(streamName));
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::createInputSidePacketsInfo() {
    inputSidePacketNames.clear();
    for (auto& name : config.input_side_packet()) {
        std::string streamName = getStreamName(name);
        if (streamName.empty()) {
            SPDLOG_ERROR("Creating Mediapipe graph input side packet name failed for: {}", name);
            return StatusCode::MEDIAPIPE_WRONG_INPUT_SIDE_PACKET_STREAM_PACKET_NAME;
        }
        inputSidePacketNames.emplace_back(std::move(streamName));
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::createOutputsInfo() {
    outputsInfo.clear();
    outputNames.clear();
    outputNames.reserve(this->config.output_stream().size());
    for (auto& name : this->config.output_stream()) {
        std::string streamName = getStreamName(name);
        if (streamName.empty()) {
            SPDLOG_ERROR("Creating Mediapipe graph outputs name failed for: {}", name);
            return StatusCode::MEDIAPIPE_WRONG_OUTPUT_STREAM_PACKET_NAME;
        }
        const auto [it, success] = outputsInfo.insert({streamName, TensorInfo::getUnspecifiedTensorInfo()});
        if (!success) {
            SPDLOG_ERROR("Creating Mediapipe graph outputs name failed for: {}. Output with the same name already exists.", name);
            return StatusCode::MEDIAPIPE_WRONG_OUTPUT_STREAM_PACKET_NAME;
        }
        outputNames.emplace_back(std::move(streamName));
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::create(std::unique_ptr<MediapipeGraphExecutor>& pipeline) {
    std::unique_ptr<ServableDefinitionUnloadGuard> unloadGuard;
    Status status = waitForLoaded(unloadGuard);
    if (!status.ok()) {
        SPDLOG_DEBUG("Failed to execute mediapipe graph: {} since it is not available", getName());
        return status;
    }
    SPDLOG_DEBUG("Creating Mediapipe graph executor: {}", getName());

    pipeline = std::make_unique<MediapipeGraphExecutor>(getName(), std::to_string(getVersion()),
        this->config, this->inputTypes, this->outputTypes, this->inputNames, this->outputNames,
        this->sidePacketMaps,
        this->pythonBackend, this->reporter.get());
    return status;
}

Status MediapipeGraphDefinition::setStreamTypes() {
    this->inputTypes.clear();
    this->outputTypes.clear();
    this->passKfsRequestFlag = false;
    if (!this->config.input_stream().size() ||
        !this->config.output_stream().size()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to prepare mediapipe graph: {}; having less than one input or output is disallowed", getName());
        // validation is incomplete in case this error is triggered
        return StatusCode::INTERNAL_ERROR;
    }
    for (auto& inputStreamName : this->config.input_stream()) {
        inputTypes.emplace(getStreamNamePair(inputStreamName, MediaPipeStreamType::INPUT));
    }
    for (auto& outputStreamName : this->config.output_stream()) {
        outputTypes.emplace(getStreamNamePair(outputStreamName, MediaPipeStreamType::OUTPUT));
    }
    bool anyInputTfLite = std::any_of(inputTypes.begin(), inputTypes.end(), [](const auto& p) {
        const auto& [k, v] = p;
        return v == mediapipe_packet_type_enum::TFLITETENSOR;
    });
    bool anyOutputTfLite = std::any_of(outputTypes.begin(), outputTypes.end(), [](const auto& p) {
        const auto& [k, v] = p;
        return v == mediapipe_packet_type_enum::TFLITETENSOR;
    });
    if (anyInputTfLite || anyOutputTfLite) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "There is no support for TfLiteTensor deserialization & serialization");
        return StatusCode::NOT_IMPLEMENTED;
    }
    bool kfsRequestPass = std::any_of(inputTypes.begin(), inputTypes.end(), [](const auto& p) {
        const auto& [k, v] = p;
        return v == mediapipe_packet_type_enum::KFS_REQUEST;
    });
    bool kfsResponsePass = std::any_of(outputTypes.begin(), outputTypes.end(), [](const auto& p) {
        const auto& [k, v] = p;
        return v == mediapipe_packet_type_enum::KFS_RESPONSE;
    });
    if (kfsRequestPass) {
        if (!kfsResponsePass) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to prepare mediapipe graph configuration: {}; KFS passthrough mode is misconfigured. KServe for mediapipe graph passing whole KFS request and response requires: {} tag in the output stream name", getName(), KFS_RESPONSE_PREFIX);
            return Status(StatusCode::MEDIAPIPE_KFS_PASSTHROUGH_MISSING_OUTPUT_RESPONSE_TAG);

        } else {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "KServe for mediapipe graph: {}; passing whole KFS request graph detected.", getName());
        }
    } else if (kfsResponsePass) {
        if (!kfsRequestPass) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to prepare mediapipe graph configuration: {}; KServe for mediapipe graph passing whole KFS request and response requires: {} tag in the input stream name", getName(), KFS_REQUEST_PREFIX);
            return Status(StatusCode::MEDIAPIPE_KFS_PASSTHROUGH_MISSING_INPUT_REQUEST_TAG);
        }
    }
    if (kfsRequestPass == true) {
        if (this->config.output_stream().size() != 1) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "KServe passthrough through mediapipe graph requires having only one output(response)");
            return StatusCode::MEDIAPIPE_KFS_PASS_WRONG_OUTPUT_STREAM_COUNT;
        }
        if (this->config.input_stream().size() != 1) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "KServe passthrough through mediapipe graph requires having only one input (request)");
            return StatusCode::MEDIAPIPE_KFS_PASS_WRONG_INPUT_STREAM_COUNT;
        }
    }
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::reload(const ServableNameChecker& checker, const MediapipeGraphConfig& config) {
    // block creating new unloadGuards
    this->status.handle(ReloadEvent());
    while (requestsHandlesCounter > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    this->mgconfig = config;
    this->sidePacketMaps.clear();
    return validate(checker);
}

void MediapipeGraphDefinition::retire() {
    this->sidePacketMaps.clear();
    this->status.handle(RetireEvent());
}

bool MediapipeGraphDefinition::isReloadRequired(const MediapipeGraphConfig& config) const {
    if (getStateCode() == PipelineDefinitionStateCode::RETIRED) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reloading previously retired mediapipe definition: {}", getName());
        return true;
    }
    return getMediapipeGraphConfig().isReloadRequired(config);
}

StatusCode MediapipeGraphDefinition::notLoadedYetCode() const {
    return StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET;
}

StatusCode MediapipeGraphDefinition::notLoadedAnymoreCode() const {
    return StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE;
}

Status MediapipeGraphDefinition::initializeNodes() {
    SPDLOG_INFO("MediapipeGraphDefinition initializing graph nodes");
    bool success = false;
    struct CleanupGuard {
        GraphSidePackets& maps;
        bool& success;
        ~CleanupGuard() {
            if (!success)
                maps.clear();
        }
    } guard{sidePacketMaps, success};

    auto& registry = NodeInitializerRegistry::instance();
    for (int i = 0; i < config.node().size(); i++) {
        for (const auto& initializer : registry.all()) {
            if (initializer->matches(config.node(i).calculator())) {
                Status status = initializer->initialize(config.node(i), getName(), mgconfig.getBasePath(), sidePacketMaps, pythonBackend);
                if (!status.ok()) {
                    return status;
                }
            }
        }
    }
    success = true;
    return StatusCode::OK;
}
}  // namespace ovms

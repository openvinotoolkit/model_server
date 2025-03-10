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
#include "../filesystem.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../kfs_frontend/kfs_request_utils.hpp"
#include "../deserialization_main.hpp"
#include "../metric.hpp"
#include "../model_metric_reporter.hpp"
#include "../modelmanager.hpp"
#include "../ov_utils.hpp"
#if (PYTHON_DISABLE == 0)
#include "../llm/servable.hpp"
#include "../llm/servable_initializer.hpp"
#include "../python/pythonnoderesources.hpp"
#endif
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../tensorinfo.hpp"
#include "../timer.hpp"
#include "../version.hpp"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe_utils.hpp"
#include "mediapipegraphexecutor.hpp"

namespace ovms {
MediapipeGraphConfig MediapipeGraphDefinition::MGC;

const std::string MediapipeGraphDefinition::SCHEDULER_CLASS_NAME{"Mediapipe"};
const std::string MediapipeGraphDefinition::PYTHON_NODE_CALCULATOR_NAME{"PythonExecutorCalculator"};
const std::string MediapipeGraphDefinition::LLM_NODE_CALCULATOR_NAME{"LLMCalculator"};

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
    this->queue = std::make_unique<GraphQueue>(this->config, 48);
    this->mgconfig.setCurrentGraphPbTxtMD5(ovms::FileSystem::getStringMD5(config.str()));
    this->chosenConfig.assign(config.str());
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::validateForConfigLoadableness() {
    if (chosenConfig.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Trying to parse empty mediapipe graph definition: {} failed", this->getName(), this->chosenConfig);
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }

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
Status MediapipeGraphDefinition::validate(ModelManager& manager) {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Started validation of mediapipe: {}", getName());
    if (!this->pythonNodeResourcesMap.empty()) {
        SPDLOG_ERROR("Internal Error: MediaPipe definition is in unexpected state.");
        return StatusCode::INTERNAL_ERROR;
    }
    if (!this->genAiServableMap.empty()) {
        SPDLOG_ERROR("Internal Error: MediaPipe definition is in unexpected state.");
        return StatusCode::INTERNAL_ERROR;
    }
    ValidationResultNotifier notifier(this->status, this->loadedNotify);
    if (manager.modelExists(this->getName()) || manager.pipelineDefinitionExists(this->getName())) {
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
    name(name),
    status(SCHEDULER_CLASS_NAME, this->name),
    pythonBackend(pythonBackend),
    reporter(std::make_unique<MediapipeServableMetricReporter>(metricConfig, registry, name))
    {
    mgconfig = config;
    passKfsRequestFlag = false;
    /*if (!sharedThreadPool) {
        SPDLOG_ERROR("Created shared Thread Pool XXX");
        //sharedThreadPool = std::make_shared<mediapipe::ThreadPoolExecutor>(std::thread::hardware_concurrency());  // TODO FIXME should be in MP factory
    }*/
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

Status MediapipeGraphDefinition::create(std::shared_ptr<MediapipeGraphExecutor>& pipeline) {
    std::unique_ptr<MediapipeGraphDefinitionUnloadGuard> unloadGuard;
    Status status = waitForLoaded(unloadGuard);
    if (!status.ok()) {
        SPDLOG_DEBUG("Failed to execute mediapipe graph: {} since it is not available", getName());
        return status;
    }
    SPDLOG_DEBUG("Creating Mediapipe graph executor: {}", getName());
    GraphIdGuard graphIdGuard(*(this->queue)); // TODO timeout?
    pipeline = std::make_shared<MediapipeGraphExecutor>(getName(), std::to_string(getVersion()),
        this->config, this->inputTypes, this->outputTypes, this->inputNames, this->outputNames,
        this->pythonNodeResourcesMap, this->llmNodeResourcesMap, this->pythonBackend, this->reporter.get(), std::move(graphIdGuard));
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

Status MediapipeGraphDefinition::reload(ModelManager& manager, const MediapipeGraphConfig& config) {
    // block creating new unloadGuards
    this->status.handle(ReloadEvent());
    while (requestsHandlesCounter > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    this->mgconfig = config;
    this->pythonNodeResourcesMap.clear();
    this->genAiServableMap.clear();
    return validate(manager);
}

void MediapipeGraphDefinition::retire(ModelManager& manager) {
    this->pythonNodeResourcesMap.clear();
    this->genAiServableMap.clear();
    this->status.handle(RetireEvent());
}

bool MediapipeGraphDefinition::isReloadRequired(const MediapipeGraphConfig& config) const {
    if (getStateCode() == PipelineDefinitionStateCode::RETIRED) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reloading previously retired mediapipe definition: {}", getName());
        return true;
    }
    return getMediapipeGraphConfig().isReloadRequired(config);
}

Status MediapipeGraphDefinition::waitForLoaded(std::unique_ptr<MediapipeGraphDefinitionUnloadGuard>& unloadGuard, const uint32_t waitForLoadedTimeoutMicroseconds) {
    unloadGuard = std::make_unique<MediapipeGraphDefinitionUnloadGuard>(*this);

    const uint32_t waitLoadedTimestepMicroseconds = 1000;
    const uint32_t waitCheckpoints = waitForLoadedTimeoutMicroseconds / waitLoadedTimestepMicroseconds;
    uint32_t waitCheckpointsCounter = waitCheckpoints;
    std::mutex cvMtx;
    std::unique_lock<std::mutex> cvLock(cvMtx);
    while (waitCheckpointsCounter-- != 0) {
        if (status.isAvailable()) {
            SPDLOG_DEBUG("Successfully waited for mediapipe definition: {}", getName());
            return StatusCode::OK;
        }
        unloadGuard.reset();
        if (!status.canEndLoaded()) {
            if (status.getStateCode() != PipelineDefinitionStateCode::RETIRED) {
                SPDLOG_DEBUG("Waiting for mediapipe definition: {} ended due to timeout.", getName());
                return StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET;
            } else {
                SPDLOG_DEBUG("Waiting for mediapipe definition: {} ended since it failed to load.", getName());
                return StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE;
            }
        }
        SPDLOG_DEBUG("Waiting for available state for mediapipe: {}, with timestep: {}us timeout: {}us check count: {}",
            getName(), waitLoadedTimestepMicroseconds, waitForLoadedTimeoutMicroseconds, waitCheckpointsCounter);
        loadedNotify.wait_for(cvLock,
            std::chrono::microseconds(waitLoadedTimestepMicroseconds),
            [this]() {
                return this->status.isAvailable() ||
                       !this->status.canEndLoaded();
            });
        unloadGuard = std::make_unique<MediapipeGraphDefinitionUnloadGuard>(*this);
    }
    if (!status.isAvailable()) {
        if (status.getStateCode() != PipelineDefinitionStateCode::RETIRED) {
            SPDLOG_DEBUG("Waiting for mediapipe definition: {} ended due to timeout.", getName());
            return StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET;
        } else {
            SPDLOG_DEBUG("Waiting for mediapipe definition: {} ended since it failed to load.", getName());
            return StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE;
        }
    }
    SPDLOG_DEBUG("Successfully waited for mediapipe definition: {}", getName());
    return StatusCode::OK;
}

#if (PYTHON_DISABLE == 0)
template <typename T>
class ResourcesCleaningGuard {
public:
    bool shouldCleanup{true};
    T& resources;
    ResourcesCleaningGuard(T& resources) :
        resources(resources) {}
    ~ResourcesCleaningGuard() {
        if (shouldCleanup) {
            resources.clear();
        }
    }
    void disableCleaning() {
        shouldCleanup = false;
    }
};
#endif

Status MediapipeGraphDefinition::initializeNodes() {
    SPDLOG_INFO("MediapipeGraphDefinition initializing graph nodes");
    for (int i = 0; i < config.node().size(); i++) {
#if (PYTHON_DISABLE == 0)
        if (config.node(i).calculator() == PYTHON_NODE_CALCULATOR_NAME) {
            ResourcesCleaningGuard<PythonNodeResourcesMap> pythonResourcesCleaningGuard(this->pythonNodeResourcesMap);
            if (!config.node(i).node_options().size()) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Python node missing options in graph: {}. ", this->name);
                return StatusCode::PYTHON_NODE_MISSING_OPTIONS;
            }
            if (config.node(i).name().empty()) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Python node name is missing in graph: {}. ", this->name);
                return StatusCode::PYTHON_NODE_MISSING_NAME;
            }
            std::string nodeName = config.node(i).name();
            if (this->pythonNodeResourcesMap.find(nodeName) != this->pythonNodeResourcesMap.end()) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Python node name: {} already used in graph: {}. ", nodeName, this->name);
                return StatusCode::PYTHON_NODE_NAME_ALREADY_EXISTS;
            }

            std::shared_ptr<PythonNodeResources> nodeResources = nullptr;
            Status status = PythonNodeResources::createPythonNodeResources(nodeResources, config.node(i), pythonBackend, mgconfig.getBasePath());
            if (nodeResources == nullptr || !status.ok()) {
                SPDLOG_ERROR("Failed to process python node graph {}", this->name);
                return status;
            }

            this->pythonNodeResourcesMap.insert(std::pair<std::string, std::shared_ptr<PythonNodeResources>>(nodeName, std::move(nodeResources)));
            pythonResourcesCleaningGuard.disableCleaning();
        }
        // Passed to both calculators that require LLM Engine (gRPC KServe & HTTP OpenAI)
        if (endsWith(config.node(i).calculator(), LLM_NODE_CALCULATOR_NAME)) {
            ResourcesCleaningGuard<GenAiServableMap> genAiServablesCleaningGuard(this->genAiServableMap);
            if (!config.node(i).node_options().size()) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node missing options in graph: {}. ", this->name);
                return StatusCode::LLM_NODE_MISSING_OPTIONS;
            }
            if (config.node(i).name().empty()) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node name is missing in graph: {}. ", this->name);
                return StatusCode::LLM_NODE_MISSING_NAME;
            }
            std::string nodeName = config.node(i).name();
            if (this->genAiServableMap.find(nodeName) != this->genAiServableMap.end()) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "LLM node name: {} already used in graph: {}. ", nodeName, this->name);
                return StatusCode::LLM_NODE_NAME_ALREADY_EXISTS;
            }
            std::shared_ptr<GenAiServable> servable;
            Status status = initializeGenAiServable(servable, config.node(i), mgconfig.getBasePath());
            if (!status.ok()) {
                SPDLOG_ERROR("Failed to process LLM node graph {}", this->name);
                return status;
            }
            this->genAiServableMap.insert(std::pair<std::string, std::shared_ptr<GenAiServable>>(nodeName, std::move(servable)));
            genAiServablesCleaningGuard.disableCleaning();
        }
#endif
    }
    return StatusCode::OK;
}
}  // namespace ovms

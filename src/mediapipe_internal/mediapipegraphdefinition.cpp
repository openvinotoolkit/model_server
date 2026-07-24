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
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../execution_context.hpp"
#include "../config.hpp"
#include "src/utils/env_guard.hpp"
#include "src/filesystem/filesystem.hpp"
#include "src/graph_export/graph_export.hpp"
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
    if (GraphExport::hasInMemoryGraphContent() && ovms::Config::instance().getServerSettings().serverMode == IN_MEMORY_GRAPH_MODE) {
        const std::string& content = GraphExport::getInMemoryGraphContent();
        this->chosenConfig = content;
        this->mgconfig.setCurrentGraphPbTxtMD5(ovms::FileSystem::getStringMD5(content));
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Using in-memory graph content for mediapipe graph definition: {}", this->getName());
        return StatusCode::OK;
    }
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

// Precondition: chosenConfig must be populated (call after validateForConfigFileExistence).
// Uses this->chosenConfig to parse the directive and this->config (parsed protobuf) for node inspection.
Status MediapipeGraphDefinition::resolveGraphQueueSize() {
    if (this->chosenConfig.empty()) {
        SPDLOG_ERROR("Internal error: resolveGraphQueueSize called with empty chosenConfig for mediapipe: {}", getName());
        return StatusCode::INTERNAL_ERROR;
    }
    // 0. Runtime kill-switch: OVMS_GRAPH_QUEUE_OFF disables all graph pools.
    if (GetEnvVar("OVMS_GRAPH_QUEUE_OFF") == "1") {
        SPDLOG_INFO("Graph queue globally disabled via OVMS_GRAPH_QUEUE_OFF=1 for mediapipe: {}", getName());
        return StatusCode::OK;
    }
    // 1. Explicit pbtxt directive: # OVMS_GRAPH_QUEUE_MAX_SIZE: <value>
    //    Always honored regardless of calculator checks.
    //    Value 0 disables the queue, AUTO or positive integer enables it.
    //    Negative values are rejected as invalid.
    static const std::regex directiveRegex(
        R"((?:^|\n)\s*#\s*OVMS_GRAPH_QUEUE_MAX_SIZE\s*:\s*(\S+)\s*(?:\r?\n|$))");
    std::smatch match;
    if (std::regex_search(this->chosenConfig, match, directiveRegex)) {
        std::string value = match[1].str();
        if (value == "AUTO") {
            this->mgconfig.setGraphQueueSizeAuto();
        } else {
            auto parsed = stoi32(value);
            if (!parsed.has_value()) {
                SPDLOG_ERROR("Invalid OVMS_GRAPH_QUEUE_MAX_SIZE value: '{}'. Expected integer or 'AUTO'.", value);
                return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
            }
            int queueSize = parsed.value();
            if (queueSize < 0) {
                SPDLOG_ERROR("Invalid OVMS_GRAPH_QUEUE_MAX_SIZE value: {}. Must be 0 (disabled) or a positive integer.", queueSize);
                return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
            }
            if (queueSize == 0) {
                SPDLOG_DEBUG("Graph queue explicitly disabled (OVMS_GRAPH_QUEUE_MAX_SIZE=0) for mediapipe: {}", getName());
                return StatusCode::OK;
            }
            unsigned int maxThreads = std::thread::hardware_concurrency();
            if (maxThreads > 0 && queueSize > static_cast<int>(maxThreads)) {
                SPDLOG_WARN("OVMS_GRAPH_QUEUE_MAX_SIZE value: {} exceeds available hardware threads: {}. Clamping to {}.", queueSize, maxThreads, maxThreads);
                queueSize = static_cast<int>(maxThreads);
            }
            this->mgconfig.setGraphQueueSize(queueSize);
        }
        // 2. Reject PythonExecutorCalculator nodes using LOOPBACK with graph queue enabled.
        //    Generative Python nodes hold per-request iterator state that cannot be shared
        //    across pooled graph instances.
        for (int i = 0; i < this->config.node_size(); ++i) {
            const auto& node = this->config.node(i);
            if (node.calculator() != "PythonExecutorCalculator") {
                continue;
            }
            for (const auto& inputStream : node.input_stream()) {
                if (inputStream.find("LOOPBACK") == 0) {
                    SPDLOG_ERROR("PythonExecutorCalculator with LOOPBACK stream is incompatible with graph queue "
                                 "(OVMS_GRAPH_QUEUE_MAX_SIZE) in mediapipe: {}. "
                                 "Generative Python nodes hold per-request state that cannot be shared across pooled graphs. "
                                 "Set OVMS_GRAPH_QUEUE_MAX_SIZE to 0 or remove the LOOPBACK stream.",
                        getName());
                    return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
                }
            }
        }
        return StatusCode::OK;
    }
    SPDLOG_DEBUG("Graph queue disabled by default for mediapipe: {}. Add '# OVMS_GRAPH_QUEUE_MAX_SIZE: <value>' directive in graph.pbtxt to enable.", getName());
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::validateForConfigLoadableness() {
    if (this->chosenConfig.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Trying to parse empty mediapipe graph definition: {} failed", this->getName(), this->chosenConfig);
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
    SPDLOG_TRACE("Will try to load pbtxt config: {}", this->chosenConfig);
    bool success = ::google::protobuf::TextFormat::ParseFromString(this->chosenConfig, &this->config);
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
    if (!this->sidePacketMaps->empty()) {
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
    // Phase-1 restriction: idle unload is not supported for graphs with Python nodes.
    // Python nodes may hold per-request iterator state (e.g. PythonExecutorCalculator)
    // that cannot be safely reconstructed after a resource-free/reload cycle.
    if (mgconfig.getIdleUnloadTimeoutSeconds() > 0) {
        for (int i = 0; i < this->config.node_size(); ++i) {
            const std::string& calculator = this->config.node(i).calculator();
            if (calculator == "PythonExecutorCalculator" || calculator == "PyTorchCalculator") {
                SPDLOG_LOGGER_ERROR(modelmanager_logger,
                    "Mediapipe graph {}: idle_unload_timeout_seconds is not supported for graphs "
                    "containing Python calculator nodes ({}). "
                    "Remove idle_unload_timeout_seconds or remove the Python node.",
                    getName(), calculator);
                return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
            }
        }
        // Phase-1 scope restriction: idle_unload_timeout_seconds is validated only for
        // LLM/VLM continuous-batching graphs (graphs containing HttpLLMCalculator).
        // Other node types (embeddings, rerank, STT, TTS, image-gen, plain passthrough)
        // have not been validated for the idle-unload/lazy-reload cycle.
        bool hasLlmCalculator = false;
        for (int i = 0; i < this->config.node_size(); ++i) {
            if (this->config.node(i).calculator() == "HttpLLMCalculator") {
                hasLlmCalculator = true;
                break;
            }
        }
        if (!hasLlmCalculator) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger,
                "Mediapipe graph {}: idle_unload_timeout_seconds is only supported for "
                "LLM/VLM continuous-batching graphs (HttpLLMCalculator) in this release. "
                "Remove idle_unload_timeout_seconds from non-LLM graph configurations.",
                getName());
            return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
        }
    }

    validationResult = resolveGraphQueueSize();
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
    status = this->initializeQueueIfRequired();
    if (!status.ok()) {
        return status;
    }

    if (!this->loraAliases.empty() && checker.aliasesConflict(this->loraAliases, getName())) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "LoRA alias in graph '{}' conflicts with an existing servable", getName());
        return StatusCode::MEDIAPIPE_GRAPH_NAME_OCCUPIED;
    }

    lock.unlock();
    notifier.passed = true;
    // Graph resources are now loaded (covers both initial load and wake-up reload).
    SET_IF_ENABLED(this->reporter->graphLoaded, 1);
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Finished validation of mediapipe: {}", getName());
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Mediapipe: {} inputs: {}", getName(), getTensorMapString(inputsInfo));
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Mediapipe: {} outputs: {}", getName(), getTensorMapString(outputsInfo));
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Mediapipe: {} kfs pass through: {}", getName(), this->passKfsRequestFlag);
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::initializeQueueIfRequired() {
    int initialQueueSize = this->mgconfig.getInitialQueueSize();
    if (initialQueueSize <= 0) {
        SPDLOG_DEBUG("Graph queue creation disabled for mediapipe: {} (graph_queue_size={})", getName(), initialQueueSize);
        return StatusCode::OK;
    }
    try {
        this->queue = std::make_shared<GraphQueue>(this->config, this->sidePacketMaps, initialQueueSize);
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create graph queue for mediapipe: {} error: {}", getName(), e.what());
        return StatusCode::INTERNAL_ERROR;
    } catch (...) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create graph queue for mediapipe: {} unknown error", getName());
        return StatusCode::INTERNAL_ERROR;
    }
    SPDLOG_DEBUG("Created graph queue with size {} for mediapipe: {}", initialQueueSize, getName());
    return StatusCode::OK;
}

MediapipeGraphDefinition::MediapipeGraphDefinition(const std::string name,
    const MediapipeGraphConfig& config,
    MetricRegistry* registry,
    const MetricConfig* metricConfig,
    PythonBackend* pythonBackend) :
    SingleVersionServableDefinition(name),
    sidePacketMaps(std::make_shared<GraphSidePackets>()),
    status(SCHEDULER_CLASS_NAME, getName()),
    pythonBackend(pythonBackend),
    reporter(std::make_unique<MediapipeServableMetricReporter>(metricConfig, registry, name)) {
    mgconfig = config;
    idleUnloadTimeoutSecondsCache.store(mgconfig.getIdleUnloadTimeoutSeconds(), std::memory_order_relaxed);
    passKfsRequestFlag = false;
    // Allocate lastActivityTimeNs initialized to now so the idle timer starts from
    // when the graph was loaded, not from the steady_clock epoch (which would
    // cause immediate unload of a freshly loaded graph before any request arrives).
    // Held as a shared_ptr so executors can safely refresh it even after the
    // definition is retired/destroyed.
    lastActivityTimeNs = std::make_shared<std::atomic<int64_t>>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    // Allocate the active-inference counter once; shared with every executor
    // created by this definition so executors can decrement it on completion.
    activeInferenceCount = std::make_shared<std::atomic<int64_t>>(0);
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
    // Update idle-tracking timestamp on every inference acquisition path.
    // Status endpoints / health checks do not reach this method, so idle
    // tracking is automatically inference-only.
    lastActivityTimeNs->store(
        std::chrono::steady_clock::now().time_since_epoch().count(),
        std::memory_order_relaxed);

    std::unique_ptr<ServableDefinitionUnloadGuard> unloadGuard;
    Status status = waitForLoaded(unloadGuard);
    if (!status.ok()) {
        SPDLOG_DEBUG("Failed to execute mediapipe graph: {} since it is not available", getName());
        return status;
    }
    SPDLOG_DEBUG("Creating Mediapipe graph executor: {}", getName());
    if (this->queue) {
        GraphIdGuard graphIdGuard(this->queue);
        pipeline = std::make_unique<MediapipeGraphExecutor>(getName(), std::to_string(getVersion()),
            this->config, this->inputTypes, this->outputTypes, this->inputNames, this->outputNames,
            *this->sidePacketMaps,
            this->pythonBackend, this->reporter.get(), std::move(graphIdGuard),
            this->activeInferenceCount, this->lastActivityTimeNs);
    } else {
        pipeline = std::make_unique<MediapipeGraphExecutor>(getName(), std::to_string(getVersion()),
            this->config, this->inputTypes, this->outputTypes, this->inputNames, this->outputNames,
            *this->sidePacketMaps,
            this->pythonBackend, this->reporter.get(),
            this->activeInferenceCount, this->lastActivityTimeNs);
    }
    SPDLOG_DEBUG("Created Mediapipe graph executor: {}", getName());
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
    // Serialize against unload()/wakeUp() on the watcher/request threads.
    // Recursive: wakeUpIfUnloaded() already holds this and calls reload().
    std::lock_guard<std::recursive_mutex> lock(lifecycleMtx);
    // block creating new unloadGuards
    this->status.handle(ReloadEvent());
    while (requestsHandlesCounter > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    this->mgconfig = config;
    // Refresh the lock-free cache while we still hold lifecycleMtx.
    idleUnloadTimeoutSecondsCache.store(this->mgconfig.getIdleUnloadTimeoutSeconds(), std::memory_order_relaxed);
    this->queue.reset();
    this->sidePacketMaps = std::make_shared<GraphSidePackets>();
    return validate(checker);
}

void MediapipeGraphDefinition::retire() {
    // Serialize against unload()/wakeUp()/reload() on other threads.
    std::lock_guard<std::recursive_mutex> lock(lifecycleMtx);
    // Block creating new unloadGuards
    this->status.handle(RetireEvent());
    while (requestsHandlesCounter > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    this->queue.reset();
    this->sidePacketMaps.reset();
}

bool MediapipeGraphDefinition::isIdleUnloadEnabled() const {
    // Lock-free read of the cached timeout (mgconfig is only safe under lifecycleMtx).
    return idleUnloadTimeoutSecondsCache.load(std::memory_order_relaxed) > 0;
}

bool MediapipeGraphDefinition::shouldUnloadDueToIdle() const {
    // Advisory pre-filter ONLY — reads no unsynchronized per-definition state.
    // It must NOT read this->status (the state-machine variant) without the lock,
    // since the config thread can mutate it concurrently. unload() performs the
    // authoritative state==AVAILABLE check under lifecycleMtx.
    // requestsHandlesCounter, lastActivityTimeNs and idleUnloadTimeoutSecondsCache
    // are all atomics, so every read here is data-race-free. We never read mgconfig
    // (only safe under lifecycleMtx) on this advisory path.
    int64_t timeoutSeconds = idleUnloadTimeoutSecondsCache.load(std::memory_order_relaxed);
    if (timeoutSeconds <= 0) {
        return false;
    }
    if (requestsHandlesCounter.load(std::memory_order_relaxed) != 0) {
        return false;
    }
    // Guard: if inferences are actively executing, never report idle.
    // activeInferenceCount is bumped when a MediapipeGraphExecutor is created (in
    // create()) by its RAII ActiveInferenceGuard, held for the executor's lifetime
    // (which spans the inference), and decremented (with a lastActivityTimeNs refresh)
    // when the executor is destroyed after the inference completes or throws.
    if (activeInferenceCount && activeInferenceCount->load(std::memory_order_acquire) > 0) {
        return false;
    }
    int64_t lastActivity = lastActivityTimeNs->load(std::memory_order_relaxed);
    int64_t nowNs = std::chrono::steady_clock::now().time_since_epoch().count();
    int64_t timeoutNs = timeoutSeconds * 1'000'000'000LL;
    return (nowNs - lastActivity) >= timeoutNs;
}

Status MediapipeGraphDefinition::unload() {
    // Serialize against wakeUpIfUnloaded()/reload()/retire() using the SAME lock so
    // all lifecycle mutations are mutually exclusive. This prevents the watcher thread
    // from tearing down resources while the config thread reloads/retires, or while a
    // request thread is in the middle of a wake-up reload.
    std::lock_guard<std::recursive_mutex> lock(lifecycleMtx);

    // Re-check the preconditions under the lock. Only AVAILABLE graphs with no
    // in-flight requests may be unloaded. If the state changed (e.g. a wake-up
    // moved us to RELOADING/AVAILABLE) or a request arrived after the watcher's
    // shouldUnloadDueToIdle() check, skip this cycle WITHOUT touching resources.
    if (status.getStateCode() != PipelineDefinitionStateCode::AVAILABLE) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger,
            "Skipping idle-unload of mediapipe graph {}: state is no longer AVAILABLE", getName());
        return StatusCode::OK;
    }
    if (requestsHandlesCounter.load(std::memory_order_acquire) != 0) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger,
            "Skipping idle-unload of mediapipe graph {}: requests in flight", getName());
        return StatusCode::OK;
    }
    // Guard: if inferences are actively executing, skip this unload cycle.
    // This catches the case where the executor's inference is running (past create())
    // but before ActiveInferenceGuard has decremented — i.e. a long generation that
    // outlives idle_unload_timeout_seconds. Re-checked under the lock so the decision
    // is consistent with the in-flight inference completing concurrently.
    if (activeInferenceCount && activeInferenceCount->load(std::memory_order_acquire) > 0) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger,
            "Skipping idle-unload of mediapipe graph {}: active inferences in progress", getName());
        return StatusCode::OK;
    }

    // Transition state: AVAILABLE -> UNLOADED (blocks new unloadGuards in waitForLoaded).
    this->status.handle(UnloadEvent());

    // Defensive: only tear down resources if the transition actually happened.
    // (UnloadEvent is a no-op on any non-AVAILABLE state.)
    if (status.getStateCode() != PipelineDefinitionStateCode::UNLOADED) {
        SPDLOG_LOGGER_WARN(modelmanager_logger,
            "Idle-unload of mediapipe graph {} aborted: state did not transition to UNLOADED (now {})",
            getName(), pipelineDefinitionStateCodeToString(status.getStateCode()));
        return StatusCode::OK;
    }

    // Once UNLOADED, no new unloadGuards can be acquired and we verified
    // requestsHandlesCounter == 0 above, so there is nothing to drain.
    // Release queue (pooled graphs hold GPU/CPU resources).
    this->queue.reset();
    // Release heavy side-packet resources (GenAI servables, embeddings, etc.)
    // Keep the sidePacketMaps object itself — clear() drops the shared_ptrs inside,
    // freeing GPU VRAM.  validate()/initializeNodes() will repopulate it on wake-up.
    this->sidePacketMaps->clear();

    SET_IF_ENABLED(this->reporter->graphLoaded, 0);
    SPDLOG_LOGGER_INFO(modelmanager_logger,
        "Mediapipe graph {} idle-unloaded (freed GPU/CPU resources after {}s idle timeout)",
        getName(), mgconfig.getIdleUnloadTimeoutSeconds());
    return StatusCode::OK;
}

Status MediapipeGraphDefinition::wakeUpIfUnloaded(const ServableNameChecker& checker) {
    // Recursive: this holds lifecycleMtx and then calls reload(), which re-acquires it.
    std::lock_guard<std::recursive_mutex> lock(lifecycleMtx);
    // Double-check under lock: another thread may have already completed the reload.
    if (status.getStateCode() != PipelineDefinitionStateCode::UNLOADED) {
        return StatusCode::OK;
    }
    // Re-use the existing reload path:
    //   handle(ReloadEvent) -> fresh sidePacketMaps -> validate() -> initializeNodes()
    // The stored mgconfig holds all required configuration.
    SPDLOG_LOGGER_INFO(modelmanager_logger,
        "Mediapipe graph {} is UNLOADED; triggering lazy wake-up reload", getName());
    auto start = std::chrono::steady_clock::now();
    Status reloadStatus = reload(checker, this->mgconfig);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    if (reloadStatus.ok()) {
        // Only reset the idle timer on a successful wake; on failure leave it so
        // the existing failure-state handling applies and we don't mask the error.
        lastActivityTimeNs->store(
            std::chrono::steady_clock::now().time_since_epoch().count(),
            std::memory_order_relaxed);
        SPDLOG_LOGGER_INFO(modelmanager_logger,
            "Mediapipe graph {} wake-up completed in {}ms",
            getName(), elapsed.count());
    } else {
        // Wake-up reload failed (e.g. model files temporarily unavailable). reload()
        // ran validate() which left the state in LOADING_PRECONDITION_FAILED. Revert
        // to UNLOADED so the NEXT inference request retries the wake — making a
        // transient failure self-healing rather than permanently wedging a previously
        // healthy idle graph. We are still holding lifecycleMtx here.
        // (If validate() somehow ended elsewhere, UnloadEvent is a no-op on states
        // other than AVAILABLE/LOADING_PRECONDITION_FAILED, so this is safe.)
        this->status.handle(UnloadEvent());
        SPDLOG_LOGGER_ERROR(modelmanager_logger,
            "Mediapipe graph {} wake-up failed after {}ms: {}. Reverted to UNLOADED; "
            "next request will retry the wake.",
            getName(), elapsed.count(), reloadStatus.string());
    }
    return reloadStatus;
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
    } guard{*sidePacketMaps, success};

    auto& registry = NodeInitializerRegistry::instance();
    for (int i = 0; i < config.node().size(); i++) {
        for (const auto& initializer : registry.all()) {
            if (initializer->matches(config.node(i).calculator())) {
                Status status = initializer->initialize(config.node(i), getName(), mgconfig.getBasePath(), *sidePacketMaps, pythonBackend);
                if (!status.ok()) {
                    return status;
                }
            }
        }
    }
    // Register LoRA aliases for routing from initialized image gen pipelines
    this->loraAliases = sidePacketMaps->loraAliases;
    this->hideBaseModelInRouting = sidePacketMaps->hideBaseModelInRouting;
    success = true;
    return StatusCode::OK;
}
}  // namespace ovms

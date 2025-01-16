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
#pragma once
#include <iostream>
#include <map>
#include <memory>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../dags/pipelinedefinitionstatus.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../metric.hpp"
#include "../tensorinfo.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/thread_pool_executor.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "mediapipegraphconfig.hpp"
#include "packettypes.hpp"

namespace ovms {
class MediapipeGraphDefinitionUnloadGuard;
class MetricConfig;
class MetricRegistry;
class MediapipeServableMetricReporter;
class ModelManager;
class MediapipeGraphExecutor;
class Status;
class PythonBackend;
class PythonNodeResources;
class GenAiServable;
using PythonNodeResourcesMap = std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>>;
using GenAiServableMap = std::unordered_map<std::string, std::shared_ptr<GenAiServable>>;

extern std::shared_ptr<mediapipe::ThreadPoolExecutor> sharedThreadPool;
class MediapipeGraphDefinition {
    friend MediapipeGraphDefinitionUnloadGuard;

public:
    virtual ~MediapipeGraphDefinition();
    MediapipeGraphDefinition(const std::string name,
        const MediapipeGraphConfig& config = MGC,
        MetricRegistry* registry = nullptr,
        const MetricConfig* metricConfig = nullptr,
        PythonBackend* pythonBackend = nullptr);

    const std::string& getName() const { return name; }
    const PipelineDefinitionStatus& getStatus() const {
        return this->status;
    }

    const PipelineDefinitionStateCode getStateCode() const { return status.getStateCode(); }
    const model_version_t getVersion() const { return VERSION; }
    const tensor_map_t getInputsInfo() const;
    const tensor_map_t getOutputsInfo() const;
    const MediapipeGraphConfig& getMediapipeGraphConfig() const { return this->mgconfig; }
    MediapipeServableMetricReporter& getMetricReporter() const { return *this->reporter; }
    Status create(std::shared_ptr<MediapipeGraphExecutor>& pipeline);

    Status reload(ModelManager& manager, const MediapipeGraphConfig& config);
    Status validate(ModelManager& manager);
    void retire(ModelManager& manager);
    Status initializeNodes();
    bool isReloadRequired(const MediapipeGraphConfig& config) const;

    static constexpr uint64_t WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS = 500000;
    static const std::string SCHEDULER_CLASS_NAME;
    static const std::string PYTHON_NODE_CALCULATOR_NAME;
    static const std::string LLM_NODE_CALCULATOR_NAME;
    Status waitForLoaded(std::unique_ptr<MediapipeGraphDefinitionUnloadGuard>& unloadGuard, const uint32_t waitForLoadedTimeoutMicroseconds = WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS);

    // Pipelines are not versioned and any available definition has constant version equal 1.
    static constexpr model_version_t VERSION = 1;

protected:
    PythonNodeResourcesMap pythonNodeResourcesMap;
    GenAiServableMap genAiServableMap;

    struct ValidationResultNotifier {
        ValidationResultNotifier(PipelineDefinitionStatus& status, std::condition_variable& loadedNotify) :
            status(status),
            loadedNotify(loadedNotify) {
        }
        ~ValidationResultNotifier() {
            if (passed) {
                status.handle(ValidationPassedEvent());
                loadedNotify.notify_all();
            } else {
                status.handle(ValidationFailedEvent());
            }
        }
        bool passed = false;

    private:
        PipelineDefinitionStatus& status;
        std::condition_variable& loadedNotify;
    };

    virtual Status validateForConfigFileExistence();
    Status validateForConfigLoadableness();

    Status setStreamTypes();
    Status dryInitializeTest();
    std::string chosenConfig;
    static MediapipeGraphConfig MGC;
    const std::string name;

    bool passKfsRequestFlag;
    std::unordered_map<std::string, mediapipe_packet_type_enum> inputTypes;
    std::unordered_map<std::string, mediapipe_packet_type_enum> outputTypes;
    PipelineDefinitionStatus status;

    MediapipeGraphConfig mgconfig;
    ::mediapipe::CalculatorGraphConfig config;

    Status createInputsInfo();
    Status createOutputsInfo();
    Status createInputSidePacketsInfo();

    std::condition_variable loadedNotify;
    mutable std::shared_mutex metadataMtx;

private:
    void increaseRequestsHandlesCount() {
        ++requestsHandlesCounter;
    }

    void decreaseRequestsHandlesCount() {
        --requestsHandlesCounter;
    }

    tensor_map_t inputsInfo;
    tensor_map_t outputsInfo;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<std::string> inputSidePacketNames;

    std::atomic<uint64_t> requestsHandlesCounter = 0;

    PythonBackend* pythonBackend;

    std::unique_ptr<MediapipeServableMetricReporter> reporter;
};

class MediapipeGraphDefinitionUnloadGuard {
public:
    MediapipeGraphDefinitionUnloadGuard(MediapipeGraphDefinition& definition) :
        definition(definition) {
        definition.increaseRequestsHandlesCount();
    }

    ~MediapipeGraphDefinitionUnloadGuard() {
        definition.decreaseRequestsHandlesCount();
    }

private:
    MediapipeGraphDefinition& definition;
};
}  // namespace ovms

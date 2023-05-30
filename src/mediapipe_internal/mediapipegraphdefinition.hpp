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
#include <memory>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../dags/pipelinedefinitionstatus.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../metric.hpp"
#include "../tensorinfo.hpp"
#include "../timer.hpp"
#include "../version.hpp"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipegraphconfig.hpp"

namespace ovms {
class MetricConfig;
class MetricRegistry;
class ModelManager;
class MediapipeGraphExecutor;
class Status;

class MediapipeGraphDefinitionUnloadGuard;

class MediapipeGraphDefinition {
    friend MediapipeGraphDefinitionUnloadGuard;
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

public:
    MediapipeGraphDefinition(const std::string name,
        const MediapipeGraphConfig& config = MGC,
        MetricRegistry* registry = nullptr,
        const MetricConfig* metricConfig = nullptr);

    const std::string& getName() const { return name; }
    const PipelineDefinitionStatus& getStatus() const {
        return this->status;
    }
    const PipelineDefinitionStateCode getStateCode() const { return status.getStateCode(); }
    const model_version_t getVersion() const { return VERSION; }
    const tensor_map_t getInputsInfo() const;
    const tensor_map_t getOutputsInfo() const;

    Status create(std::shared_ptr<MediapipeGraphExecutor>& pipeline, const KFSRequest* request, KFSResponse* response);

    Status reload(ModelManager& manager, const MediapipeGraphConfig& config);
    Status validate(ModelManager& manager);
    void retire(ModelManager& manager);

    static constexpr uint64_t WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS = 500000;
    static const std::string SCHEDULER_CLASS_NAME;
    Status waitForLoaded(std::unique_ptr<MediapipeGraphDefinitionUnloadGuard>& unloadGuard, const uint waitForLoadedTimeoutMicroseconds = WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS);

    // Pipelines are not versioned and any available definition has constant version equal 1.
    static constexpr model_version_t VERSION = 1;

protected:
    Status validateForConfigFileExistence();
    Status validateForConfigLoadableness();

private:
    Status createInputsInfo();
    Status createOutputsInfo();

    void increaseRequestsHandlesCount() {
        ++requestsHandlesCounter;
    }

    void decreaseRequestsHandlesCount() {
        --requestsHandlesCounter;
    }

    static MediapipeGraphConfig MGC;
    const std::string name;

    std::string chosenConfig;  // TODO make const @atobiszei
    MediapipeGraphConfig mgconfig;
    ::mediapipe::CalculatorGraphConfig config;

    tensor_map_t inputsInfo;
    tensor_map_t outputsInfo;

    PipelineDefinitionStatus status;
    mutable std::shared_mutex metadataMtx;
    std::atomic<uint64_t> requestsHandlesCounter = 0;
    std::condition_variable loadedNotify;
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

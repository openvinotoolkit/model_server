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
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "..//kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../dags/pipelinedefinitionstatus.hpp"
#include "../execution_context.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../metric.hpp"
#include "../modelmanager.hpp"
#include "../status.hpp"
#include "../tensorinfo.hpp"
#include "../timer.hpp"
#include "../version.hpp"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipedemo.hpp"

namespace ovms {
class MediapipeGraphExecutor {
    // Pipelines are not versioned and any available definition has constant version equal 1.
    static constexpr model_version_t VERSION = 1;
    const std::string name;
    PipelineDefinitionStatus status;
    std::string chosenConfig;  // TODO make const @atobiszei
    ::mediapipe::CalculatorGraphConfig config;
    const uint16_t SERVABLE_VERSION = 1;
    tensor_map_t inputsInfo;
    tensor_map_t outputsInfo;

public:
    MediapipeGraphExecutor(const std::string name,
        MetricRegistry* registry = nullptr,
        const MetricConfig* metricConfig = nullptr);
    const std::string& getName() const { return name; }
    const PipelineDefinitionStatus& getStatus() const {
        return this->status;
    }
    const PipelineDefinitionStateCode getStateCode() const { return status.getStateCode(); }
    const model_version_t getVersion() const { return VERSION; }
    // METADATA
    const tensor_map_t getInputsInfo() const;
    const tensor_map_t getOutputsInfo() const;

    Status infer(const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) const;
    // TODO simultaneous infer & reload handling
private:
    Status createInputsInfo();
    Status createOutputsInfo();
};
}  // namespace ovms

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
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../metric.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"
#pragma GCC diagnostic pop
#include "packettypes.hpp"

namespace ovms {
class Status;
class PythonNodeResource;

class MediapipeGraphExecutor {
    const std::string name;
    const std::string version;
    const ::mediapipe::CalculatorGraphConfig config;
    stream_types_mapping_t inputTypes;
    stream_types_mapping_t outputTypes;
    const std::vector<std::string> inputNames;
    const std::vector<std::string> outputNames;

    std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>> pythonNodeResources;

    ::mediapipe::Timestamp currentStreamTimestamp;

    static Status deserializeTimestampIfAvailable(const KFSRequest& request, ::mediapipe::Timestamp& timestamp);
    Status partialDeserialize(std::shared_ptr<const ::inference::ModelInferRequest> request, ::mediapipe::CalculatorGraph& graph);
    Status validateSubsequentRequest(const ::inference::ModelInferRequest& request) const;

protected:
    Status serializePacket(const std::string& name, ::inference::ModelInferResponse& response, const ::mediapipe::Packet& packet) const;

public:
    static const std::string TIMESTAMP_PARAMETER_NAME;
    MediapipeGraphExecutor(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config,
        stream_types_mapping_t inputTypes,
        stream_types_mapping_t outputTypes,
        std::vector<std::string> inputNames, std::vector<std::string> outputNames,
        const std::unordered_map<std::string, std::shared_ptr<PythonNodeResource>>& pythonNodeResources);
    Status infer(const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) const;

    Status inferStream(const ::inference::ModelInferRequest& firstRequest, ::grpc::ServerReaderWriterInterface<::inference::ModelStreamInferResponse, ::inference::ModelInferRequest>& stream);
};
}  // namespace ovms

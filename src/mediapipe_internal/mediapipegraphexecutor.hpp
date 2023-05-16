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
#include "../kfs_frontend/kfs_utils.hpp"
#include "../metric.hpp"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/status.h"

namespace ovms {
class Status;

class MediapipeGraphExecutor {
    const std::string name;
    const std::string version;
    const ::mediapipe::CalculatorGraphConfig config;

public:
    MediapipeGraphExecutor(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config);
    Status infer(const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) const;
};
}  // namespace ovms

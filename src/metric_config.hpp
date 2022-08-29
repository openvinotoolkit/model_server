//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <rapidjson/document.h>

#include "layout_configuration.hpp"
#include "model_version_policy.hpp"
#include "shape.hpp"
#include "status.hpp"

namespace ovms {
/**
     * @brief This class represents metrics configuration
     */
class MetricConfig {
public:
    bool metricsEnabled;
    std::string endpointsPath;

    // Request Success/Fail
    // TFS
    bool requestSuccessGrpcPredict;
    bool requestSuccessGrpcGetModelMetadata;
    bool requestSuccessGrpcGetModelStatus;

    bool requestSuccessRestPredict;
    bool requestSuccessRestGetModelMetadata;
    bool requestSuccessRestGetModelStatus;

    bool requestFailGrpcPredict;
    bool requestFailGrpcGetModelMetadata;
    bool requestFailGrpcGetModelStatus;

    bool requestFailRestPredict;
    bool requestFailRestGetModelMetadata;
    bool requestFailRestGetModelStatus;

    // KFS
    bool requestSuccessGrpcModelInfer;
    bool requestSuccessGrpcModelMetadata;
    bool requestSuccessGrpcModelStatus;

    bool requestSuccessRestModelInfer;
    bool requestSuccessRestModelMetadata;
    bool requestSuccessRestModelStatus;

    bool requestFailGrpcModelInfer;
    bool requestFailGrpcModelMetadata;
    bool requestFailGrpcModelStatus;

    bool requestFailRestModelInfer;
    bool requestFailRestModelMetadata;
    bool requestFailRestModelStatus;

    Status parseMetricsArray(const rapidjson::Value& v);
    Status parseMetricsConfig(const rapidjson::Value& v);

    ModelConfig(){
        metricsEnabled = false;
        endpointsPath = "/metrics";

        requestSuccessGrpcPredict = false;
        requestSuccessGrpcGetModelMetadata= false;
        requestSuccessGrpcGetModelStatus= false;

        requestSuccessRestPredict= false;
        requestSuccessRestGetModelMetadata= false;
        requestSuccessRestGetModelStatus= false;

        requestFailGrpcPredict= false;
        requestFailGrpcGetModelMetadata= false;
        requestFailGrpcGetModelStatus= false;

        requestFailRestPredict= false;
        requestFailRestGetModelMetadata= false;
        requestFailRestGetModelStatus= false;

        // KFS
        requestSuccessGrpcModelInfer= false;
        requestSuccessGrpcModelMetadata= false;
        requestSuccessGrpcModelStatus= false;

        requestSuccessRestModelInfer= false;
        requestSuccessRestModelMetadata= false;
        requestSuccessRestModelStatus= false;

        requestFailGrpcModelInfer= false;
        requestFailGrpcModelMetadata= false;
        requestFailGrpcModelStatus= false;

        requestFailRestModelInfer= false;
        requestFailRestModelMetadata= false;
        requestFailRestModelStatus= false;
    }
    
};
}  // namespace ovms

//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include <string>

#include <rapidjson/document.h>

#include "config.hpp"
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
    bool requestSuccessGrpcModelReady;

    bool requestSuccessRestModelInfer;
    bool requestSuccessRestModelMetadata;
    bool requestSuccessRestModelReady;

    bool requestFailGrpcModelInfer;
    bool requestFailGrpcModelMetadata;
    bool requestFailGrpcModelReady;

    bool requestFailRestModelInfer;
    bool requestFailRestModelMetadata;
    bool requestFailRestModelReady;

    Status parseMetricsArray(const rapidjson::Value& v);
    Status parseMetricsConfig(const rapidjson::Value& v);
    bool validateEndpointPath(std::string endpoint);

    void setAllMetricsTo(bool enabled);
    Status loadSettings(Config& config);

    MetricConfig() {
        metricsEnabled = false;
        endpointsPath = "/metrics";

        setAllMetricsTo(metricsEnabled);
    }

    MetricConfig(bool enabled) {
        metricsEnabled = enabled;
        endpointsPath = "/metrics";

        setAllMetricsTo(metricsEnabled);
    }
};
}  // namespace ovms

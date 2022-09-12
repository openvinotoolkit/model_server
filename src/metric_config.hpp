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
#include <unordered_map>
#include <unordered_set>

#include <rapidjson/document.h>

#include "status.hpp"

namespace ovms {
/**
     * @brief This class represents metrics configuration
     */
class MetricConfig {
public:
    bool metricsEnabled;
    std::string endpointsPath;

    std::unordered_set<std::string> enabledFamiliesList;

    std::unordered_map<std::string, std::unordered_set<std::string>> additionalMetricFamilies = {
        {"ovms_streams", {"ovms_streams"}},
        {"ovms_infer_req_queue_size", {"ovms_infer_req_queue_size"}},
        {"ovms_infer_req_active", {"ovms_infer_req_active"}},
        {"ovms_current_requests", {"ovms_current_requests"}}};

    std::unordered_map<std::string, std::unordered_set<std::string>> defaultMetricFamilies = {
        {"ovms_requests_success",
            {"ovms_requests_success_grpc_predict",
                "ovms_requests_success_grpc_getmodelmetadata",
                "ovms_requests_success_grpc_getmodelstatus",
                "ovms_requests_success_grpc_modelinfer",
                "ovms_requests_success_grpc_modelmetadata",
                "ovms_requests_success_grpc_modelready",
                "ovms_requests_success_rest_modelinfer",
                "ovms_requests_success_rest_predict",
                "ovms_requests_success_rest_get_modelmetadata",
                "ovms_requests_success_rest_get_modelstatus",
                "ovms_requests_success_rest_modelmetadata",
                "ovms_requests_success_rest_modelready"}},
        {"ovms_requests_fail",
            {"ovms_requests_fail_grpc_predict",
                "ovms_requests_fail_grpc_getmodelmetadata",
                "ovms_requests_fail_grpc_getmodelstatus",
                "ovms_requests_fail_grpc_modelinfer",
                "ovms_requests_fail_grpc_modelmetadata",
                "ovms_requests_fail_grpc_modelready",
                "ovms_requests_fail_rest_modelinfer",
                "ovms_requests_fail_rest_predict",
                "ovms_requests_fail_rest_get_modelmetadata",
                "ovms_requests_fail_rest_get_modelstatus",
                "ovms_requests_fail_rest_modelmetadata",
                "ovms_requests_fail_rest_modelready"}},
        {"ovms_request_time_us", {"ovms_request_time_us_grpc", "ovms_request_time_us_rest"}},
        {"ovms_inference_time_us", {"ovms_inference_time_us"}},
        {"ovms_wait_for_infer_req_time_us", {"ovms_wait_for_infer_req_time_us"}}};

    Status parseMetricsArray(const rapidjson::Value& v);
    Status parseMetricsConfig(const rapidjson::Value& v);
    bool validateEndpointPath(std::string endpoint);

    void setDefaultMetricsTo(bool enabled);
    Status loadFromCLIString(bool isEnabled, const std::string& metricsList);

    MetricConfig() {
        metricsEnabled = false;
        endpointsPath = "/metrics";

        setDefaultMetricsTo(metricsEnabled);
    }

    MetricConfig(bool enabled) {
        metricsEnabled = enabled;
        endpointsPath = "/metrics";

        setDefaultMetricsTo(metricsEnabled);
    }
};
}  // namespace ovms

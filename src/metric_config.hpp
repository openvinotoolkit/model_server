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

const std::string METRIC_NAME_REQUESTS_SUCCESS = "ovms_requests_success";
const std::string METRIC_NAME_REQUESTS_FAIL = "ovms_requests_fail";

const std::string METRIC_NAME_STREAMS = "ovms_streams";
const std::string METRIC_NAME_INFER_REQ_QUEUE_SIZE = "ovms_infer_req_queue_size";

const std::string METRIC_NAME_INFER_REQ_ACTIVE = "ovms_infer_req_active";

const std::string METRIC_NAME_INFERENCE_TIME = "ovms_inference_time_us";
const std::string METRIC_NAME_CURRENT_REQUESTS = "ovms_current_requests";
const std::string METRIC_NAME_REQUEST_TIME = "ovms_request_time_us";
const std::string METRIC_NAME_WAIT_FOR_INFER_REQ_TIME = "ovms_wait_for_infer_req_time_us";

/**
     * @brief This class represents metrics configuration
     */
class MetricConfig {
public:
    bool metricsEnabled;
    std::string endpointsPath;

    Status parseMetricsConfig(const rapidjson::Value& v, bool forceFailureIfMetricsAreEnabled = false);
    bool isFamilyEnabled(const std::string& family) const;

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

protected:
    std::unordered_set<std::string> enabledFamiliesList;

private:
    Status parseMetricsArray(const rapidjson::Value& v);
    bool validateEndpointPath(const std::string& endpoint);

    std::unordered_set<std::string> additionalMetricFamilies = {
        {METRIC_NAME_INFER_REQ_QUEUE_SIZE},
        {METRIC_NAME_INFER_REQ_ACTIVE}};

    std::unordered_set<std::string> defaultMetricFamilies = {
        {METRIC_NAME_CURRENT_REQUESTS},
        {METRIC_NAME_REQUESTS_SUCCESS},
        {METRIC_NAME_REQUESTS_FAIL},
        {METRIC_NAME_REQUEST_TIME},
        {METRIC_NAME_STREAMS},
        {METRIC_NAME_INFERENCE_TIME},
        {METRIC_NAME_WAIT_FOR_INFER_REQ_TIME}};
};
}  // namespace ovms

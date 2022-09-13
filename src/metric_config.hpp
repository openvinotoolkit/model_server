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

    Status parseMetricsArray(const rapidjson::Value& v);
    Status parseMetricsConfig(const rapidjson::Value& v);
    bool validateEndpointPath(const std::string& endpoint);
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
    std::unordered_set<std::string> additionalMetricFamilies = {
        {"ovms_infer_req_queue_size"},
        {"ovms_infer_req_active"}};

    std::unordered_set<std::string> defaultMetricFamilies = {
        {"ovms_current_requests"},
        {"ovms_requests_success"},
        {"ovms_requests_fail"},
        {"ovms_request_time_us"},
        {"ovms_streams"},
        {"ovms_inference_time_us"},
        {"ovms_wait_for_infer_req_time_us"}};
};
}  // namespace ovms

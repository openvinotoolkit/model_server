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
#include "metric_config.hpp"

#include <regex>
#include <sstream>
#include <string>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)
#include <spdlog/spdlog.h>

#include "logging.hpp"
#include "schema.hpp"
#include "status.hpp"
#include "stringutils.hpp"

#ifndef __linux__
// Workaround : https://github.com/Tencent/rapidjson/issues/1448
#pragma push_macro("GetObject")
#undef GetObject
#endif
namespace ovms {

// Single Model / DAG
const std::string METRIC_NAME_REQUESTS_SUCCESS = "ovms_requests_success";
const std::string METRIC_NAME_REQUESTS_FAIL = "ovms_requests_fail";

const std::string METRIC_NAME_STREAMS = "ovms_streams";
const std::string METRIC_NAME_INFER_REQ_QUEUE_SIZE = "ovms_infer_req_queue_size";

const std::string METRIC_NAME_INFER_REQ_ACTIVE = "ovms_infer_req_active";

const std::string METRIC_NAME_INFERENCE_TIME = "ovms_inference_time_us";
const std::string METRIC_NAME_CURRENT_REQUESTS = "ovms_current_requests";
const std::string METRIC_NAME_REQUEST_TIME = "ovms_request_time_us";
const std::string METRIC_NAME_WAIT_FOR_INFER_REQ_TIME = "ovms_wait_for_infer_req_time_us";

// MediaPipe
const std::string METRIC_NAME_CURRENT_GRAPHS = "ovms_current_graphs";
const std::string METRIC_NAME_RESPONSES = "ovms_responses";

const std::string METRIC_NAME_REQUESTS_ACCEPTED = "ovms_requests_accepted";
const std::string METRIC_NAME_REQUESTS_REJECTED = "ovms_requests_rejected";

const std::string METRIC_NAME_GRAPH_ERROR = "ovms_graph_error";

bool MetricConfig::validateEndpointPath(const std::string& endpoint) {
    std::regex valid_endpoint_regex("^/[a-zA-Z0-9]*$");
    return std::regex_match(endpoint, valid_endpoint_regex);
}

// Getting the "monitoring" metrics config as input
Status MetricConfig::parseMetricsConfig(const rapidjson::Value& metrics, bool forceFailureIfMetricsAreEnabled) {
    Status status = StatusCode::OK;
    if (!metrics.HasMember("metrics"))
        return status;

    const auto& v = metrics["metrics"].GetObject();

    if (v.HasMember("enable")) {
        this->metricsEnabled = v["enable"].GetBool();
    } else {
        this->metricsEnabled = false;
    }

    if (metricsEnabled && forceFailureIfMetricsAreEnabled) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "CLI parameter rest_port is not defined. It must be set to enable metrics on the REST interface");
        return StatusCode::METRICS_REST_PORT_MISSING;
    }

    if (v.HasMember("endpoint_path")) {
        if (validateEndpointPath(v["endpoint_path"].GetString()))
            endpointsPath = v["endpoint_path"].GetString();
        else
            return StatusCode::INVALID_METRICS_ENDPOINT;
    } else {
        endpointsPath = "/metrics";
    }

    if (v.HasMember("metrics_list")) {
        status = parseMetricsArray(v["metrics_list"]);
    } else {
        setDefaultMetricsTo(this->metricsEnabled);
    }

    if (status == StatusCode::OK && this->metricsEnabled) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Metrics enabled.");

        std::stringstream ss;
        for (const auto& family : this->enabledFamiliesList) {
            ss << family << ", ";
        }
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Enabled metrics list: ", ss.str());
    }

    return status;
}

Status MetricConfig::parseMetricsArray(const rapidjson::Value& v) {
    for (auto& sh : v.GetArray()) {
        std::string metric = std::string(sh.GetString());

        const size_t listSize = this->enabledFamiliesList.size();

        for (const auto& family : this->defaultMetricFamilies) {
            if (metric == family) {
                this->enabledFamiliesList.insert(family);
            }
        }

        for (const auto& family : this->additionalMetricFamilies) {
            if (metric == family) {
                this->enabledFamiliesList.insert(family);
            }
        }

        if (this->enabledFamiliesList.size() == listSize) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Metrics family name not supported: {}", metric);
            return StatusCode::INVALID_METRICS_FAMILY_NAME;
        }
    }

    return StatusCode::OK;
}

bool MetricConfig::isFamilyEnabled(const std::string& family) const {
    return this->enabledFamiliesList.find(family) != this->enabledFamiliesList.end();
}

void MetricConfig::setDefaultMetricsTo(bool enabled) {
    this->enabledFamiliesList.clear();
    if (enabled) {
        for (const auto& family : this->defaultMetricFamilies) {
            this->enabledFamiliesList.insert(family);
        }
    }
}

Status MetricConfig::loadFromCLIString(bool isEnabled, const std::string& metricsList) {
    using namespace rapidjson;
    Document document;
    document.SetObject();
    Document::AllocatorType& allocator = document.GetAllocator();

    Value metrics(kObjectType);
    metrics.SetObject();
    metrics.AddMember("enable", isEnabled, allocator);

    // Create metrics array
    if (metricsList != "") {
        Value array(kArrayType);

        const char separator = ',';
        std::stringstream streamData(metricsList);
        std::string val;
        while (std::getline(streamData, val, separator)) {
            trim(val);
            Value metric(val.c_str(), allocator);
            array.PushBack(metric, allocator);
        }

        metrics.AddMember("metrics_list", array, allocator);
    }

    document.AddMember("metrics", metrics, allocator);

    return this->parseMetricsConfig(document.GetObject());
}

}  // namespace ovms

#ifndef __linux__
// Workaround : https://github.com/Tencent/rapidjson/issues/1448
#pragma pop_macro("GetObject")
#endif

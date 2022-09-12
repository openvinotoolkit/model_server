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
#include <string>

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <spdlog/spdlog.h>

#include "rapidjson/document.h"
#include "schema.hpp"
#include "stringutils.hpp"

namespace ovms {

bool MetricConfig::validateEndpointPath(std::string endpoint) {
    std::regex valid_endpoint_regex("^/[a-zA-Z0-9]*$");
    return std::regex_match(endpoint, valid_endpoint_regex);
}

// Getting the "monitoring" metrics config as input
Status MetricConfig::parseMetricsConfig(const rapidjson::Value& metrics) {
    Status status = StatusCode::OK;
    if (!metrics.HasMember("metrics"))
        return status;

    const auto& v = metrics["metrics"].GetObject();

    if (v.HasMember("enable")) {
        metricsEnabled = v["enable"].GetBool();
    } else {
        metricsEnabled = false;
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
        setAllMetricsTo(metricsEnabled);
    }

    return status;
}

Status MetricConfig::parseMetricsArray(const rapidjson::Value& v) {
    for (auto& sh : v.GetArray()) {
        std::string metric = std::string(sh.GetString());
        if (metric == "request_success_grpc_predict") {
            requestSuccessGrpcPredict = true;
        }
        if (metric == "request_success_grpc_get_model_metadata") {
            requestSuccessGrpcGetModelMetadata = true;
        }
        if (metric == "request_success_grpc_get_model_status") {
            requestSuccessGrpcGetModelStatus = true;
        }
        if (metric == "request_success_rest_predict") {
            requestSuccessRestPredict = true;
        }
        if (metric == "request_success_rest_get_model_metadata") {
            requestSuccessRestGetModelMetadata = true;
        }
        if (metric == "request_success_rest_get_model_status") {
            requestSuccessRestGetModelStatus = true;
        }
        if (metric == "request_fail_grpc_predict") {
            requestFailGrpcPredict = true;
        }
        if (metric == "request_fail_grpc_get_model_metadata") {
            requestFailGrpcGetModelMetadata = true;
        }
        if (metric == "request_fail_grpc_get_model_status") {
            requestFailGrpcGetModelStatus = true;
        }
        if (metric == "request_fail_rest_predict") {
            requestFailRestPredict = true;
        }
        if (metric == "request_fail_rest_get_model_metadata") {
            requestFailRestGetModelMetadata = true;
        }
        if (metric == "request_fail_rest_get_model_status") {
            requestFailRestGetModelStatus = true;
        }
        if (metric == "request_success_grpc_model_infer") {
            requestSuccessGrpcModelInfer = true;
        }
        // KFS
        if (metric == "request_success_grpc_model_metadata") {
            requestSuccessGrpcModelMetadata = true;
        }
        if (metric == "request_success_grpc_model_ready") {
            requestSuccessGrpcModelReady = true;
        }
        if (metric == "request_success_rest_model_infer") {
            requestSuccessRestModelInfer = true;
        }
        if (metric == "request_success_rest_model_metadata") {
            requestSuccessRestModelMetadata = true;
        }
        if (metric == "request_success_rest_model_ready") {
            requestSuccessRestModelReady = true;
        }
        if (metric == "request_fail_grpc_model_infer") {
            requestFailGrpcModelInfer = true;
        }
        if (metric == "request_fail_grpc_model_metadata") {
            requestFailGrpcModelMetadata = true;
        }
        if (metric == "request_fail_grpc_model_ready") {
            requestFailGrpcModelReady = true;
        }

        if (metric == "request_fail_rest_model_infer") {
            requestFailRestModelInfer = true;
        }
        if (metric == "request_fail_rest_model_metadata") {
            requestFailRestModelMetadata = true;
        }
        if (metric == "request_fail_rest_model_ready") {
            requestFailRestModelReady = true;
        }
        if (metric == "request_time_grpc") {
            requestTimeGrpc = true;
        }
        if (metric == "request_time_rest") {
            requestTimeRest = true;
        }
        if (metric == "inference_time") {
            inferenceTime = true;
        }
        if (metric == "wait_for_infer_req_time") {
            waitForInferReqTime = true;
        }
        if (metric == "streams") {
            inferenceTime = true;
        }
        if (metric == "infer_req_queue_size") {
            inferReqQueueSize = true;
        }
        if (metric == "infer_req_active") {
            inferReqActive = true;
        }
        if (metric == "current_requests") {
            currentRequests = true;
        }
    }

    return StatusCode::OK;
}

void MetricConfig::setAllMetricsTo(bool enabled) {
    requestSuccessGrpcPredict = enabled;
    requestSuccessGrpcGetModelMetadata = enabled;
    requestSuccessGrpcGetModelStatus = enabled;

    requestSuccessRestPredict = enabled;
    requestSuccessRestGetModelMetadata = enabled;
    requestSuccessRestGetModelStatus = enabled;

    requestFailGrpcPredict = enabled;
    requestFailGrpcGetModelMetadata = enabled;
    requestFailGrpcGetModelStatus = enabled;

    requestFailRestPredict = enabled;
    requestFailRestGetModelMetadata = enabled;
    requestFailRestGetModelStatus = enabled;

    // KFS
    requestSuccessGrpcModelInfer = enabled;
    requestSuccessGrpcModelMetadata = enabled;
    requestSuccessGrpcModelReady = enabled;

    requestSuccessRestModelInfer = enabled;
    requestSuccessRestModelMetadata = enabled;
    requestSuccessRestModelReady = enabled;

    requestFailGrpcModelInfer = enabled;
    requestFailGrpcModelMetadata = enabled;
    requestFailGrpcModelReady = enabled;

    requestFailRestModelInfer = enabled;
    requestFailRestModelMetadata = enabled;
    requestFailRestModelReady = enabled;

    requestTimeGrpc = enabled;
    requestTimeRest = enabled;

    inferenceTime = enabled;
    waitForInferReqTime = enabled;

    streams = enabled;

    inferReqQueueSize = enabled;
    inferReqActive = enabled;

    currentRequests = enabled;
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

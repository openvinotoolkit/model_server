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
#include "metric_config.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <spdlog/spdlog.h>

#include "logging.hpp"
#include "schema.hpp"
#include "stringutils.hpp"

namespace ovms {

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
        endpointsPath = v["endpoint_path"].GetString();
    } else {
        endpointsPath = "/metrics";
    }

    if (v.HasMember("metrics_list")) {
        status = parseMetricsArray(v["metrics_list"]);
    } else {
        // Log missing metrics list or list empty
        // or enable all metrics ?
    }

    return status;
}

Status MetricConfig::parseMetricsArray(const rapidjson::Value& v) {
    for (auto& sh : v.GetArray()) {
        std::string metric = std::string(sh.GetString());
        if (metric == "requestSuccessGrpcPredict") {
            requestSuccessGrpcPredict = true;
        }
        if (metric == "requestSuccessGrpcGetModelMetadata") {
            requestSuccessGrpcGetModelMetadata = true;
        }
        if (metric == "requestSuccessGrpcGetModelStatus") {
            requestSuccessGrpcGetModelStatus = true;
        }
        if (metric == "requestSuccessRestPredict") {
            requestSuccessRestPredict = true;
        }
        if (metric == "requestSuccessRestGetModelMetadata") {
            requestSuccessRestGetModelMetadata = true;
        }
        if (metric == "requestSuccessRestGetModelStatus") {
            requestSuccessRestGetModelStatus = true;
        }
        if (metric == "requestFailGrpcPredict") {
            requestFailGrpcPredict = true;
        }
        if (metric == "requestFailGrpcGetModelMetadata") {
            requestFailGrpcGetModelMetadata = true;
        }
        if (metric == "requestFailGrpcGetModelStatus") {
            requestFailGrpcGetModelStatus = true;
        }
        if (metric == "requestFailRestPredict") {
            requestFailRestPredict = true;
        }
        if (metric == "requestFailRestGetModelMetadata") {
            requestFailRestGetModelMetadata = true;
        }
        if (metric == "requestFailRestGetModelStatus") {
            requestFailRestGetModelStatus = true;
        }
        if (metric == "requestSuccessGrpcModelInfer") {
            requestSuccessGrpcModelInfer = true;
        }
        // KFS
        if (metric == "requestSuccessGrpcModelMetadata") {
            requestSuccessGrpcModelMetadata = true;
        }
        if (metric == "requestSuccessGrpcModelStatus") {
            requestSuccessGrpcModelStatus = true;
        }
        if (metric == "requestSuccessRestModelInfer") {
            requestSuccessRestModelInfer = true;
        }
        if (metric == "requestSuccessRestModelMetadata") {
            requestSuccessRestModelMetadata = true;
        }
        if (metric == "requestSuccessRestModelStatus") {
            requestSuccessRestModelStatus = true;
        }
        if (metric == "requestFailGrpcModelInfer") {
            requestFailGrpcModelInfer = true;
        }
        if (metric == "requestFailGrpcModelMetadata") {
            requestFailGrpcModelMetadata = true;
        }
        if (metric == "requestFailGrpcModelStatus") {
            requestFailGrpcModelStatus = true;
        }

        if (metric == "requestFailRestModelInfer") {
            requestFailRestModelInfer = true;
        }
        if (metric == "requestFailRestModelMetadata") {
            requestFailRestModelMetadata = true;
        }
        if (metric == "requestFailRestModelStatus") {
            requestFailRestModelStatus = true;
        }
    }

    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "requestFailRestModelStatus {}", requestFailRestModelStatus);

    return StatusCode::OK;
}

}  // namespace ovms

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
#include "metrics_config.hpp"

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

// Getting the "monitoring" config as input
Status MetricConfig::parseMetricsConfig(const rapidjson::Value& v) {
    if (v.HasMember("enable")) {
        // if (v.HasMember("enable").isBool())
        metricsEnabled = v["enable"].GetBool();
    } else {
        metricsEnabled = false;
    }

    if (v.HasMember("endpoint_path")) {
        endpointsPath = v["enable"].GetString();
    } else {
        endpointsPath = "/metrics";
    }

    if (v.HasMember("metrics_list")) {
        status = parseMetricsArray(v["metrics_list"]);
    } else {
        // Log missing metrics list or list empty
        // or enable all metrics ?
    }

    return StatusCode::OK;
}

Status MetricConfig::parseMetricsArray(const rapidjson::Value& v) {
    for (auto& sh : v.value.GetArray()) {
        if (sh.GetString() == "requestSuccessGrpcPredict") {
            requestSuccessGrpcPredict = true;
        }
        if (sh.GetString() == "requestSuccessGrpcGetModelMetadata") {
            requestSuccessGrpcGetModelMetadata = true;
        }
        if (sh.GetString() == "requestSuccessGrpcGetModelStatus") {
            requestSuccessGrpcGetModelStatus = true;
        }
        if (sh.GetString() == "requestSuccessRestPredict") {
            requestSuccessRestPredict = true;
        }
        if (sh.GetString() == "requestSuccessRestGetModelMetadata") {
            requestSuccessRestGetModelMetadata = true;
        }
        if (sh.GetString() == "requestSuccessRestGetModelStatus") {
            requestSuccessRestGetModelStatus = true;
        }
        if (sh.GetString() == "requestFailGrpcPredict") {
            requestFailGrpcPredict = true;
        }
        if (sh.GetString() == "requestFailGrpcGetModelMetadata") {
            requestFailGrpcGetModelMetadata = true;
        }
        if (sh.GetString() == "requestFailGrpcGetModelStatus") {
            requestFailGrpcGetModelStatus = true;
        }
        if (sh.GetString() == "requestFailRestPredict") {
            requestFailRestPredict = true;
        }
        if (sh.GetString() == "requestFailRestGetModelMetadata") {
            requestFailRestGetModelMetadata = true;
        }
        if (sh.GetString() == "requestFailRestGetModelStatus") {
            requestFailRestGetModelStatus = true;
        }
        if (sh.GetString() == "requestSuccessGrpcModelInfer") {
            requestSuccessGrpcModelInfer = true;
        }
        // KFS
        if (sh.GetString() == "requestSuccessGrpcModelMetadata") {
            requestSuccessGrpcModelMetadata = true;
        }
        if (sh.GetString() == "requestSuccessGrpcModelStatus") {
            requestSuccessGrpcModelStatus = true;
        }
        if (sh.GetString() == "requestSuccessRestModelInfer") {
            requestSuccessRestModelInfer = true;
        }
        if (sh.GetString() == "requestSuccessRestModelMetadata") {
            requestSuccessRestModelMetadata = true;
        }
        if (sh.GetString() == "requestSuccessRestModelStatus") {
            requestSuccessRestModelStatus = true;
        }
        if (sh.GetString() == "requestFailGrpcModelInfer") {
            requestFailGrpcModelInfer = true;
        }
        if (sh.GetString() == "requestFailGrpcModelMetadata") {
            requestFailGrpcModelMetadata = true;
        }
        if (sh.GetString() == "requestFailGrpcModelStatus") {
            requestFailGrpcModelStatus = true;
        }

        if (sh.GetString() == "requestFailRestModelInfer") {
            requestFailRestModelInfer = true;
        }
        if (sh.GetString() == "requestFailRestModelMetadata") {
            requestFailRestModelMetadata = true;
        }
        if (sh.GetString() == "requestFailRestModelStatus") {
            requestFailRestModelStatus = true;
        }
    }
    return StatusCode::OK;
}

}  // namespace ovms

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

#include "logging.hpp"
#include "rapidjson/document.h"
#include "schema.hpp"
#include "stringutils.hpp"

namespace ovms {

bool MetricConfig::validateEndpointPath(const std::string& endpoint) {
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
        setDefaultMetricsTo(metricsEnabled);
    }

    return status;
}

Status MetricConfig::parseMetricsArray(const rapidjson::Value& v) {
    for (auto& sh : v.GetArray()) {
        std::string metric = std::string(sh.GetString());

        const size_t listSize = this->enabledFamiliesList.size();

        for (const auto& [family, metrics] : this->defaultMetricFamilies) {
            if (metric == family) {
                this->enabledFamiliesList.insert(family);
            }
        }

        for (const auto& [family, metrics] : this->additionalMetricFamilies) {
            if (metric == family) {
                this->enabledFamiliesList.insert(family);
            }
        }

        if (listSize == this->enabledFamiliesList.size()) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Metrics family name not supported: {}", metric);
            return StatusCode::CONFIG_FILE_INVALID;
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
        for (const auto& [family, metrics] : this->defaultMetricFamilies) {
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

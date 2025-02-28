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

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include "logging.hpp"
#include "shape.hpp"
#include "stringutils.hpp"

using plugin_config_t = std::map<std::string, ov::Any>;

namespace ovms {
class Status;
class TensorInfo;

Status createSharedTensor(ov::Tensor& destinationTensor, ov::element::Type_t precision, const ov::Shape& shape);
ov::Tensor createTensorWithNoDataOwnership(ov::element::Type_t precision, const shape_t& shape, void* data);

std::string getTensorMapString(const std::map<std::string, std::shared_ptr<const TensorInfo>>& tensorMap);

Status tensorClone(ov::Tensor& destinationTensor, const ov::Tensor& sourceTensor);

std::optional<ov::Layout> getLayoutFromRTMap(const ov::RTMap& rtMap);

Status validatePluginConfiguration(const plugin_config_t& pluginConfig, const std::string& targetDevice, const ov::Core& ieCore);

// Logging
// #1 model/global plugin  CompiledMode:DUMMY / Global OpenVINO plugin:CPU
// #2 version/_
// #3 target_device/_
// {} {}
template <typename PropertyExtractor>
static void logOVPluginConfig(PropertyExtractor&& propertyExtractor, const std::string& loggingAuthor, const std::string& loggingDetails) {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Logging {}; {}plugin configuration", loggingAuthor, loggingDetails);
    auto supportedPropertiesKey = std::string("SUPPORTED_PROPERTIES");  // ov::supported_properties;
    std::vector<ov::PropertyName> supportedConfigKeys;
    try {
        OV_LOGGER("ov::Any::operator=()");
        ov::Any value = propertyExtractor(supportedPropertiesKey);
        OV_LOGGER("ov::Any::as<std::vector<ov::PropertyName>>()");
        supportedConfigKeys = value.as<std::vector<ov::PropertyName>>();
    } catch (std::exception& e) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from OpenVINO when requesting {};{} config key: {}; Error: {}", loggingAuthor, loggingDetails, supportedPropertiesKey, e.what());
        return;
    } catch (...) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from OpenVINO when requesting {};{} config key: {};", loggingAuthor, loggingDetails, supportedPropertiesKey);
        return;
    }
    std::vector<std::string> pluginConfigNameValues;
    for (auto& key : supportedConfigKeys) {
        if (key == "SUPPORTED_PROPERTIES")
            continue;
        std::string value;
        try {
            ov::Any paramValue = propertyExtractor(key);
            OV_LOGGER("key: {}; ov::Any::as<std::string>()", key);
            value = paramValue.as<std::string>();
        } catch (std::exception& e) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from OpenVINO when requesting {};{} config key: {}; Error: {}", loggingAuthor, loggingDetails, key, e.what());
            continue;
        } catch (...) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from OpenVINO when requesting {};{} config key: {};", loggingAuthor, loggingDetails, key);
            continue;
        }
        pluginConfigNameValues.emplace_back(joins({key, value}, ": "));
    }
    std::sort(pluginConfigNameValues.begin(), pluginConfigNameValues.end());
    std::string pluginConfigNameValuesString = joins(pluginConfigNameValues, ", ");
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "{}; {}plugin configuration: {{ {} }}", loggingAuthor, loggingDetails, pluginConfigNameValuesString);
}
}  // namespace ovms

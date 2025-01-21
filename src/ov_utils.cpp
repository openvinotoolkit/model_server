//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include "ov_utils.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

#include "logging.hpp"
#include "profiler.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

ov::Tensor createTensorWithNoDataOwnership(ov::element::Type_t precision, const shape_t& shape, void* data) {
    OV_LOGGER("ov::Tensor(precision, shape, data)");
    auto tensor = ov::Tensor(precision, shape, data);
    return tensor;
}

Status createSharedTensor(ov::Tensor& destinationTensor, ov::element::Type_t precision, const ov::Shape& shape) {
    OV_LOGGER("ov::Tensor(precision, shape)");
    destinationTensor = ov::Tensor(precision, shape);
    return StatusCode::OK;
}

std::string getTensorMapString(const std::map<std::string, std::shared_ptr<const TensorInfo>>& inputsInfo) {
    std::stringstream stringStream;
    for (const auto& pair : inputsInfo) {
        const auto& name = pair.first;
        auto inputInfo = pair.second;
        const auto precision = inputInfo->getPrecision();
        const auto& layout = inputInfo->getLayout();
        const auto& shape = inputInfo->getShape();

        stringStream << "\nname: " << name
                     << "; mapping: " << inputInfo->getMappedName()
                     << "; shape: " << shape.toString()
                     << "; precision: " << TensorInfo::getPrecisionAsString(precision)
                     << "; layout: " << TensorInfo::getStringFromLayout(layout);
    }
    return stringStream.str();
}

Status tensorClone(ov::Tensor& destinationTensor, const ov::Tensor& sourceTensor) {
    OVMS_PROFILE_FUNCTION();
    if (sourceTensor.get_element_type() == ov::element::Type_t::string) {
        destinationTensor = ov::Tensor(sourceTensor.get_element_type(), sourceTensor.get_shape());
        std::string* srcData = sourceTensor.data<std::string>();
        std::string* destData = destinationTensor.data<std::string>();
        for (size_t i = 0; i < sourceTensor.get_shape()[0]; i++) {
            destData[i].assign(srcData[i]);
        }
        return StatusCode::OK;
    }
    OV_LOGGER("ov::Tensor(ov::element::type, shape)");
    destinationTensor = ov::Tensor(sourceTensor.get_element_type(), sourceTensor.get_shape());
    if (destinationTensor.get_byte_size() != sourceTensor.get_byte_size()) {
        SPDLOG_ERROR("tensorClone byte size mismatch destination:{}; source:{}",
            destinationTensor.get_byte_size(),
            sourceTensor.get_byte_size());
        return StatusCode::OV_CLONE_TENSOR_ERROR;
    }
    std::memcpy(destinationTensor.data(), sourceTensor.data(), sourceTensor.get_byte_size());
    return StatusCode::OK;
}

std::optional<ov::Layout> getLayoutFromRTMap(const ov::RTMap& rtMap) {
    OV_LOGGER("const auto& [k, v] : ov::RTMap& rtMap");
    for (const auto& [k, v] : rtMap) {
        try {
            OV_LOGGER("v.as<ov::LayoutAttribute>().value");
            return v.as<ov::LayoutAttribute>().value;
        } catch (ov::Exception& e) {
        }
    }
    return std::nullopt;
}

static void insertSupportedKeys(std::set<std::string>& aggregatedPluginSupportedConfigKeys, const std::string& pluginName, const ov::Core& ieCore) {
    auto prop = ov::supported_properties;
    try {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Validating plugin: {}; configuration", pluginName);
        OV_LOGGER("ov::Core: {}, ieCore->get_property({}, ov::supported_properties)", reinterpret_cast<const void*>(&ieCore), pluginName);
        std::vector<ov::PropertyName> pluginSupportedConfigKeys = ieCore.get_property(pluginName, prop);
        std::set<std::string> pluginSupportedConfigKeysSet(pluginSupportedConfigKeys.begin(), pluginSupportedConfigKeys.end());
        aggregatedPluginSupportedConfigKeys.insert(pluginSupportedConfigKeys.begin(), pluginSupportedConfigKeys.end());
    } catch (std::exception& e) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Exception thrown from IE when requesting plugin: {}; key: {}; value. Error: {}", pluginName, prop.name(), e.what());
    } catch (...) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Exception thrown from IE when requesting plugin: {}; key: {}; value.", pluginName, prop.name());
    }
}

Status validatePluginConfiguration(const plugin_config_t& pluginConfig, const std::string& targetDevice, const ov::Core& ieCore) {
    std::set<std::string> pluginSupportedConfigKeys;
    std::string pluginDelimiter = ":";
    auto pluginDelimeterPos = targetDevice.find(pluginDelimiter);
    if (pluginDelimeterPos != std::string::npos) {
        std::string pluginName = targetDevice.substr(0, pluginDelimeterPos);
        insertSupportedKeys(pluginSupportedConfigKeys, pluginName, ieCore);
        char deviceDelimiter = ',';
        std::stringstream ss(targetDevice.substr(pluginDelimeterPos + 1, targetDevice.length()));
        std::string deviceName;

        while (getline(ss, deviceName, deviceDelimiter)) {
            insertSupportedKeys(pluginSupportedConfigKeys, deviceName, ieCore);
        }
    } else {
        insertSupportedKeys(pluginSupportedConfigKeys, targetDevice, ieCore);
    }

    for (auto& config : pluginConfig) {
        if (std::find(pluginSupportedConfigKeys.begin(), pluginSupportedConfigKeys.end(), config.first) == pluginSupportedConfigKeys.end()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Plugin config key: {} not found in supported config keys for device: {}.", config.first, targetDevice);
            SPDLOG_LOGGER_INFO(modelmanager_logger, "List of supported keys for this device:");
            for (std::string supportedKey : pluginSupportedConfigKeys) {
                SPDLOG_LOGGER_INFO(modelmanager_logger, "{}", supportedKey);
            }
            return StatusCode::MODEL_CONFIG_INVALID;
        }
    }

    return StatusCode::OK;
}
}  // namespace ovms

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

#include "plugin_configuration.hpp"

#include "logging.hpp"

namespace ovms {
Status validatePluginConfiguration(const plugin_config_t& pluginConfig, const std::string& targetDevice, const ov::Core& ieCore) {
    auto availableDevices = ieCore.get_available_devices();
    auto availablePlugins = availableDevices;
    const std::string supportedConfigKey = METRIC_KEY(SUPPORTED_CONFIG_KEYS);
    for (const auto& plugin : availablePlugins) {
        std::vector<std::string> supportedConfigKeys;
        try {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Validating plugin: {}; configuration", plugin);
            std::vector<std::string> supportedConfigKeys2 = ieCore.get_property(plugin, supportedConfigKey).as<std::vector<std::string>>();
            supportedConfigKeys = std::move(supportedConfigKeys2);
        } catch (std::exception& e) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting plugin: {}; key: {}; value. Error: {}", plugin, supportedConfigKey, e.what());
        } catch (...) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting plugin: {}; key: {}; value.", plugin, supportedConfigKey);
        }
        if(plugin == targetDevice){
            for (auto& config : pluginConfig) {
                if (std::find(supportedConfigKeys.begin(), supportedConfigKeys.end(), config.first) == supportedConfigKeys.end())
                {
                    SPDLOG_LOGGER_ERROR(modelmanager_logger, "Plugin config key: {} not found in supported config keys for {} device", config.first, plugin);
                    return StatusCode::MODEL_CONFIG_INVALID;
                }
            }
        }
    }
    return StatusCode::OK;
}
}  // namespace ovms
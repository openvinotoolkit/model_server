//****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "json_parser.hpp"

#include <algorithm>
#include <map>
#include <string>

#include <openvino/openvino.hpp>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#pragma warning(pop)

#include "logging.hpp"
#include "status.hpp"

namespace ovms {

using plugin_config_t = std::map<std::string, ov::Any>;

bool validType(const rapidjson::Value::ConstMemberIterator& node) {
    return (node->value.IsString() || node->value.IsBool() || node->value.IsInt64() || node->value.IsDouble());
}

// OpenVINO's strict plugin_config typing (e.g. intel_gpu) requires NUM_STREAMS to be
// either an ov::streams::Num value or a string the plugin can parse via operator>>
// ("AUTO", "NUMA", or an integer literal). An ov::Any holding int64_t/double has no
// conversion to ov::streams::Num and gets rejected as "Invalid value: N for property:
// NUM_STREAMS". Keep NUM_STREAMS as a string regardless of how the JSON expressed it.
bool isStringOnlyPluginKey(const std::string& key) {
    return key == "NUM_STREAMS";
}

std::string numericValueToString(const rapidjson::Value& v) {
    if (v.IsInt64()) {
        return std::to_string(v.GetInt64());
    }
    if (v.IsDouble()) {
        return std::to_string(v.GetDouble());
    }
    if (v.IsBool()) {
        return v.GetBool() ? "true" : "false";
    }
    return "";
}

std::string normalizeWindowsPathSeparators(const std::string& value) {
#ifdef _WIN32
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
#else
    return value;
#endif
}

bool isPathLikePluginKey(const std::string& key) {
    return key == "CACHE_DIR";
}

std::string maybeNormalizePathPluginValue(const std::string& key, const std::string& value) {
    if (!isPathLikePluginKey(key)) {
        return value;
    }
    return normalizeWindowsPathSeparators(value);
}

/**
* @brief Parses json node for plugin config keys and values
* 
* @param pluginConfig output representing plugin_config_t map
* 
* @param json node representing plugin_config
*
* @return status
*/
Status JsonParser::parsePluginConfig(const rapidjson::Value& node, plugin_config_t& pluginConfig) {
    if (!node.IsObject()) {
        return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
    }

    for (auto it = node.MemberBegin(); it != node.MemberEnd(); ++it) {
        if (it->value.IsObject() && it->name.GetString() == std::string("DEVICE_PROPERTIES")) {
            auto devicesProperties = ov::AnyMap{};
            for (auto propertiesIt = it->value.GetObject().MemberBegin(); propertiesIt != it->value.GetObject().MemberEnd(); ++propertiesIt) {
                auto properties = ov::AnyMap{};
                if (propertiesIt->value.IsObject()) {
                    auto deviceProperties = propertiesIt->value.GetObject();
                    for (auto propertyIt = deviceProperties.MemberBegin(); propertyIt != deviceProperties.MemberEnd(); ++propertyIt) {
                        if (!validType(propertyIt)) {
                            return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
                        }
                        const std::string propertyKey = propertyIt->name.GetString();
                        if (isStringOnlyPluginKey(propertyKey) && !propertyIt->value.IsString()) {
                            properties[propertyKey] = numericValueToString(propertyIt->value);
                            continue;
                        }
                        if (propertyIt->value.IsString()) {
                            const std::string propertyKey = propertyIt->name.GetString();
                            properties[propertyKey] = maybeNormalizePathPluginValue(propertyKey, propertyIt->value.GetString());
                        }
                        if (propertyIt->value.IsInt64()) {
                            properties[propertyIt->name.GetString()] = propertyIt->value.GetInt64();
                        }
                        if (propertyIt->value.IsDouble()) {
                            properties[propertyIt->name.GetString()] = propertyIt->value.GetDouble();
                        }
                        if (propertyIt->value.IsBool()) {
                            properties[propertyIt->name.GetString()] = propertyIt->value.GetBool();
                        }
                    }
                } else {
                    return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
                }
                devicesProperties[propertiesIt->name.GetString()] = properties;
            }
            pluginConfig[it->name.GetString()] = devicesProperties;
            continue;
        }
        if (!validType(it)) {
            return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
        }
        const std::string topKey = it->name.GetString();
        if (isStringOnlyPluginKey(topKey) && !it->value.IsString()) {
            pluginConfig[topKey] = numericValueToString(it->value);
            continue;
        }
        if (it->value.IsString()) {
            const std::string topKey = it->name.GetString();
            if (((it->name.GetString() == std::string("CPU_THROUGHPUT_STREAMS")) && (it->value.GetString() == std::string("CPU_THROUGHPUT_AUTO"))) || ((it->name.GetString() == std::string("GPU_THROUGHPUT_STREAMS")) && (it->value.GetString() == std::string("GPU_THROUGHPUT_AUTO")))) {
                pluginConfig["PERFORMANCE_HINT"] = "THROUGHPUT";
                SPDLOG_WARN("{} plugin config key is deprecated. Use PERFORMANCE_HINT instead", it->name.GetString());
            } else {
                if ((it->name.GetString() == std::string("CPU_THROUGHPUT_STREAMS")) || (it->name.GetString() == std::string("GPU_THROUGHPUT_STREAMS"))) {
                    pluginConfig["NUM_STREAMS"] = it->value.GetString();
                    SPDLOG_WARN("{} plugin config key is deprecated. Use NUM_STREAMS instead", it->name.GetString());
                } else if (it->name.GetString() == std::string("CPU_THREADS_NUM")) {
                    pluginConfig["INFERENCE_NUM_THREADS"] = it->value.GetString();
                    SPDLOG_WARN("{} plugin config key is deprecated. Use INFERENCE_NUM_THREADS instead", it->name.GetString());
                } else {
                    pluginConfig[topKey] = maybeNormalizePathPluginValue(topKey, it->value.GetString());
                }
            }
        }
        if (it->value.IsInt64()) {
            if (it->name.GetString() == std::string("CPU_THROUGHPUT_STREAMS") || it->name.GetString() == std::string("GPU_THROUGHPUT_STREAMS")) {
                pluginConfig["NUM_STREAMS"] = std::to_string(it->value.GetInt64());
                SPDLOG_WARN("{} plugin config key is deprecated. Use  NUM_STREAMS instead", it->name.GetString());
            } else {
                pluginConfig[it->name.GetString()] = it->value.GetInt64();
            }
        }
        if (it->value.IsDouble()) {
            if (it->name.GetString() == std::string("CPU_THROUGHPUT_STREAMS") || it->name.GetString() == std::string("GPU_THROUGHPUT_STREAMS")) {
                pluginConfig["NUM_STREAMS"] = std::to_string(it->value.GetDouble());
                SPDLOG_WARN("{} plugin config key is deprecated. Use  NUM_STREAMS instead", it->name.GetString());
            } else {
                pluginConfig[it->name.GetString()] = it->value.GetDouble();
            }
        }
        if (it->value.IsBool()) {
            pluginConfig[it->name.GetString()] = bool(it->value.GetBool());
        }
    }

    return StatusCode::OK;
}

/**
* @brief Parses string for plugin config keys and values
* 
* @param string representing plugin_config
* 
* @return status
*/
Status JsonParser::parsePluginConfig(std::string command, plugin_config_t& pluginConfig) {
    rapidjson::Document node;
    if (command.empty()) {
        return StatusCode::OK;
    }
    if (node.Parse(command.c_str()).HasParseError()) {
        return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
    }

    return parsePluginConfig(node, pluginConfig);
}

}  // namespace ovms

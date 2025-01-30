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
        if (it->value.IsString()) {
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
                    pluginConfig[it->name.GetString()] = it->value.GetString();
                }
            }

        } else if (it->value.IsInt64()) {
            if (it->name.GetString() == std::string("CPU_THROUGHPUT_STREAMS") || it->name.GetString() == std::string("GPU_THROUGHPUT_STREAMS")) {
                pluginConfig["NUM_STREAMS"] = std::to_string(it->value.GetInt64());
                SPDLOG_WARN("{} plugin config key is deprecated. Use  NUM_STREAMS instead", it->name.GetString());
            } else {
                pluginConfig[it->name.GetString()] = std::to_string(it->value.GetInt64());
            }
        } else if (it->value.IsDouble()) {
            if (it->name.GetString() == std::string("CPU_THROUGHPUT_STREAMS") || it->name.GetString() == std::string("GPU_THROUGHPUT_STREAMS")) {
                pluginConfig["NUM_STREAMS"] = std::to_string(it->value.GetDouble());
                SPDLOG_WARN("{} plugin config key is deprecated. Use  NUM_STREAMS instead", it->name.GetString());
            } else {
                pluginConfig[it->name.GetString()] = std::to_string(it->value.GetDouble());
            }
        } else if (it->value.IsBool()) {
            pluginConfig[it->name.GetString()] = bool(it->value.GetBool());
        } else {
            return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
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

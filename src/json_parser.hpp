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
#ifndef SRC_JSON_PARSER_HPP_
#define SRC_JSON_PARSER_HPP_
#endif  // SRC_JSON_PARSER_HPP_
#include <map>
#include <string>

#include <openvino/openvino.hpp>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#pragma warning(pop)

using plugin_config_t = std::map<std::string, ov::Any>;

namespace ovms {
class Status;

class JsonParser {
public:
    /**
        * @brief Parses json node for plugin config keys and values
        * 
        * @param json node representing plugin_config
        * 
        * @return status
        */
    static Status parsePluginConfig(const rapidjson::Value& node, plugin_config_t& pluginConfig);

    /**
        * @brief Parses string for plugin config keys and values
        * 
        * @param string representing plugin_config
        * 
        * @return status
        */
    static Status parsePluginConfig(std::string command, plugin_config_t& pluginConfig);
};
}  // namespace ovms

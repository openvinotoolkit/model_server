//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <optional>
#include <vector>

#include "base_output_parser.hpp"
#include "utils.hpp"

namespace ovms {

ParametersTypeMap_t parseToolSchema(const rapidjson::Value& schema) {
    // Map each declared parameter name to its ParameterType from the tool's JSON schema.
    ParametersTypeMap_t result;
    if (!schema.IsObject()) {
        return result;
    }
    if (!schema.HasMember("properties") || !schema["properties"].IsObject()) {
        return result;
    }
    const rapidjson::Value& properties = schema["properties"];
    for (auto it = properties.MemberBegin(); it != properties.MemberEnd(); ++it) {
        if (!it->value.IsObject()) {
            continue;
        }
        if (!it->value.HasMember("type") || !it->value["type"].IsString()) {
            continue;
        }
        std::string paramName = it->name.GetString();
        std::string typeStr = it->value["type"].GetString();
        ParameterType type = ParameterType::UNKNOWN;
        if (typeStr == "string") {
            type = ParameterType::STRING;
        } else if (typeStr == "number" || typeStr == "integer") {
            type = ParameterType::NUMBER;
        } else if (typeStr == "boolean") {
            type = ParameterType::BOOLEAN;
        } else if (typeStr == "array") {
            type = ParameterType::ARRAY;
        } else if (typeStr == "object") {
            type = ParameterType::OBJECT;
        }
        result.emplace(paramName, type);
    }
    return result;
}

ToolsParameterTypeMap_t createToolsParametersTypesMap(const ToolsSchemas_t& toolsSchemas) {
    ToolsParameterTypeMap_t toolsParametersTypes;
    for (const auto& [toolName, toolSchemaWrapper] : toolsSchemas) {
        toolsParametersTypes.emplace(toolName, parseToolSchema(*toolSchemaWrapper.rapidjsonRepr));
    }
    return toolsParametersTypes;
}

std::string BaseOutputParser::buildParsingConfigStringRepresentation() const {
    std::string result = "StartTags: [";
    for (const auto& tag : parsingConfig.startTags) {
        result += tag + ", ";
    }
    result += "], EndTag: " + parsingConfig.endTag + ", ContentTagsToErase: [";
    for (const auto& tag : parsingConfig.stringsToErase) {
        result += tag + ", ";
    }
    result += "]";

    // Additionally include the resolved start token IDs and their corresponding tags in the string representation
    result += ", ResolvedStartTokenToTag: {";
    for (const auto& [tokenId, tag] : resolvedStartTokenToTag) {
        result += std::to_string(tokenId) + ": " + tag + ", ";
    }
    result += "}";

    result += ", ImplicitStart: " + std::string(implicitStart ? "true" : "false");
    result += ", NeedsSpecialTokens: " + std::string(parsingConfig.needsSpecialTokens ? "true" : "false");
    return result;
}

}  // namespace ovms

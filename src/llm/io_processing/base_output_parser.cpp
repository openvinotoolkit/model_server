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

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

#include "base_output_parser.hpp"
#include "utils.hpp"

namespace ovms {

ParametersTypeMap_t parseToolSchema(const std::string& functionName, const rapidjson::Value& schema) {
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
        toolsParametersTypes.emplace(toolName, parseToolSchema(toolName, *toolSchemaWrapper.rapidjsonRepr));
    }
    return toolsParametersTypes;
}

rapidjson::Document BaseOutputParser::wrapFirstDelta(const std::string& functionName, int toolCallIndex) {
    rapidjson::Document wrappedDelta;
    wrappedDelta.SetObject();
    rapidjson::Value toolCalls(rapidjson::kArrayType);
    rapidjson::Value toolCallObj(rapidjson::kObjectType);
    rapidjson::Value idValue(generateRandomId().c_str(), wrappedDelta.GetAllocator());
    toolCallObj.AddMember("id", idValue, wrappedDelta.GetAllocator());
    toolCallObj.AddMember("type", "function", wrappedDelta.GetAllocator());
    toolCallObj.AddMember("index", toolCallIndex, wrappedDelta.GetAllocator());
    rapidjson::Value functionObj(rapidjson::kObjectType);
    rapidjson::Value nameValue(functionName.c_str(), wrappedDelta.GetAllocator());
    functionObj.AddMember("name", nameValue, wrappedDelta.GetAllocator());

    toolCallObj.AddMember("function", functionObj, wrappedDelta.GetAllocator());
    toolCalls.PushBack(toolCallObj, wrappedDelta.GetAllocator());
    rapidjson::Value deltaWrapper(rapidjson::kObjectType);
    deltaWrapper.AddMember("tool_calls", toolCalls, wrappedDelta.GetAllocator());
    wrappedDelta.AddMember("delta", deltaWrapper, wrappedDelta.GetAllocator());
    return wrappedDelta;
}

rapidjson::Document BaseOutputParser::wrapDelta(const rapidjson::Document& delta, int toolCallIndex) {
    rapidjson::Document wrappedDelta;
    wrappedDelta.SetObject();
    rapidjson::Value toolCalls(rapidjson::kArrayType);
    rapidjson::Value toolCallObj(rapidjson::kObjectType);
    toolCallObj.AddMember("index", toolCallIndex, wrappedDelta.GetAllocator());
    rapidjson::Value functionObj(rapidjson::kObjectType);
    for (auto it = delta.MemberBegin(); it != delta.MemberEnd(); ++it) {
        rapidjson::Value key(it->name, wrappedDelta.GetAllocator());
        rapidjson::Value value(it->value, wrappedDelta.GetAllocator());
        functionObj.AddMember(key, value, wrappedDelta.GetAllocator());
    }
    toolCallObj.AddMember("function", functionObj, wrappedDelta.GetAllocator());
    toolCalls.PushBack(toolCallObj, wrappedDelta.GetAllocator());
    rapidjson::Value deltaWrapper(rapidjson::kObjectType);
    deltaWrapper.AddMember("tool_calls", toolCalls, wrappedDelta.GetAllocator());
    wrappedDelta.AddMember("delta", deltaWrapper, wrappedDelta.GetAllocator());
    return wrappedDelta;
}

}  // namespace ovms

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

#include <memory>
#include <string>
#include <utility>
#include <openvino/genai/generation_config.hpp>

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include "generation_config_builder.hpp"

namespace ovms {

void Phi4GenerationConfigBuilder::applyToolGuidance(const OpenAIRequest& request) {
    // Set tool guided generation config specific to Phi-4 model as described in template from:
    // https://github.com/vllm-project/vllm/blob/v0.9.2/examples/tool_chat_template_phi4_mini.jinja

    static const std::string beginOfToolsString = "functools";
    auto triggeredTags = std::make_shared<ov::genai::StructuredOutputConfig::TriggeredTags>();
    triggeredTags->triggers.push_back(beginOfToolsString);
    ov::genai::StructuredOutputConfig::Tag tagItem;
    tagItem.begin = beginOfToolsString;

    rapidjson::Document schema(rapidjson::kObjectType);
    auto& allocator = schema.GetAllocator();
    schema.AddMember("type", "array", allocator);
    if (request.toolChoice == "required" || requiresNamedToolCall(request)) {
        schema.AddMember("minItems", 1, allocator);
    }
    if (requiresNamedToolCall(request)) {
        schema.AddMember("maxItems", 1, allocator);
    }
    rapidjson::Value alternatives(rapidjson::kArrayType);
    for (const auto& [toolName, toolSchemaWrapper] : request.toolNameSchemaMap) {
        rapidjson::Value functionSchema(rapidjson::kObjectType);
        functionSchema.AddMember("type", "object", allocator);
        rapidjson::Value properties(rapidjson::kObjectType);
        rapidjson::Value nameSchema(rapidjson::kObjectType);
        nameSchema.AddMember("type", "string", allocator);
        rapidjson::Value names(rapidjson::kArrayType);
        names.PushBack(rapidjson::Value(toolName.c_str(), allocator), allocator);
        nameSchema.AddMember("enum", names, allocator);
        properties.AddMember("name", nameSchema, allocator);
        rapidjson::Document arguments;
        arguments.Parse(toolSchemaWrapper.stringRepr.c_str());
        rapidjson::Value argumentSchema;
        argumentSchema.CopyFrom(arguments, allocator);
        properties.AddMember("arguments", argumentSchema, allocator);
        functionSchema.AddMember("properties", properties, allocator);
        rapidjson::Value required(rapidjson::kArrayType);
        required.PushBack(rapidjson::Value("name", allocator), allocator);
        required.PushBack(rapidjson::Value("arguments", allocator), allocator);
        functionSchema.AddMember("required", required, allocator);
        alternatives.PushBack(functionSchema, allocator);
    }
    rapidjson::Value items(rapidjson::kObjectType);
    items.AddMember("anyOf", alternatives, allocator);
    schema.AddMember("items", items, allocator);
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    schema.Accept(writer);

    tagItem.content = buildToolArgumentGrammar(buffer.GetString());
    triggeredTags->tags.push_back(tagItem);
    setToolGrammar(request, triggeredTags);
}

}  // namespace ovms

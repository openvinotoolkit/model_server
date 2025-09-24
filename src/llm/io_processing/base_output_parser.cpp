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

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "base_output_parser.hpp"
#include "utils.hpp"

namespace ovms {

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

void BaseOutputParser::enableImmediateParsing() {
    immediateParsingEnabled = true;
}

bool BaseOutputParser::isImmediateParsingEnabled() const {
    return immediateParsingEnabled;
}

}  // namespace ovms

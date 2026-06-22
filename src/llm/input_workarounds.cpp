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
#include "input_workarounds.hpp"

#include <string>

#include <rapidjson/document.h>

namespace ovms {
namespace input_workarounds {

void funcArgsToObjectJson(rapidjson::Document& doc) {
    if (!doc.HasMember("messages") || !doc["messages"].IsArray()) {
        return;
    }
    auto& allocator = doc.GetAllocator();
    for (auto& message : doc["messages"].GetArray()) {
        if (!message.IsObject() || !message.HasMember("tool_calls") || !message["tool_calls"].IsArray()) {
            continue;
        }
        for (auto& toolCall : message["tool_calls"].GetArray()) {
            if (!toolCall.IsObject() || !toolCall.HasMember("function") || !toolCall["function"].IsObject()) {
                continue;
            }
            auto& function = toolCall["function"];
            if (!function.HasMember("arguments") || !function["arguments"].IsString()) {
                continue;
            }
            const char* argsStr = function["arguments"].GetString();
            rapidjson::Document argsDoc;
            argsDoc.Parse(argsStr);
            if (argsDoc.HasParseError()) {
                continue;
            }
            function["arguments"].CopyFrom(argsDoc, allocator);
        }
    }
}

void ensureNonNullContentJson(rapidjson::Document& doc) {
    if (!doc.HasMember("messages") || !doc["messages"].IsArray()) {
        return;
    }
    auto& allocator = doc.GetAllocator();
    for (auto& message : doc["messages"].GetArray()) {
        if (!message.IsObject() || !message.HasMember("tool_calls")) {
            continue;
        }
        if (!message.HasMember("content")) {
            message.AddMember("content", rapidjson::Value().SetString("", allocator), allocator);
        } else if (message["content"].IsNull()) {
            message["content"].SetString("", allocator);
        }
    }
}

void applyToJson(const ChatTemplateCaps& caps, const std::string& modelFamily, rapidjson::Document& doc) {
    if (caps.requiresObjectArguments) {
        funcArgsToObjectJson(doc);
    }
    if (caps.requiresNonNullContent) {
        ensureNonNullContentJson(doc);
    }
}

}  // namespace input_workarounds
}  // namespace ovms

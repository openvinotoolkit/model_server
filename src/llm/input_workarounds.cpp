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

namespace ovms {
namespace input_workarounds {

// --- JSON path implementations ---

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

// --- ChatHistory path implementations ---

void funcArgsToObjectHistory(ov::genai::ChatHistory& chatHistory) {
    for (size_t msgIdx = 0; msgIdx < chatHistory.size(); ++msgIdx) {
        auto message = chatHistory[msgIdx];
        if (!message.contains("tool_calls")) {
            continue;
        }
        auto toolCalls = message["tool_calls"];
        if (!toolCalls.is_array()) {
            continue;
        }
        for (size_t i = 0; i < toolCalls.size(); ++i) {
            auto toolCall = toolCalls[i];
            if (!toolCall.is_object() || !toolCall.contains("function")) {
                continue;
            }
            auto function = toolCall["function"];
            if (!function.is_object() || !function.contains("arguments")) {
                continue;
            }
            auto args = function["arguments"];
            if (!args.is_string()) {
                continue;
            }
            std::string argsStr = args.get_string();
            // Parse and replace string arguments with the parsed JSON object
            try {
                function["arguments"] = ov::genai::JsonContainer::from_json_string(argsStr);
            } catch (...) {
                // If parsing fails, leave as-is
                continue;
            }
        }
    }
}

void ensureNonNullContentHistory(ov::genai::ChatHistory& chatHistory) {
    for (size_t msgIdx = 0; msgIdx < chatHistory.size(); ++msgIdx) {
        auto message = chatHistory[msgIdx];
        if (!message.contains("tool_calls")) {
            continue;
        }
        if (!message.contains("content")) {
            message["content"] = "";
        } else if (message["content"].is_null()) {
            message["content"] = "";
        }
    }
}

void applyToHistory(const ChatTemplateCaps& caps, const std::string& modelFamily, ov::genai::ChatHistory& chatHistory) {
    if (caps.requiresObjectArguments) {
        funcArgsToObjectHistory(chatHistory);
    }
    if (caps.requiresNonNullContent) {
        ensureNonNullContentHistory(chatHistory);
    }
}

}  // namespace input_workarounds
}  // namespace ovms

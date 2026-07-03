//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "chat_template_adapter.hpp"

#include <string>

#include "../../logging.hpp"

namespace ovms {
namespace chat_template_adapter {

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
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Failed to parse function arguments as JSON: {}", argsStr);
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

void applyToHistory(const ChatTemplateCaps& caps, ov::genai::ChatHistory& chatHistory) {
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Applying chat template adaptations: {}", caps.toString());
    if (caps.requiresObjectArguments) {
        funcArgsToObjectHistory(chatHistory);
    }
    if (caps.requiresNonNullContent) {
        ensureNonNullContentHistory(chatHistory);
    }
}

}  // namespace chat_template_adapter
}  // namespace ovms

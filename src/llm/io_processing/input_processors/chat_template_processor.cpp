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

#include "chat_template_processor.hpp"

#include <optional>
#include <string>
#include <utility>

#include "../../../logging.hpp"

namespace ovms {

#if (PYTHON_DISABLE == 0)
ChatTemplateProcessor::ChatTemplateProcessor(const ov::genai::Tokenizer& tokenizer,
    PyJinjaTemplateProcessor& templateProcessor) :
    tokenizer(&tokenizer),
    templateProcessor(templateProcessor) {}

ChatTemplateProcessor::ChatTemplateProcessor(const ov::genai::Tokenizer& tokenizer) :
    tokenizer(&tokenizer),
    templateProcessor(std::nullopt) {}

std::string ChatTemplateProcessor::serializeForPyJinja(const ov::genai::ChatHistory& chatHistory) {
    // Build the minimal JSON object that PyJinjaTemplateProcessor::applyChatTemplate expects:
    // {"messages":[...], "tools":[...], "chat_template_kwargs":{...}}
    std::string json = "{\"messages\":" + chatHistory.get_messages().to_json_string();
    const auto& tools = chatHistory.get_tools();
    if (!tools.empty()) {
        json += ",\"tools\":" + tools.to_json_string();
    }
    const auto& kwargs = chatHistory.get_extra_context();
    if (!kwargs.empty()) {
        json += ",\"chat_template_kwargs\":" + kwargs.to_json_string();
    }
    json += "}";
    return json;
}

#else
ChatTemplateProcessor::ChatTemplateProcessor(const ov::genai::Tokenizer& tokenizer) :
    tokenizer(&tokenizer) {}
#endif

absl::Status ChatTemplateProcessor::process(InputRequest& req) {
    const ov::genai::ChatHistory& chatHistory = std::get<ov::genai::ChatHistory>(req.input);

#if (PYTHON_DISABLE == 0)
    if (templateProcessor.has_value()) {
        const std::string jsonBody = serializeForPyJinja(chatHistory);
        std::string promptText;
        const bool success = PyJinjaTemplateProcessor::applyChatTemplate(
            templateProcessor.value().get(), jsonBody, promptText);
        if (!success) {
            return absl::Status(absl::StatusCode::kInvalidArgument, promptText);
        }
        req.promptText = std::move(promptText);
    } else {
#endif
        constexpr bool addGenerationPrompt = true;
        const auto& tools = chatHistory.get_tools();
        const auto& kwargs = chatHistory.get_extra_context();
        const std::optional<ov::genai::JsonContainer> optTools =
            tools.empty() ? std::nullopt : std::make_optional(tools);
        const std::optional<ov::genai::JsonContainer> optKwargs =
            kwargs.empty() ? std::nullopt : std::make_optional(kwargs);
        try {
            req.promptText = tokenizer->apply_chat_template(
                chatHistory, addGenerationPrompt, {}, optTools, optKwargs);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to apply chat template: {}", e.what());
            return absl::Status(absl::StatusCode::kInvalidArgument,
                "Failed to apply chat template. The model either does not have chat template or has an invalid one.");
        }
#if (PYTHON_DISABLE == 0)
    }
#endif

    if (req.promptText.empty()) {
        return absl::Status(absl::StatusCode::kInvalidArgument,
            "Final prompt after applying chat template is empty");
    }
    return absl::OkStatus();
}

}  // namespace ovms

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
#include <variant>

#include "../../../logging.hpp"
#if (PYTHON_DISABLE == 0)
#include "../../py_jinja_template_processor.hpp"
#endif

namespace ovms {
std::string ChatTemplateProcessor::serializeForJinja(const ov::genai::ChatHistory& chatHistory) {
    // Build the minimal JSON object expected by Jinja runtime/in-process handlers:
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

#if (PYTHON_DISABLE == 0)
ChatTemplateProcessor::ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer,
    bool useMinja,
    const PreparedRuntimeChatTemplate* preparedRuntimeChatTemplate,
    PyJinjaTemplateProcessor* templateProcessorPtr) :
    tokenizer(tokenizer),
    useMinja(useMinja),
    preparedRuntimeChatTemplate(preparedRuntimeChatTemplate),
    templateProcessor(std::nullopt) {
    if (templateProcessorPtr != nullptr) {
        templateProcessor = std::ref(*templateProcessorPtr);
    }
}
#else
ChatTemplateProcessor::ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer,
    bool useMinja,
    const PreparedRuntimeChatTemplate* preparedRuntimeChatTemplate,
    PyJinjaTemplateProcessor* templateProcessorPtr) :
    tokenizer(tokenizer),
    useMinja(useMinja),
    preparedRuntimeChatTemplate(preparedRuntimeChatTemplate) {
    (void)templateProcessorPtr;
}
#endif

absl::Status ChatTemplateProcessor::extractAddGenerationPrompt(const ov::genai::ChatHistory& chatHistory,
    ov::genai::JsonContainer& kwargs, bool& addGenerationPrompt) {
    kwargs = chatHistory.get_extra_context();
    addGenerationPrompt = true;
    if (kwargs.contains("add_generation_prompt")) {
        const auto asBool = kwargs["add_generation_prompt"].as_bool();
        if (!asBool.has_value()) {
            return absl::Status(absl::StatusCode::kInvalidArgument,
                "add_generation_prompt accepts values true or false");
        }
        addGenerationPrompt = asBool.value();
        kwargs.erase("add_generation_prompt");
    }
    return absl::OkStatus();
}

absl::Status ChatTemplateProcessor::process(InputRequest& req) {
    if (!std::holds_alternative<ov::genai::ChatHistory>(req.input)) {
        return absl::Status(absl::StatusCode::kInternal,
            "ChatTemplateProcessor received input that is not a ChatHistory");
    }
    const auto& chatHistory = std::get<ov::genai::ChatHistory>(req.input);
    if (llm_calculator_logger->should_log(spdlog::level::trace)) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Chat history messages: {}", chatHistory.get_messages().to_json_string());
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "chatHistory.get_extra_context(): {}", chatHistory.get_extra_context().to_json_string());
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "tools: {}", chatHistory.get_tools().empty() ? std::string("<none>") : chatHistory.get_tools().to_json_string());
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "chatTemplateKwargs: {}", chatHistory.get_extra_context().empty() ? std::string("<none>") : chatHistory.get_extra_context().to_json_string());
    }

    const std::string jsonBody = serializeForJinja(chatHistory);

    if (!useMinja && preparedRuntimeChatTemplate != nullptr && preparedRuntimeChatTemplate->isPrepared()) {
        std::string runtimeOutput;
        RuntimeChatTemplateError runtimeError = RuntimeChatTemplateError::NONE;
        auto runtimeStatus = tryApplyPreparedChatTemplateRuntime(
            *preparedRuntimeChatTemplate,
            jsonBody,
            runtimeOutput,
            &runtimeError);
        if (runtimeStatus == RuntimeChatTemplateStatus::APPLIED) {
            req.promptText = std::move(runtimeOutput);
        } else if (runtimeStatus == RuntimeChatTemplateStatus::ERROR) {
            (void)runtimeError;
            return absl::Status(absl::StatusCode::kInvalidArgument, runtimeOutput);
        }
    }
    if (req.promptText.empty()) {
        const auto& tools = chatHistory.get_tools();
        ov::genai::JsonContainer kwargs;
        bool addGenerationPrompt = true;
        auto status = extractAddGenerationPrompt(chatHistory, kwargs, addGenerationPrompt);
        if (!status.ok()) {
            return status;
        }
        const std::optional<ov::genai::JsonContainer> optTools =
            tools.empty() ? std::nullopt : std::make_optional(tools);
        const std::optional<ov::genai::JsonContainer> optKwargs =
            kwargs.empty() ? std::nullopt : std::make_optional(kwargs);
        try {
            req.promptText = tokenizer.apply_chat_template(
                chatHistory, addGenerationPrompt, {}, optTools, optKwargs);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to apply chat template: {}", e.what());
            return absl::Status(absl::StatusCode::kInvalidArgument,
                "Failed to apply chat template. The model either does not have chat template or has an invalid one.");
        }
    }

    if (req.promptText.empty()) {
        return absl::Status(absl::StatusCode::kInvalidArgument,
            "Final prompt after applying chat template is empty");
    }
    return absl::OkStatus();
}

}  // namespace ovms

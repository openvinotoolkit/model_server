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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "../../../logging.hpp"

namespace ovms {

namespace {

bool promptEndsInOpenGemma4Reasoning(const std::string& renderedPrompt) {
    static const std::string marker = "<|channel>thought";
    const size_t end = renderedPrompt.find_last_not_of(" \t\r\n");
    if (end == std::string::npos || end + 1 < marker.size()) {
        return false;
    }
    return renderedPrompt.compare(end + 1 - marker.size(), marker.size(), marker) == 0;
}

}  // namespace

bool adaptGemma4HardToolGrammarForRenderedPrompt(
    ov::genai::GenerationConfig& config,
    const std::string& renderedPrompt) {
    using Structured = ov::genai::StructuredOutputConfig;

    if (!config.structured_output_config.has_value() || !promptEndsInOpenGemma4Reasoning(renderedPrompt)) {
        return false;
    }
    auto& structuralConfig = config.structured_output_config->structural_tags_config;
    if (!structuralConfig.has_value()) {
        return false;
    }
    auto* root = std::get_if<Structured::StructuralTag>(&structuralConfig.value());
    if (root == nullptr) {
        return false;
    }
    auto* alternativesHolder = std::get_if<std::shared_ptr<Structured::Union>>(root);
    if (alternativesHolder == nullptr || !*alternativesHolder || (*alternativesHolder)->elements.size() != 2) {
        return false;
    }

    auto& alternatives = **alternativesHolder;
    auto* requiredHolder = std::get_if<std::shared_ptr<Structured::TagsWithSeparator>>(&alternatives.elements[0]);
    auto* thoughtSequenceHolder = std::get_if<std::shared_ptr<Structured::Concat>>(&alternatives.elements[1]);
    if (requiredHolder == nullptr || !*requiredHolder || thoughtSequenceHolder == nullptr || !*thoughtSequenceHolder) {
        return false;
    }

    const auto& requiredTags = **requiredHolder;
    const auto& thoughtSequence = **thoughtSequenceHolder;
    if (!requiredTags.at_least_one || requiredTags.tags.empty() || thoughtSequence.elements.size() != 2) {
        return false;
    }
    auto* thoughtHolder = std::get_if<std::shared_ptr<Structured::Tag>>(&thoughtSequence.elements[0]);
    auto* repeatedRequiredHolder = std::get_if<std::shared_ptr<Structured::TagsWithSeparator>>(&thoughtSequence.elements[1]);
    if (thoughtHolder == nullptr || !*thoughtHolder || repeatedRequiredHolder == nullptr || !*repeatedRequiredHolder ||
        *repeatedRequiredHolder != *requiredHolder) {
        return false;
    }

    const auto& thought = **thoughtHolder;
    if (thought.begin != "<|channel>thought\n" || thought.end != "<channel|>") {
        return false;
    }
    for (const auto& tag : requiredTags.tags) {
        if (tag.begin.rfind("<|tool_call>call:", 0) != 0 || tag.end != "<tool_call|>") {
            return false;
        }
    }

    auto triggered = std::make_shared<Structured::TriggeredTags>();
    triggered->triggers = {"<|tool_call>"};
    triggered->tags = requiredTags.tags;
    triggered->at_least_one = true;
    triggered->stop_after_first = requiredTags.stop_after_first;
    Structured::StructuralTag adapted = triggered;
    structuralConfig = adapted;
    return true;
}

#if (PYTHON_DISABLE == 0)
ChatTemplateProcessor::ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer,
    PyJinjaTemplateProcessor& templateProcessor) :
    tokenizer(tokenizer),
    templateProcessor(templateProcessor) {}

ChatTemplateProcessor::ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer) :
    tokenizer(tokenizer),
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
ChatTemplateProcessor::ChatTemplateProcessor(ov::genai::Tokenizer& tokenizer) :
    tokenizer(tokenizer) {}
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
#if (PYTHON_DISABLE == 0)
    }
#endif

    if (req.promptText.empty()) {
        return absl::Status(absl::StatusCode::kInvalidArgument,
            "Final prompt after applying chat template is empty");
    }

    // Current Google Gemma4 continues the same model turn after a tool response
    // and, with thinking enabled, can leave the rendered prompt ending in an open
    // thought channel. The hard grammar was built before rendering and therefore
    // described a new-turn suffix. Reconcile it here, where prompt state is known.
    if (adaptGemma4HardToolGrammarForRenderedPrompt(req.generationConfig, req.promptText)) {
        try {
            req.generationConfig.structured_output_config.value().validate(tokenizer);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
                "Gemma4 prompt-aware hard tool grammar validation failed: {}", e.what());
            return absl::Status(absl::StatusCode::kInvalidArgument,
                std::string("Gemma4 prompt-aware hard tool grammar validation failed: ") + e.what());
        }
    }

    return absl::OkStatus();
}

}  // namespace ovms

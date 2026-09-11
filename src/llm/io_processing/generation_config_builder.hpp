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

#pragma once
#include <algorithm>
#include <cctype>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/tokenizer.hpp>

#include "base_generation_config_builder.hpp"
#include "phi4/generation_config_builder.hpp"
#include "llama3/generation_config_builder.hpp"
#include "hermes3/generation_config_builder.hpp"
#include "devstral/generation_config_builder.hpp"
#include "../apis/openai_request.hpp"
#include "../../logging.hpp"

namespace ovms {

class Gemma4GenerationConfigBuilder : public BaseGenerationConfigBuilder {
    enum class ToolConstraintMode {
        Disabled,
        Auto,
        Hard,
    };

    bool hardToolChoice{false};

    static bool isValidToolName(const std::string& name) {
        return !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '_' || c == '-' || c == '.';
        });
    }

    static bool isNamedToolChoice(const std::string& toolChoice) {
        return !toolChoice.empty() && toolChoice != "auto" && toolChoice != "none" && toolChoice != "required";
    }

    static bool isHardToolChoiceImpl(const std::string& toolChoice) {
        return toolChoice == "required" || isNamedToolChoice(toolChoice);
    }

    static ToolConstraintMode getToolConstraintMode(const OpenAIRequest& request) {
        if (request.toolNameSchemaMap.empty() || request.toolChoice == "none") {
            return ToolConstraintMode::Disabled;
        }
        if (request.toolChoice.empty() || request.toolChoice == "auto") {
            return ToolConstraintMode::Auto;
        }
        return ToolConstraintMode::Hard;
    }

    static ov::genai::StructuredOutputConfig::Tag buildToolTag(const std::string& toolName, const ToolSchemaWrapper& toolSchemaWrapper) {
        if (toolSchemaWrapper.stringRepr.empty()) {
            throw std::invalid_argument("Gemma4 guided tool schema for '" + toolName + "' is empty");
        }
        ov::genai::StructuredOutputConfig::Tag tag;
        tag.begin = "<|tool_call>call:" + toolName;
        tag.content = ov::genai::StructuredOutputConfig::JSONSchema(toolSchemaWrapper.stringRepr);
        tag.end = "<tool_call|>";
        return tag;
    }

    static std::vector<ov::genai::StructuredOutputConfig::Tag> buildToolTags(const OpenAIRequest& request) {
        std::vector<ov::genai::StructuredOutputConfig::Tag> tags;
        if (isNamedToolChoice(request.toolChoice)) {
            const auto it = request.toolNameSchemaMap.find(request.toolChoice);
            if (it == request.toolNameSchemaMap.end()) {
                throw std::invalid_argument("Gemma4 named tool_choice references an unavailable tool: " + request.toolChoice);
            }
            if (!isValidToolName(it->first)) {
                throw std::invalid_argument("Gemma4 tool name contains unsupported characters: " + it->first);
            }
            tags.push_back(buildToolTag(it->first, it->second));
            return tags;
        }
        tags.reserve(request.toolNameSchemaMap.size());
        for (const auto& [toolName, toolSchemaWrapper] : request.toolNameSchemaMap) {
            if (!isValidToolName(toolName)) {
                throw std::invalid_argument("Gemma4 tool name contains unsupported characters: " + toolName);
            }
            tags.push_back(buildToolTag(toolName, toolSchemaWrapper));
        }
        return tags;
    }

    static ov::genai::StructuredOutputConfig::StructuralTag buildAutoToolGrammar(
        std::vector<ov::genai::StructuredOutputConfig::Tag> toolTags,
        bool parallelToolCalls) {
        using Structured = ov::genai::StructuredOutputConfig;
        auto triggeredTags = std::make_shared<Structured::TriggeredTags>();
        triggeredTags->triggers = {"<|tool_call>"};
        triggeredTags->tags = std::move(toolTags);
        // TriggeredTags itself supplies the free-text prefix. Setting at_least_one
        // would prohibit ordinary prose and turn OpenAI `auto` into `required`.
        triggeredTags->at_least_one = false;
        // xgrammar's structural-tag contract maps parallel_tool_calls=false to
        // stop_after_first=true. With the default true, later triggers remain legal.
        triggeredTags->stop_after_first = !parallelToolCalls;
        return triggeredTags;
    }

    static ov::genai::StructuredOutputConfig::StructuralTag buildMandatoryToolGrammar(
        std::vector<ov::genai::StructuredOutputConfig::Tag> toolTags,
        bool parallelToolCalls) {
        using Structured = ov::genai::StructuredOutputConfig;

        auto requiredTags = std::make_shared<Structured::TagsWithSeparator>();
        requiredTags->tags = std::move(toolTags);
        requiredTags->separator = "";
        requiredTags->at_least_one = true;
        requiredTags->stop_after_first = !parallelToolCalls;

        // Google Gemma4 may open/close its thought channel before choosing a
        // tool after a tool response, including for a named choice. Selecting
        // a name restricts the available tags, not the model's thought phase.
        // xgrammar rejects empty ConstString, so optional thought is a Union of
        // tools-only versus thought-then-tools rather than Concat("", thought).
        auto thought = std::make_shared<Structured::Tag>();
        thought->begin = "<|channel>thought\n";
        thought->content = Structured::AnyText();
        thought->end = "<channel|>";

        auto thoughtThenTools = std::make_shared<Structured::Concat>();
        thoughtThenTools->elements = {thought, requiredTags};

        auto alternatives = std::make_shared<Structured::Union>();
        alternatives->elements = {requiredTags, thoughtThenTools};
        return alternatives;
    }

public:
    Gemma4GenerationConfigBuilder() = delete;
    explicit Gemma4GenerationConfigBuilder(const ov::genai::GenerationConfig& baseConfig, bool enableToolGuidedGeneration, DecodingMethod decodingMethod) :
        BaseGenerationConfigBuilder(baseConfig, enableToolGuidedGeneration, decodingMethod) {}

    bool shouldPreserveStructuredOutputOnValidationFailure() const override {
        return hardToolChoice;
    }

    void parseConfigFromRequest(const OpenAIRequest& request) override {
        BaseGenerationConfigBuilder::parseConfigFromRequest(request);
        hardToolChoice = isHardToolChoiceImpl(request.toolChoice);

        if (hardToolChoice && request.toolNameSchemaMap.empty()) {
            throw std::invalid_argument("Gemma4 hard tool_choice requires at least one available tool schema");
        }
        if (request.responseFormat.has_value() && request.toolChoice != "none" && !request.toolNameSchemaMap.empty()) {
            throw std::invalid_argument("Gemma4 response_format cannot be combined with active tool generation constraints");
        }

        const ToolConstraintMode mode = getToolConstraintMode(request);
        if (mode == ToolConstraintMode::Disabled) {
            return;
        }

        auto toolTags = buildToolTags(request);
        if (toolTags.empty()) {
            throw std::invalid_argument("Gemma4 active tool_choice did not produce an enforceable tool tag");
        }

        switch (mode) {
        case ToolConstraintMode::Auto:
            // OpenVINO GenAI TriggeredTags maps to xgrammar's lazy structural-tag
            // dispatch: normal text is unconstrained until the tool marker appears,
            // then the selected request tool name and JSON schema become authoritative.
            setStructuralTagsConfig(buildAutoToolGrammar(std::move(toolTags), request.parallelToolCalls));
            return;
        case ToolConstraintMode::Hard:
            setStructuralTagsConfig(buildMandatoryToolGrammar(std::move(toolTags), request.parallelToolCalls));
            return;
        case ToolConstraintMode::Disabled:
            return;
        }
    }
};

class GenerationConfigBuilder {
    std::unique_ptr<BaseGenerationConfigBuilder> builder_impl;

public:
    GenerationConfigBuilder() = delete;
    explicit GenerationConfigBuilder(const ov::genai::GenerationConfig& baseConfig, std::string toolParserName, bool enableToolGuidedGeneration, DecodingMethod decodingMethod) {
        if (toolParserName == "llama3") {
            builder_impl = std::make_unique<Llama3GenerationConfigBuilder>(baseConfig, enableToolGuidedGeneration, decodingMethod);
        } else if (toolParserName == "qwen3") {
            builder_impl = std::make_unique<Hermes3GenerationConfigBuilder>(baseConfig, enableToolGuidedGeneration, decodingMethod);
        } else if (toolParserName == "hermes3") {
            builder_impl = std::make_unique<Hermes3GenerationConfigBuilder>(baseConfig, enableToolGuidedGeneration, decodingMethod);
        } else if (toolParserName == "gemma4") {
            builder_impl = std::make_unique<Gemma4GenerationConfigBuilder>(baseConfig, enableToolGuidedGeneration, decodingMethod);
        } else if (toolParserName == "phi4") {
            builder_impl = std::make_unique<Phi4GenerationConfigBuilder>(baseConfig, enableToolGuidedGeneration, decodingMethod);
        } else if (toolParserName == "devstral") {
            builder_impl = std::make_unique<DevstralGenerationConfigBuilder>(baseConfig, enableToolGuidedGeneration, decodingMethod);
        } else {
            if (enableToolGuidedGeneration) {
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Option enable_tool_guided_generation is set, but will not be effective since no valid tool parser has been provided.");
            }
            builder_impl = std::make_unique<BaseGenerationConfigBuilder>(baseConfig, enableToolGuidedGeneration, decodingMethod);
        }
    }

    ov::genai::GenerationConfig& getConfig() { return builder_impl->getConfig(); }
    void adjustConfigForDecodingMethod() { builder_impl->adjustConfigForDecodingMethod(); }
    void validateStructuredOutputConfig(ov::genai::Tokenizer& tokenizer) { builder_impl->validateStructuredOutputConfig(tokenizer); }

    void unsetStructuredOutputConfig() {
        if (builder_impl->shouldPreserveStructuredOutputOnValidationFailure()) {
            SPDLOG_LOGGER_WARN(llm_calculator_logger,
                "Refusing to clear structured output after validation failure for Gemma4 required/named tool_choice; keeping generation fail-closed.");
            return;
        }
        builder_impl->unsetStructuredOutputConfig();
    }

    void parseConfigFromRequest(const OpenAIRequest& request) {
        builder_impl->parseConfigFromRequest(request);
    }

    bool hasHardToolChoice() const { return builder_impl->shouldPreserveStructuredOutputOnValidationFailure(); }
    void addStopString(const std::string& decodedStopString) { builder_impl->addStopString(decodedStopString); }
};
}  // namespace ovms

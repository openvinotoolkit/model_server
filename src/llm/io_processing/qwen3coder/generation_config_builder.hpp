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
// Qwenmonster: Qwen XML generation policy. This is NOT Hermes JSON-in-XML.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/genai/generation_config.hpp>
#include "../base_generation_config_builder.hpp"

namespace ovms {
class Qwen3CoderGenerationConfigBuilder : public BaseGenerationConfigBuilder {
    bool hardToolPolicy = false;

    static bool isNamedToolChoice(const std::string& name) {
        return !name.empty() && name != "none" && name != "auto" && name != "required";
    }

    static bool safeName(const std::string& name) {
        if (name.empty()) return false;
        for (unsigned char ch : name) {
            const bool ok = (ch >= 'a' && ch <= 'z') ||
                            (ch >= 'A' && ch <= 'Z') ||
                            (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.';
            if (!ok) return false;
        }
        return true;
    }

    static ov::genai::StructuredOutputConfig::Tag makeXMLToolTag(
        const std::string& name, const ToolSchemaWrapper& schema) {
        if (!safeName(name))
            throw std::invalid_argument("Qwen XML tool name contains unsafe characters: " + name);
        if (schema.stringRepr.empty())
            throw std::invalid_argument("Qwen XML tool schema is empty: " + name);
        using Structured = ov::genai::StructuredOutputConfig;
        // QwenXMLParametersFormat uses XGrammar's qwen_xml_parameter schema
        // dialect. A plain JSONSchema would generate JSON, not native Qwen XML.
        return Structured::Tag{
            "<tool_call>\n<function=" + name + ">\n",
            Structured::QwenXMLParametersFormat(schema.stringRepr),
            "</function>\n</tool_call>"};
    }

    // Qwen's template leaves generation inside <think> unless enable_thinking is explicitly false.
    static bool thinkingIsOpen(const OpenAIRequest& request) {
        return !request.enableThinking.has_value() || *request.enableThinking;
    }

    // Tool grammar starts only after the think closer, so a draft <tool_call>
    // inside the thought cannot terminate reasoning or fire a tool trigger.
    static ov::genai::StructuredOutputConfig::StructuralTag afterOpenThought(
        ov::genai::StructuredOutputConfig::StructuralTag toolGrammar) {
        using Structured = ov::genai::StructuredOutputConfig;
        auto thought = std::make_shared<Structured::Tag>(
            std::string{},
            Structured::AnyText{},
            std::string{"</think>\n\n"});
        return std::make_shared<Structured::Concat>(
            Structured::StructuralTag{std::move(thought)},
            std::move(toolGrammar));
    }

    static std::vector<ov::genai::StructuredOutputConfig::Tag> makeToolTags(const OpenAIRequest& request) {
        std::vector<ov::genai::StructuredOutputConfig::Tag> tags;
        if (isNamedToolChoice(request.toolChoice)) {
            auto it = request.toolNameSchemaMap.find(request.toolChoice);
            if (it == request.toolNameSchemaMap.end())
                throw std::invalid_argument("Unavailable Qwen XML named tool: " + request.toolChoice);
            tags.push_back(makeXMLToolTag(it->first, it->second));
            return tags;
        }
        tags.reserve(request.toolNameSchemaMap.size());
        for (const auto& [name, schema] : request.toolNameSchemaMap)
            tags.push_back(makeXMLToolTag(name, schema));
        return tags;
    }

public:
    Qwen3CoderGenerationConfigBuilder() = delete;
    explicit Qwen3CoderGenerationConfigBuilder(
        const ov::genai::GenerationConfig& base,
        bool enableToolGuidedGeneration,
        DecodingMethod method) :
        BaseGenerationConfigBuilder(base, enableToolGuidedGeneration, method) {}

    void parseConfigFromRequest(const OpenAIRequest& request) override {
        hardToolPolicy = false;
        BaseGenerationConfigBuilder::parseConfigFromRequest(request);
        const bool hard = request.toolChoice == "required" || isNamedToolChoice(request.toolChoice);
        if (hard && request.toolNameSchemaMap.empty())
            throw std::invalid_argument("Qwen XML hard tool_choice requires nonempty tools");
        const bool active = !request.toolNameSchemaMap.empty() && request.toolChoice != "none";
        if (active && request.responseFormat.has_value())
            throw std::invalid_argument("Qwen XML tool grammar conflicts with response_format");
        if (!active) return;

        const auto tags = makeToolTags(request);
        using Structured = ov::genai::StructuredOutputConfig;
        const bool openThought = thinkingIsOpen(request);
        if (hard) {
            // Hard constraints are active from the first generated token.
            // With an open think, that token is still the thought: require
            // </think> before the XML tool tags. Thinking-off prompts already
            // closed the think block, so the tool tags apply immediately.
            hardToolPolicy = true;
            auto required = std::make_shared<Structured::TagsWithSeparator>();
            required->tags = tags;
            required->separator = "\n";
            required->at_least_one = true;
            required->stop_after_first = !request.parallelToolCalls;
            if (openThought)
                setStructuralTagsConfig(afterOpenThought(required));
            else
                setStructuralTagsConfig(required);
        } else if ((request.toolChoice.empty() || request.toolChoice == "auto") &&
                   enableToolGuidedGeneration) {
            // Opt-in lazy profile. No forced call in auto mode. The <tool_call>
            // trigger is installed only after an open thought closes, so a draft
            // call inside <think> cannot start the tool grammar.
            auto triggered = std::make_shared<Structured::TriggeredTags>();
            triggered->triggers = {"<tool_call>"};
            triggered->tags = tags;
            triggered->at_least_one = false;
            triggered->stop_after_first = !request.parallelToolCalls;
            if (openThought)
                setStructuralTagsConfig(afterOpenThought(triggered));
            else
                setStructuralTagsConfig(triggered);
        }
    }

    bool requiresValidStructuredOutput() const override {
        return hardToolPolicy;
    }
};
}  // namespace ovms

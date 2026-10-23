//*****************************************************************************
// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0.
//*****************************************************************************

#pragma once
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/genai/generation_config.hpp>

#include "../base_generation_config_builder.hpp"

namespace ovms {
class Gemma4GenerationConfigBuilder : public BaseGenerationConfigBuilder {
    bool hardToolPolicy = false;

    static bool isNamedToolChoice(const std::string& toolChoice) {
        return !toolChoice.empty() && toolChoice != "none" && toolChoice != "auto" && toolChoice != "required";
    }

    static bool isSafeToolName(const std::string& name) {
        if (name.empty())
            return false;
        for (unsigned char c : name) {
            const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
            if (!ok)
                return false;
        }
        return true;
    }

    static ov::genai::StructuredOutputConfig::Tag buildToolTag(
        const std::string& toolName,
        const ToolSchemaWrapper& toolSchemaWrapper) {
        if (!isSafeToolName(toolName))
            throw std::invalid_argument("Gemma4 tool name '" + toolName + "' contains characters outside [A-Za-z0-9_.-]");
        if (toolSchemaWrapper.stringRepr.empty())
            throw std::invalid_argument("Gemma4 guided tool schema for '" + toolName + "' is empty");
        ov::genai::StructuredOutputConfig::Tag tag;
        tag.begin = "<|tool_call>call:" + toolName;
        // 2026.4 JSONSchema takes the schema only. A hand-built
        // max_whitespace_cnt string is not applied, so the model can fill the
        // open call with newlines until max_tokens and the parser publishes nothing.
        tag.content = ov::genai::StructuredOutputConfig::JSONSchema(toolSchemaWrapper.stringRepr);
        tag.end = "<tool_call|>";
        return tag;
    }

    static std::vector<ov::genai::StructuredOutputConfig::Tag> buildToolTags(const OpenAIRequest& request) {
        std::vector<ov::genai::StructuredOutputConfig::Tag> toolTags;
        if (isNamedToolChoice(request.toolChoice)) {
            const auto it = request.toolNameSchemaMap.find(request.toolChoice);
            if (it == request.toolNameSchemaMap.end())
                throw std::invalid_argument("Gemma4 named tool_choice references an unavailable tool: " + request.toolChoice);
            toolTags.push_back(buildToolTag(it->first, it->second));
            return toolTags;
        }
        toolTags.reserve(request.toolNameSchemaMap.size());
        for (const auto& [toolName, toolSchemaWrapper] : request.toolNameSchemaMap)
            toolTags.push_back(buildToolTag(toolName, toolSchemaWrapper));
        return toolTags;
    }

public:
    Gemma4GenerationConfigBuilder() = delete;
    explicit Gemma4GenerationConfigBuilder(
        const ov::genai::GenerationConfig& baseConfig,
        bool enableToolGuidedGeneration,
        DecodingMethod decodingMethod) :
        BaseGenerationConfigBuilder(baseConfig, enableToolGuidedGeneration, decodingMethod) {}

    void parseConfigFromRequest(const OpenAIRequest& request) override {
        hardToolPolicy = false;
        BaseGenerationConfigBuilder::parseConfigFromRequest(request);
        const bool hardChoice = request.toolChoice == "required" || isNamedToolChoice(request.toolChoice);
        if (hardChoice && request.toolNameSchemaMap.empty())
            throw std::invalid_argument("Gemma4 hard tool_choice requires at least one available tool schema");
        const bool activeTools = !request.toolNameSchemaMap.empty() && request.toolChoice != "none";
        if (request.responseFormat.has_value() && activeTools)
            throw std::invalid_argument("Gemma4 response_format cannot be combined with active tool generation constraints");
        if (!activeTools)
            return;

        using Structured = ov::genai::StructuredOutputConfig;
        const auto toolTags = buildToolTags(request);
        const bool stopAfterFirst = !request.parallelToolCalls;
        if (hardChoice) {
            hardToolPolicy = true;
            auto required = std::make_shared<Structured::TagsWithSeparator>();
            required->tags = toolTags;
            required->separator = "";
            required->at_least_one = true;
            required->stop_after_first = stopAfterFirst;
            setStructuralTagsConfig(required);
            return;
        }
        if ((request.toolChoice.empty() || request.toolChoice == "auto") && enableToolGuidedGeneration) {
            auto triggered = std::make_shared<Structured::TriggeredTags>();
            triggered->triggers = {"<|tool_call>"};
            triggered->tags = toolTags;
            triggered->at_least_one = false;
            triggered->stop_after_first = stopAfterFirst;
            setStructuralTagsConfig(triggered);
        }
    }

    bool requiresValidStructuredOutput() const override {
        return hardToolPolicy;
    }
};
}  // namespace ovms

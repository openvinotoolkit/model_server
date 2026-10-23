//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//*****************************************************************************

#include "rendered_prompt_state.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <variant>
#include <stdexcept>
#ifdef _MSC_VER
#pragma warning(disable : 6313)
#endif
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ovms {
namespace {

constexpr const char* GEMMA4_THOUGHT_START = "<|channel>thought\n";
constexpr const char* GEMMA4_THOUGHT_END = "<channel|>";

size_t latestGenerationBoundary(const std::string& prompt) {
    // These anchors begin the only suffix that can describe the next decode.
    // Markers in earlier user/tool content therefore cannot open current reasoning.
    static const std::vector<std::string> anchors{
        "<|turn>model\n",
        "<|im_start|>assistant\n",
        "<tool_response|>",
    };
    size_t boundary = 0;
    for (const auto& anchor : anchors) {
        const size_t pos = prompt.rfind(anchor);
        if (pos != std::string::npos)
            boundary = std::max(boundary, pos + anchor.size());
    }
    return boundary;
}

size_t latestTag(
    const std::string& prompt,
    const std::vector<std::string>& tags,
    size_t from) {
    size_t latest = std::string::npos;
    for (const auto& tag : tags) {
        if (tag.empty())
            continue;
        const size_t pos = prompt.rfind(tag);
        if (pos != std::string::npos && pos >= from &&
            (latest == std::string::npos || pos > latest))
            latest = pos;
    }
    return latest;
}

}  // namespace

Gemma4RenderedPromptState classifyRenderedPromptState(
    const std::string& renderedPrompt,
    const std::vector<std::string>& thoughtStartTags,
    const std::string& thoughtEndTag) {
    const size_t boundary = latestGenerationBoundary(renderedPrompt);
    const size_t start = latestTag(renderedPrompt, thoughtStartTags, boundary);
    const size_t end = thoughtEndTag.empty() ? std::string::npos :
                                               renderedPrompt.rfind(thoughtEndTag);
    const bool endInCurrentSuffix = end != std::string::npos && end >= boundary;

    if (start != std::string::npos && (!endInCurrentSuffix || start > end))
        return Gemma4RenderedPromptState::OPEN_THOUGHT;
    if (endInCurrentSuffix && (start == std::string::npos || end > start))
        return Gemma4RenderedPromptState::CLOSED_THOUGHT;
    return Gemma4RenderedPromptState::NEW_TURN;
}

Gemma4RenderedPromptState classifyGemma4RenderedPromptState(
    const std::string& renderedPrompt) {
    return classifyRenderedPromptState(
        renderedPrompt, {GEMMA4_THOUGHT_START}, GEMMA4_THOUGHT_END);
}

bool adaptGemma4ToolGrammarForRenderedPrompt(
    ov::genai::GenerationConfig& config,
    const std::string& renderedPrompt) {
    using Structured = ov::genai::StructuredOutputConfig;
    if (classifyGemma4RenderedPromptState(renderedPrompt) !=
        Gemma4RenderedPromptState::OPEN_THOUGHT)
        return false;
    if (!config.structured_output_config ||
        !config.structured_output_config->structural_tags_config)
        return false;

    auto& outer = *config.structured_output_config->structural_tags_config;
    auto* grammar = std::get_if<Structured::StructuralTag>(&outer);
    if (grammar == nullptr)
        return false;
    auto* raw = std::get_if<std::string>(grammar);
    if (raw == nullptr)
        return false;
    return adaptGemma4TokenGrammarForRenderedPrompt(*raw, renderedPrompt);
}

bool adaptGemma4TokenGrammarForRenderedPrompt(std::string& grammar, const std::string& renderedPrompt) {
    if (classifyGemma4RenderedPromptState(renderedPrompt) != Gemma4RenderedPromptState::OPEN_THOUGHT)
        return false;
    rapidjson::Document document;
    document.Parse(grammar.c_str());
    if (document.HasParseError() || !document.IsObject() || !document.HasMember("format"))
        throw std::invalid_argument("Invalid Gemma4 tool grammar");
    auto& format = document["format"];
    if (!format.IsObject() || !format.HasMember("type") || !format["type"].IsString())
        return false;
    const std::string type = format["type"].GetString();
    rapidjson::Value continuation;
    auto& allocator = document.GetAllocator();
    if (type == "or" && format.HasMember("elements") && format["elements"].IsArray() &&
        !format["elements"].Empty() && format["elements"][0].IsObject() &&
        format["elements"][0].HasMember("at_least_one") &&
        format["elements"][0]["at_least_one"].IsBool() && format["elements"][0]["at_least_one"].GetBool()) {
        continuation.CopyFrom(format["elements"][0], allocator);
    } else if (type == "token_triggered_tags") {
        continuation.CopyFrom(format, allocator);
    } else {
        return false; // Includes an already adapted residual sequence.
    }
    rapidjson::Document residual;
    residual.Parse(R"({"type":"sequence","elements":[{"type":"tag","begin":"","content":{"type":"any_tokens"},"end":{"type":"token","token":"<channel|>"}}]})");
    rapidjson::Value sequence;
    sequence.CopyFrom(residual, allocator);
    sequence["elements"].PushBack(continuation, allocator);
    format = std::move(sequence);
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    grammar = buffer.GetString();
    return true;
}

}  // namespace ovms

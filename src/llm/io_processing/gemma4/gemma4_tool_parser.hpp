//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//*****************************************************************************
#pragma once

#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "src/llm/io_processing/base_output_parser.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

namespace ovms {

class Gemma4ToolParser : public BaseOutputParser {
public:
    // Public protocol constants are also used by the private recursive parser
    // implementation and conformance tests. They are semantic markers, not state.
    static const std::string TOOL_CALL_START_TAG;
    static const std::string TOOL_CALL_END_TAG;
    static const std::string TOOL_CALL_NAME_PREFIX;
    static const std::string TOOL_ARGS_STRING_INDICATOR;
    static const std::string TURN_END_TAG;
    static const std::string TOOL_RESPONSE_START_TAG;

protected:
    enum class State {
        Content,
        ToolCallStarted,
        ToolCallParameters,
        ToolCallEnded,
        AfterToolCall
    };

public:
    Gemma4ToolParser() = delete;

    static OutputParsingConfig defaultParsingConfig() {
        OutputParsingConfig cfg;
        cfg.startTags = {"<|tool_call>"};
        cfg.tokenIdStartTags = {"<|tool_call>"};
        // Real Gemma4 traces occasionally omit the repeated tool marker after a
        // reasoning channel and continue directly with call:name{...}. The generic
        // OutputParser returns to UNKNOWN after reasoning; preambleStartTags is the
        // deliberately narrow hook for this variant and is not matched mid-content.
        cfg.preambleStartTags = {"call:"};
        cfg.endTag = "<tool_call|>";
        cfg.needsSpecialTokens = true;
        return cfg;
    }

    explicit Gemma4ToolParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    // Registry-aware overload. Empty registry deliberately means "syntax-only" for
    // backwards compatibility; OutputParser wiring can pass request tool names to
    // enable executable-call validation without changing parser grammar.
    Gemma4ToolParser(ov::genai::Tokenizer& tokenizer,
        const ToolsSchemas_t& toolsSchemas,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {
        for (const auto& [name, schema] : toolsSchemas) {
            (void)schema;
            allowedToolNames.insert(name);
        }
        enforceToolRegistry = !allowedToolNames.empty();
    }

    void resetState() override {
        streamingContent.clear();
        streamingPosition = 0;
        currentState = State::Content;
        toolCall = {};
        toolCallIndex = -1;
        currentArgsOpen = '{';
        currentArgsClose = '}';
        currentCallValid = true;
    }

    std::optional<Delta> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;

    // Compatibility helpers retained for existing unit tests/callers. They now use
    // the same recursive native-value parser as the streaming path.
    static std::string normalizeArgStr(const std::string& arg);
    static std::string parseArrayParameter(const std::string& argumentStr);
    static std::string parseObjectParameter(const std::string& argumentStr);

private:
    static std::optional<std::string> parseNativeArgumentsBody(const std::string& argumentsBody);
    static std::optional<size_t> findMatchingContainerEnd(const std::string& text, size_t openPos, char openChar, char closeChar);
    static std::string normalizeToolName(std::string rawName);

    bool toolNameAllowed(const std::string& name) const {
        return !enforceToolRegistry || allowedToolNames.count(name) != 0;
    }

    bool parseNewContent();
    bool parseInContentState();
    bool parseInToolCallState();
    bool parseToolCallParametersState();
    bool parseInToolCallEndedState();

    std::optional<Delta> wrapDeltaContent(const std::string& content);
    ToolCallDelta wrapDeltaArgs(const std::string& argsStr, int toolCallIndex);

    std::string streamingContent;
    size_t streamingPosition{0};
    State currentState{State::Content};
    ToolCall toolCall;
    int toolCallIndex{-1};
    char currentArgsOpen{'{'};
    char currentArgsClose{'}'};
    bool currentCallValid{true};
    bool enforceToolRegistry{false};
    std::unordered_set<std::string> allowedToolNames;
};
}  // namespace ovms

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

#include <algorithm>
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
        // Bare `call:` is a tolerance/recovery form, not the canonical Google
        // Gemma4 protocol. Canonical tool calls use <|tool_call>call:name{...}.
        // The syntax-only constructor retains broad text markers for compatibility;
        // the registry-aware production constructor narrows both canonical and bare
        // text starts to request tools while keeping the special-token start separate.
        cfg.preambleStartTags = {"call:"};
        cfg.endTag = "<tool_call|>";
        cfg.needsSpecialTokens = true;
        cfg.ownsToolCallBoundaries = true;
        return cfg;
    }

    explicit Gemma4ToolParser(ov::genai::Tokenizer& tokenizer,
        std::optional<OutputParsingConfig> configOverride = std::nullopt) :
        BaseOutputParser(tokenizer,
            configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

    // Registry-aware production form. Text-phase detection requires the complete
    // native prefix for a request tool, including the immediate argument opener.
    // This prevents ordinary prose containing a literal <|tool_call> marker from
    // entering tool phase. tokenIdStartTags intentionally remains <|tool_call> so
    // OVMSTextStreamer can still make the special token visible before text-phase
    // confirmation. Bare `call:` recovery is narrowed the same way.
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
        if (enforceToolRegistry && !configOverride.has_value()) {
            parsingConfig.startTags.clear();
            parsingConfig.preambleStartTags.clear();
            parsingConfig.startTags.reserve(allowedToolNames.size() * 4);
            parsingConfig.preambleStartTags.reserve(allowedToolNames.size() * 2);
            for (const auto& name : allowedToolNames) {
                const std::string bracePreamble = TOOL_CALL_NAME_PREFIX + name + "{";
                const std::string parenPreamble = TOOL_CALL_NAME_PREFIX + name + "(";
                parsingConfig.startTags.push_back(TOOL_CALL_START_TAG + bracePreamble);
                parsingConfig.startTags.push_back(TOOL_CALL_START_TAG + parenPreamble);
                parsingConfig.startTags.push_back(TOOL_CALL_START_TAG + ":" + name + "{");
                parsingConfig.startTags.push_back(TOOL_CALL_START_TAG + ":" + name + "(");
                parsingConfig.preambleStartTags.push_back(bracePreamble);
                parsingConfig.preambleStartTags.push_back(parenPreamble);
            }
        }
    }

    std::optional<PendingToolFrameDiagnostic> pendingToolFrameDiagnostic() const override {
        if (currentState != State::ToolCallStarted && currentState != State::ToolCallParameters)
            return std::nullopt;
        const size_t start = currentState == State::ToolCallParameters && streamingPosition > 0
            ? streamingPosition - 1 : streamingPosition;
        return PendingToolFrameDiagnostic{
            currentState == State::ToolCallParameters ? "ToolCallParameters" : "ToolCallStarted",
            streamingContent.size() - std::min(start, streamingContent.size()), toolCall.name};
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
        currentCallBare = false;
        currentCallStartPos = 0;
    }

    std::optional<Delta> parseChunk(const std::string& chunk, const std::vector<int64_t>& tokens, ov::genai::GenerationFinishReason finishReason) override;

    // Compatibility helpers retained for existing unit tests/callers. They now use
    // the same recursive native-value parser as the streaming path.
    static std::string normalizeArgStr(const std::string& arg);
    static std::string parseArrayParameter(const std::string& argumentStr);
    static std::string parseObjectParameter(const std::string& argumentStr);

private:
    friend struct Gemma4ToolParserTestAccess;
    static std::optional<std::string> parseNativeArgumentsBody(const std::string& argumentsBody);
    static std::optional<size_t> findMatchingContainerEnd(const std::string& text, size_t openPos, char openChar, char closeChar, size_t& malformedEndTag);
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
    // Whether the in-flight call started from a bare line-start `call:`
    // (no "<|tool_call>" anchor) and where that start sits in
    // streamingContent. An unknown tool name on a bare call must be rewound
    // and re-emitted as ordinary content; an anchored unknown call stays dropped.
    bool currentCallBare{false};
    size_t currentCallStartPos{0};
    bool enforceToolRegistry{false};
    std::unordered_set<std::string> allowedToolNames;
};
}  // namespace ovms

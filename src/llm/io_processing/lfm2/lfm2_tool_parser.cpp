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
#include "lfm2_tool_parser.hpp"

#include "python_literal.hpp"

#include "../utils.hpp"
#include "../../../logging.hpp"
#include "../../../stringutils.hpp"

#include <cctype>
#include <utility>

namespace ovms {

namespace {

// LFM2.5 assigns token ID 124905 to <|tool_call_start|>; LFM2 uses 10.
// (Token-ID resolution happens automatically via tokenIdStartTags.)

// Tool-call format delimiters shared by LFM2 and LFM2.5.
const std::string TOOL_LIST_START_INDICATOR = "[";
const std::string TOOL_LIST_END_INDICATOR = "]";
const std::string TOOL_ARGS_START_INDICATOR = "(";
const std::string TOOL_ARGS_END_INDICATOR = ")";
// EOS token emitted by the LFM2.5 chat template after tool-call blocks.
const std::string EOS_TOKEN_STR = "<|im_end|>";

// Closing parenthesis of the active argument list. Even runs of backslashes do
// not keep a quote open, so a following ")" stays structural. Odd runs do.
size_t findArgumentListEnd(const std::string& text, size_t start) {
    int parenDepth = 0;
    int bracketDepth = 0;
    int braceDepth = 0;
    bool inString = false;
    char quote = 0;
    bool escaped = false;
    for (size_t i = start; i < text.size(); ++i) {
        const char c = text[i];
        if (inString) {
            if (escaped) {
                escaped = false;
                continue;
            }
            if (c == '\\') {
                escaped = true;
                continue;
            }
            if (c == quote)
                inString = false;
            continue;
        }
        if (c == '\'' || c == '"') {
            inString = true;
            quote = c;
            continue;
        }
        if (c == '(') {
            ++parenDepth;
        } else if (c == ')') {
            if (parenDepth == 0 && bracketDepth == 0 && braceDepth == 0)
                return i;
            if (parenDepth > 0)
                --parenDepth;
        } else if (c == '[') {
            ++bracketDepth;
        } else if (c == ']') {
            if (bracketDepth > 0)
                --bracketDepth;
        } else if (c == '{') {
            ++braceDepth;
        } else if (c == '}') {
            if (braceDepth > 0)
                --braceDepth;
        }
    }
    return std::string::npos;
}

// Comma that separates tool calls after a completed argument list. Whitespace
// around the comma is optional; a "]" or end tag is not a separator.
size_t findCallSeparator(const std::string& text, size_t start) {
    for (size_t i = start; i < text.size(); ++i) {
        if (std::isspace(static_cast<unsigned char>(text[i])) != 0)
            continue;
        return text[i] == ',' ? i : std::string::npos;
    }
    return std::string::npos;
}

// ---------------------------------------------------------------------------
// State-machine step functions
// ---------------------------------------------------------------------------

bool parseInContentState(const std::string& streamingContent, size_t& streamingPosition,
    Lfm2ParseState& currentState,
    const std::string& startTag, const std::string& endTag) {
    size_t startTagPos = streamingContent.find(startTag, streamingPosition);
    size_t endTagPos = streamingContent.find(endTag, streamingPosition);
    if (endTagPos != std::string::npos && startTagPos == std::string::npos) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected stray end tag at position: {}", endTagPos);
        streamingPosition = endTagPos + endTag.length();
        return false;
    }
    if (startTagPos != std::string::npos) {
        if (startTagPos > streamingPosition) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Content before tool-call start tag at position: {}", startTagPos);
            return true;
        }
        currentState = Lfm2ParseState::ToolCallStarted;
        streamingPosition = startTagPos + startTag.length();
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Detected tool-call start at position: {}", startTagPos);
        return false;
    }
    return true;
}

bool parseInToolCallState(const std::string& streamingContent, ToolCall& toolCall,
    size_t& streamingPosition, Lfm2ParseState& currentState) {
    // Only the list opener at the current cursor is structural. A later "["
    // such as a=[...] belongs to arguments and must not move the name start.
    while (streamingPosition < streamingContent.size() &&
           std::isspace(static_cast<unsigned char>(streamingContent[streamingPosition])) != 0) {
        ++streamingPosition;
    }
    if (streamingPosition < streamingContent.size() &&
        streamingContent.compare(streamingPosition, TOOL_LIST_START_INDICATOR.size(), TOOL_LIST_START_INDICATOR) == 0) {
        streamingPosition += TOOL_LIST_START_INDICATOR.size();
        while (streamingPosition < streamingContent.size() &&
               std::isspace(static_cast<unsigned char>(streamingContent[streamingPosition])) != 0) {
            ++streamingPosition;
        }
    }

    const size_t argsPos = streamingContent.find(TOOL_ARGS_START_INDICATOR, streamingPosition);
    if (argsPos == std::string::npos)
        return false;
    // Require the tool-list "[" before this argument list. Nested brackets after
    // the name are irrelevant; a call with no list opener is rejected.
    const size_t listOpener = streamingContent.find(TOOL_LIST_START_INDICATOR);
    if (listOpener == std::string::npos || listOpener >= argsPos)
        return false;

    std::string toolName = streamingContent.substr(streamingPosition, argsPos - streamingPosition);
    trim(toolName);
    if (toolName.empty())
        return false;
    toolCall = ToolCall{generateRandomId(), toolName, ""};
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed tool name: {}", toolName);
    streamingPosition = argsPos + TOOL_ARGS_START_INDICATOR.length();
    currentState = Lfm2ParseState::ToolCallParameters;
    return true;
}

bool parseInToolCallParametersState(const std::string& streamingContent, ToolCall& toolCall,
    size_t& streamingPosition, Lfm2ParseState& currentState) {
    const size_t pos = findArgumentListEnd(streamingContent, streamingPosition);
    if (pos == std::string::npos)
        return false;

    const std::string argumentsStr = streamingContent.substr(streamingPosition, pos - streamingPosition);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Parsed arguments string: {}", argumentsStr);
    // Throws before any argument delta is published. A name already emitted
    // therefore cannot become a successful tool call.
    toolCall.arguments = ovms::lfm2::parseArguments(argumentsStr);
    currentState = Lfm2ParseState::ToolCallEnded;
    streamingPosition = pos + TOOL_ARGS_END_INDICATOR.length();
    return true;
}

bool parseInToolCallEndedState(const std::string& streamingContent, size_t& streamingPosition,
    Lfm2ParseState& currentState, const std::string& endTag) {
    const size_t listEndPos = streamingContent.find(TOOL_LIST_END_INDICATOR, streamingPosition);
    const size_t separatorPos = findCallSeparator(streamingContent, streamingPosition);
    const size_t endTagPos = streamingContent.find(endTag, streamingPosition);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "ToolCallEnded: content from pos {}: {}",
        streamingPosition, streamingContent.substr(streamingPosition));
    if (listEndPos == std::string::npos && separatorPos == std::string::npos && endTagPos == std::string::npos)
        return false;
    // size_t npos comparisons are unsafe; rank only positions that exist.
    const auto before = [](size_t candidate, size_t limit) {
        return candidate != std::string::npos && (limit == std::string::npos || candidate < limit);
    };
    if (before(separatorPos, listEndPos) && before(separatorPos, endTagPos)) {
        streamingPosition = separatorPos + 1;
        while (streamingPosition < streamingContent.size() &&
               std::isspace(static_cast<unsigned char>(streamingContent[streamingPosition])) != 0) {
            ++streamingPosition;
        }
        currentState = Lfm2ParseState::ToolCallStarted;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Tool-call separator at {}, expecting next call", separatorPos);
        return true;
    }
    if (endTagPos != std::string::npos) {
        streamingPosition = endTagPos + endTag.length();
        currentState = Lfm2ParseState::AfterToolCall;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "End tag at {}", endTagPos);
        return true;
    }
    if (listEndPos != std::string::npos) {
        const size_t afterList = listEndPos + TOOL_LIST_END_INDICATOR.length();
        const std::string tail = streamingContent.substr(afterList);
        // Hold while "]<" is only a prefix of <|tool_call_end|>; otherwise the
        // partial tag is published as content and the real closer never matches.
        if (tail.empty() || endTag.compare(0, tail.size(), tail) == 0)
            return false;
        streamingPosition = afterList;
        currentState = Lfm2ParseState::AfterToolCall;
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "End of tool list at {}", listEndPos);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Delta-wrapping helpers
// ---------------------------------------------------------------------------

ContentDelta wrapDeltaContent(const std::string& content) {
    return ContentDelta{content};
}

ToolCallDelta wrapDeltaArgs(const std::string& argsStr, int toolCallIndex) {
    return ToolCallDelta{toolCallIndex, std::nullopt, std::nullopt, argsStr};
}

void cutEOSFromContent(std::string& content) {
    size_t pos = content.find(EOS_TOKEN_STR);
    if (pos != std::string::npos)
        content = content.substr(0, pos);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Lfm2ToolParser implementation
// ---------------------------------------------------------------------------

bool Lfm2ToolParser::parseNewContent() {
    const std::string& startTag = parsingConfig.startTags[0];
    const std::string& endTag = parsingConfig.endTags.front();
    switch (this->currentState) {
    case Lfm2ParseState::Content:
        return parseInContentState(this->streamingContent, this->streamingPosition,
            this->currentState, startTag, endTag);
    case Lfm2ParseState::ToolCallStarted: {
        auto ok = parseInToolCallState(this->streamingContent, this->toolCall,
            this->streamingPosition, this->currentState);
        if (ok)
            this->toolCallIndex++;
        return ok;
    }
    case Lfm2ParseState::ToolCallParameters:
        return parseInToolCallParametersState(this->streamingContent, this->toolCall,
            this->streamingPosition, this->currentState);
    case Lfm2ParseState::ToolCallEnded:
        return parseInToolCallEndedState(this->streamingContent, this->streamingPosition,
            this->currentState, endTag);
    case Lfm2ParseState::AfterToolCall:
        break;
    }
    return false;
}

bool Lfm2ToolParser::isStructuralEndTag(const std::string& chunk, size_t offset) const {
    if (parsingConfig.startTags.empty() || parsingConfig.endTags.empty() || offset > chunk.size())
        return true;
    const std::string& startTag = parsingConfig.startTags[0];
    const std::string& endTag = parsingConfig.endTags.front();
    const std::string& prior = streamingContent;
    const size_t candidate = prior.size() + offset;
    auto at = [&](size_t index) -> char {
        return index < prior.size() ? prior[index] : chunk[index - prior.size()];
    };
    auto match = [&](size_t index, const std::string& tag) {
        if (tag.empty() || index + tag.size() > prior.size() + chunk.size())
            return false;
        for (size_t k = 0; k < tag.size(); ++k) {
            if (at(index + k) != tag[k])
                return false;
        }
        return true;
    };

    bool inTool = false;
    bool inString = false;
    char quote = 0;
    bool escaped = false;
    for (size_t i = 0; i < candidate;) {
        const char c = at(i);
        if (inString) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == quote) {
                inString = false;
            }
            ++i;
            continue;
        }
        if (!inTool && match(i, startTag)) {
            inTool = true;
            i += startTag.size();
            continue;
        }
        if (inTool && match(i, endTag)) {
            inTool = false;
            i += endTag.size();
            continue;
        }
        if (inTool && (c == '\'' || c == '"')) {
            inString = true;
            quote = c;
        }
        ++i;
    }
    return !inString;
}

std::optional<Delta> Lfm2ToolParser::parseChunk(const std::string& chunk,
    const std::vector<int64_t>& /*tokens*/,
    ov::genai::GenerationFinishReason finishReason) {
    if (!chunk.empty())
        this->streamingContent += chunk;

    const std::string& startTag = parsingConfig.startTags[0];
    const std::string& endTag = parsingConfig.endTags.front();
    // One observable delta per call. Non-emitting transitions keep walking the
    // same states so a closed envelope can be drained without a new phase.
    for (int step = 0; step < 32; ++step) {
        const size_t positionBefore = this->streamingPosition;
        switch (this->currentState) {
        case Lfm2ParseState::Content: {
            const bool contentReady = parseInContentState(this->streamingContent, this->streamingPosition,
                this->currentState, startTag, endTag);
            if (this->currentState != Lfm2ParseState::Content)
                continue;
            if (!contentReady && this->streamingPosition != positionBefore)
                continue;
            const size_t contentEnd = this->streamingContent.find(startTag, this->streamingPosition);
            std::string content = (contentEnd != std::string::npos)
                                      ? this->streamingContent.substr(this->streamingPosition, contentEnd - this->streamingPosition)
                                      : this->streamingContent.substr(this->streamingPosition);
            this->streamingPosition += content.size();
            cutEOSFromContent(content);
            if (!content.empty())
                return wrapDeltaContent(content);
            return std::nullopt;
        }
        case Lfm2ParseState::ToolCallStarted:
            if (!parseInToolCallState(this->streamingContent, this->toolCall, this->streamingPosition, this->currentState))
                return std::nullopt;
            ++this->toolCallIndex;
            return ToolCallDelta{this->toolCallIndex, generateRandomId(), this->toolCall.name, ""};
        case Lfm2ParseState::ToolCallParameters:
            if (!parseInToolCallParametersState(this->streamingContent, this->toolCall, this->streamingPosition, this->currentState)) {
                if (finishReason != ov::genai::GenerationFinishReason::NONE && !this->toolCall.arguments.empty())
                    return wrapDeltaArgs(this->toolCall.arguments, this->toolCallIndex);
                return std::nullopt;
            }
            {
                auto delta = wrapDeltaArgs(this->toolCall.arguments, this->toolCallIndex);
                this->toolCall = ToolCall{};
                return delta;
            }
        case Lfm2ParseState::ToolCallEnded:
            if (!parseInToolCallEndedState(this->streamingContent, this->streamingPosition, this->currentState, endTag)) {
                if (finishReason != ov::genai::GenerationFinishReason::NONE && !this->toolCall.arguments.empty())
                    return wrapDeltaArgs(this->toolCall.arguments, this->toolCallIndex);
                return std::nullopt;
            }
            if (this->currentState == Lfm2ParseState::AfterToolCall)
                this->currentState = Lfm2ParseState::Content;
            continue;
        case Lfm2ParseState::AfterToolCall:
            this->currentState = Lfm2ParseState::Content;
            continue;
        }
    }
    return std::nullopt;
}

}  // namespace ovms

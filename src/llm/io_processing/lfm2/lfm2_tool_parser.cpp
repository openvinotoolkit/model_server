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
#include "../qwen3coder/tool_schema_profile.hpp"

#include "../utils.hpp"
#include "../../../logging.hpp"
#include "../../../stringutils.hpp"

#include <cctype>
#include <stdexcept>
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

void Lfm2ToolParser::retainEnvelope(const std::string& body) {
    // Transactional across the complete native list: no public identity exists
    // until every call in the envelope has passed syntax and schema checks.
    std::vector<ToolCall> calls;
    size_t position = 0;
    const auto whitespace = [&]() {
        while (position < body.size() && std::isspace(static_cast<unsigned char>(body[position]))) ++position;
    };
    const auto require = [&](char expected) {
        whitespace();
        if (position >= body.size() || body[position++] != expected)
            throw std::invalid_argument("Invalid LFM tool envelope");
    };
    require('[');
    whitespace();
    if (position >= body.size() || body[position] == ']')
        throw std::invalid_argument("Empty LFM tool envelope");
    for (;;) {
        whitespace();
        const size_t nameStart = position;
        if (position >= body.size() || !((body[position] >= 'A' && body[position] <= 'Z') ||
                (body[position] >= 'a' && body[position] <= 'z') || body[position] == '_'))
            throw std::invalid_argument("Invalid LFM tool name");
        while (position < body.size() && ((body[position] >= 'A' && body[position] <= 'Z') ||
                   (body[position] >= 'a' && body[position] <= 'z') ||
                   (body[position] >= '0' && body[position] <= '9') || body[position] == '_')) ++position;
        const std::string name = body.substr(nameStart, position - nameStart);
        require('(');
        const size_t argsEnd = findArgumentListEnd(body, position);
        if (argsEnd == std::string::npos)
            throw std::invalid_argument("Incomplete LFM tool arguments");
        std::string arguments = lfm2::parseArguments(std::string_view(body).substr(position, argsEnd - position));
        const auto schema = registeredSchemas.find(name);
        if (schema == registeredSchemas.end() || !validateQwenXmlToolArguments(schema->second, arguments))
            throw std::invalid_argument("Unregistered LFM tool or invalid arguments");
        calls.push_back(ToolCall{"", name, std::move(arguments)});
        if (!parallelToolCalls && (calls.size() > 1 || toolCallIndex >= 0))
            throw std::invalid_argument("Parallel LFM tool calls are disabled");
        // Bound the entire generation, including published IDs and deltas a
        // caller has not drained, rather than only the current envelope.
        if (static_cast<size_t>(toolCallIndex + 1) + calls.size() > 4096)
            throw std::invalid_argument("LFM tool count exceeds limit");
        position = argsEnd + 1;
        whitespace();
        if (position < body.size() && body[position] == ']') {
            ++position;
            break;
        }
        require(',');
    }
    whitespace();
    if (position != body.size())
        throw std::invalid_argument("Trailing text in LFM tool envelope");
    for (auto& call : calls) {
        std::string id;
        do { id = generateRandomId(); } while (!publishedCallIds.insert(id).second);
        pendingDeltas.emplace_back(ToolCallDelta{++toolCallIndex, std::move(id), std::move(call.name), std::move(call.arguments)});
    }
}

std::optional<Delta> Lfm2ToolParser::parseRetainedChunk(const std::string& chunk,
    ov::genai::GenerationFinishReason finishReason) {
    static constexpr size_t maxRetainedBytes = 1024 * 1024;
    const std::string& startTag = parsingConfig.startTags.front();
    const std::string& endTag = parsingConfig.endTags.front();
    const bool finished = finishReason != ov::genai::GenerationFinishReason::NONE;
    if (terminal && !chunk.empty())
        throw std::invalid_argument("LFM output received after terminal notification");
    if (chunk.size() > maxRetainedBytes || streamingContent.size() > maxRetainedBytes - chunk.size()) {
        resetState();
        terminal = true;
        throw std::invalid_argument("LFM retained output exceeds byte limit");
    }
    streamingContent += chunk;
    try {
        while (!streamingContent.empty()) {
            if (!retainedEnvelope) {
                const size_t opener = streamingContent.find(startTag);
                if (opener == std::string::npos) {
                    // Retain a possible split opener, emit only proven content.
                    size_t suffix = 0;
                    if (!finished) {
                        for (size_t length = 1; length < startTag.size() && length <= streamingContent.size(); ++length)
                            if (streamingContent.compare(streamingContent.size() - length, length, startTag, 0, length) == 0) suffix = length;
                    }
                    std::string content = streamingContent.substr(0, streamingContent.size() - suffix);
                    streamingContent.erase(0, content.size());
                    cutEOSFromContent(content);
                    if (!content.empty()) pendingDeltas.emplace_back(ContentDelta{std::move(content)});
                    break;
                }
                if (opener != 0) {
                    std::string content = streamingContent.substr(0, opener);
                    cutEOSFromContent(content);
                    if (!content.empty()) pendingDeltas.emplace_back(ContentDelta{std::move(content)});
                }
                streamingContent.erase(0, opener);
                retainedEnvelope = true;
                retainedScanPosition = startTag.size();
                retainedQuote = 0;
                retainedEscape = false;
            }
            size_t closer = std::string::npos;
            // Scan each retained byte once, apart from the bounded delimiter
            // suffix. Large generated code strings must not cause a quadratic
            // rescan on every token.
            for (size_t& i = retainedScanPosition; i < streamingContent.size(); ++i) {
                const char value = streamingContent[i];
                if (retainedQuote) {
                    if (retainedEscape) retainedEscape = false;
                    else if (value == '\\') retainedEscape = true;
                    else if (value == retainedQuote) retainedQuote = 0;
                } else if (value == '\'' || value == '"') {
                    retainedQuote = value;
                } else if (streamingContent.compare(i, endTag.size(), endTag) == 0) {
                    closer = i;
                    break;
                } else if (streamingContent.size() - i < endTag.size() &&
                    endTag.compare(0, streamingContent.size() - i, streamingContent, i, streamingContent.size() - i) == 0) {
                    break;
                }
            }
            if (closer == std::string::npos) break;
            retainEnvelope(streamingContent.substr(startTag.size(), closer - startTag.size()));
            streamingContent.erase(0, closer + endTag.size());
            retainedEnvelope = false;
            retainedScanPosition = 0;
        }
    } catch (...) {
        // Nothing from the failing invocation can be executable, including an
        // earlier valid call in an otherwise malformed native list.
        pendingDeltas.clear();
        streamingContent.clear();
        retainedEnvelope = false;
        terminal = true;
        throw;
    }
    if (finished) {
        // STOP, LENGTH and TOOL_CALL do not repair missing semantic delimiters.
        // Cancellation/error use resetState at the owner; this API has no
        // distinct cancellation/error enum values.
        streamingContent.clear();
        retainedEnvelope = false;
        terminal = true;
    }
    if (pendingDeltas.empty()) return std::nullopt;
    Delta result = std::move(pendingDeltas.front());
    pendingDeltas.pop_front();
    return result;
}

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
    if (registryEnabled) return parseRetainedChunk(chunk, finishReason);
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

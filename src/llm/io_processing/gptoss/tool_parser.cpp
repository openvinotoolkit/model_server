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

#include <algorithm>
#include <openvino/genai/tokenizer.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <regex>

#include "../../../logging.hpp"
#include "../../../stringutils.hpp"
#include "tool_parser.hpp"
#include "harmony.hpp"
#include "../utils.hpp"

namespace ovms {

/*
    Prepares document with {"arguments": "escaped_chunk"}
    String gets escaped automatically by rapidjson
*/
std::optional<Delta> GptOssToolParser::wrapDeltaIntoDocument(const std::string& chunk) {
    return ToolCallDelta{toolCallIndex, std::nullopt, std::nullopt, chunk};
}

void GptOssToolParser::clearHeaderState() {
    cache.clear();
    isStreamingFunctionName = false;
    functionNameCache.clear();
}

// <|start|>assistant to=functions.foo <|constrain|>json<|message|>{...}<|call|>
// consumeToolCallStartTag -> consumeHeader -> consumeConstrain -> consumeMessage
std::optional<Delta> GptOssToolParser::parseChunk(const std::string& newChunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Tool | Processing Chunk [{}]", newChunk);

    std::string chunk = newChunk;
    std::optional<Delta> pendingDelta;

    for (;;) {
        const StepResult step = [&]() -> StepResult {
            switch (streamState) {
            case StreamState::READING_HEADER:
                if (consumeToolCallStartTag(chunk)) {
                    return continueParsing();
                }
                return consumeHeader(chunk, pendingDelta);
            case StreamState::READING_CONSTRAIN:
                return consumeConstrain(chunk, pendingDelta);
            case StreamState::READING_MESSAGE:
                return consumeMessage(chunk, pendingDelta);
            default:
                throw std::logic_error("Unexpected GPT-OSS tool parser state");
            }
        }();

        if (step.action == StepAction::CONTINUE) {
            continue;
        }
        if (step.action == StepAction::EMIT_DELTA) {
            return std::move(step.delta);
        }
        if (pendingDelta.has_value()) {
            return pendingDelta;
        }
        return std::nullopt;
    }
}

bool GptOssToolParser::consumeToolCallStartTag(std::string& chunk) {
    for (const auto& parsingStartTag : parsingConfig.startTags) {
        const std::size_t startPos = chunk.find(parsingStartTag);
        if (startPos != std::string::npos) {
            toolCallIndex++;  // starting with -1, first call will be 0
            streamState = StreamState::READING_HEADER;
            clearHeaderState();
            chunk = chunk.substr(startPos + parsingStartTag.size());
            return true;
        }
    }
    return false;
}

GptOssToolParser::StepResult GptOssToolParser::consumeHeader(std::string& chunk, std::optional<Delta>& pendingDelta) {
    const StreamState startingState = streamState;
    if (consumeCompleteHeader(chunk, pendingDelta)) {
        return pendingDelta.has_value() ? emitDelta(std::move(*pendingDelta)) : needMoreInput();
    }
    if (streamState != startingState) {
        return continueParsing();
    }
    if (consumeHeaderMarker(chunk, pendingDelta)) {
        return pendingDelta.has_value() ? emitDelta(std::move(*pendingDelta)) : needMoreInput();
    }
    if (streamState != startingState) {
        return continueParsing();
    }
    if (closeMessage(chunk, pendingDelta)) {
        return pendingDelta.has_value() ? emitDelta(std::move(*pendingDelta)) : needMoreInput();
    }
    if (streamState != startingState) {
        return continueParsing();
    }
    cache += chunk;
    return consumePartialHeader(std::move(chunk));
}

GptOssToolParser::StepResult GptOssToolParser::consumeConstrain(std::string& chunk, std::optional<Delta>& pendingDelta) {
    const StreamState startingState = streamState;
    if (consumeHeaderMarker(chunk, pendingDelta)) {
        return pendingDelta.has_value() ? emitDelta(std::move(*pendingDelta)) : needMoreInput();
    }
    if (streamState != startingState) {
        return continueParsing();
    }
    if (closeMessage(chunk, pendingDelta)) {
        return pendingDelta.has_value() ? emitDelta(std::move(*pendingDelta)) : needMoreInput();
    }
    if (streamState != startingState) {
        return continueParsing();
    }
    return needMoreInput();
}

GptOssToolParser::StepResult GptOssToolParser::consumeMessage(std::string& chunk, std::optional<Delta>& pendingDelta) {
    const StreamState startingState = streamState;
    if (closeMessage(chunk, pendingDelta)) {
        return pendingDelta.has_value() ? emitDelta(std::move(*pendingDelta)) : needMoreInput();
    }
    if (streamState != startingState) {
        return continueParsing();
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Tool | Sending Argument Part [{}]", chunk);
    const auto* toolDelta = pendingDelta.has_value() ? std::get_if<ToolCallDelta>(&*pendingDelta) : nullptr;
    if (toolDelta != nullptr && toolDelta->name.has_value()) {
        return emitDelta(ToolCallDelta{toolCallIndex, toolDelta->id, toolDelta->name, ""});
    }
    return emitDelta(*wrapDeltaIntoDocument(chunk));
}

GptOssToolParser::StepResult GptOssToolParser::consumePartialHeader(std::string chunk) {
    if (!isStreamingFunctionName && startsWith(cache, "functions.")) {
        isStreamingFunctionName = true;
        functionNameCache.clear();
        const std::size_t dotPos = chunk.find('.');
        if (dotPos != std::string::npos) {
            chunk = chunk.substr(dotPos + 1);
        }
    }

    if (isStreamingFunctionName) {
        const std::size_t spacePos = chunk.find(' ');
        if (spacePos != std::string::npos) {
            isStreamingFunctionName = false;
            chunk = chunk.substr(0, spacePos);
            cache.clear();
        }
        if (!chunk.empty()) {
            functionNameCache += chunk;
        }
    }
    return needMoreInput();
}

bool GptOssToolParser::consumeCompleteHeader(std::string& chunk, std::optional<Delta>& result) {
    const std::size_t constrainPos = chunk.find(openai::Harmony::TOKEN_CONSTRAIN);
    const std::size_t messagePos = chunk.find(openai::Harmony::TOKEN_MESSAGE);
    const std::size_t headerEnd = std::min(constrainPos, messagePos);
    const std::size_t functionPos = chunk.find("functions.");
    if (functionPos != std::string::npos && headerEnd != std::string::npos && functionPos < headerEnd) {
        const std::size_t nameStart = functionPos + std::string("functions.").size();
        const std::size_t nameEnd = chunk.find_first_of(" \t\n\r<", nameStart);
        functionNameCache = chunk.substr(nameStart, nameEnd == std::string::npos ? std::string::npos : nameEnd - nameStart);
        if (!functionNameCache.empty()) {
            result = ToolCallDelta{toolCallIndex, generateRandomId(), functionNameCache, ""};
        }
        cache.clear();
        isStreamingFunctionName = false;
        if (constrainPos != std::string::npos && constrainPos < messagePos) {
            streamState = StreamState::READING_CONSTRAIN;
            chunk = chunk.substr(constrainPos + openai::Harmony::TOKEN_CONSTRAIN.size());
        } else if (messagePos != std::string::npos) {
            streamState = StreamState::READING_MESSAGE;
            chunk = chunk.substr(messagePos + openai::Harmony::TOKEN_MESSAGE.size());
            if (!chunk.empty() && !result.has_value()) {
                result = wrapDeltaIntoDocument(chunk);
            }
            return true;
        }
    }
    return false;
}

bool GptOssToolParser::consumeHeaderMarker(std::string& chunk, std::optional<Delta>& result) {
    // This should only happen during header parsing if model does not produce garbage
    if (chunk == openai::Harmony::TOKEN_CONSTRAIN) {
        // If previous state was header, it means constrain was skipped
        // We can push function name in case there is some in cache
        if (streamState == StreamState::READING_HEADER) {
            if (functionNameCache.size()) {
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Tool | Sending Function Name [{}]", functionNameCache);
                result = ToolCallDelta{toolCallIndex, generateRandomId(), functionNameCache, ""};
            }
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Error: <|constrain|> appearance without previous <|channel|>, ignoring");
        }

        streamState = StreamState::READING_CONSTRAIN;
        clearHeaderState();
        return true;
    }

    // Message appears after channel and constrain, before actual message
    std::size_t pos = chunk.find(openai::Harmony::TOKEN_MESSAGE);
    if (pos != std::string::npos) {
        // If previous state was header, it means constrain was skipped
        // We can push function name in case there is some in cache
        if (streamState == StreamState::READING_HEADER) {
            if (functionNameCache.size()) {
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Tool | Sending Function Name [{}]", functionNameCache);
                result = ToolCallDelta{toolCallIndex, generateRandomId(), functionNameCache, ""};
            }
        }

        // StreamState::READING_CONSTRAIN implement here if required

        streamState = StreamState::READING_MESSAGE;
        clearHeaderState();

        if (chunk.size() > openai::Harmony::TOKEN_MESSAGE.size()) {
            // Move chunk pointer after message tag, continue with everything after message token
            chunk = chunk.substr(pos + openai::Harmony::TOKEN_MESSAGE.size());
        } else {
            // Entire chunk is only message tag, wait for next chunks in next iterations
            return true;
        }
    }
    return false;
}

bool GptOssToolParser::closeMessage(std::string& chunk, std::optional<Delta>& result) {
    const std::size_t callPos = chunk.find(openai::Harmony::TOKEN_CALL);
    const std::size_t endPos = chunk.find(openai::Harmony::TOKEN_END);
    const std::size_t returnPos = chunk.find(openai::Harmony::TOKEN_RETURN);
    const std::size_t terminatorPos = std::min({callPos, endPos, returnPos});
    if (terminatorPos != std::string::npos) {
        std::string clearedChunk = chunk.substr(0, terminatorPos);
        if (!clearedChunk.empty()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Tool | Sending Argument Part [{}]", clearedChunk);
            const auto* toolDelta = result.has_value() ? std::get_if<ToolCallDelta>(&*result) : nullptr;
            if (toolDelta != nullptr && toolDelta->name.has_value()) {
                result = ToolCallDelta{toolCallIndex, toolDelta->id, toolDelta->name, ""};
            } else {
                result = wrapDeltaIntoDocument(clearedChunk);
            }
        }

        streamState = StreamState::READING_HEADER;
        clearHeaderState();
        if (result.has_value()) {
            return true;
        }
        chunk.clear();
        return true;
    }
    return false;
}

const std::string GptOssToolParser::parsingStartTag = "<|channel|>commentary to=";
const std::string GptOssToolParser::parsingEndTag = "<|call|>";
}  // namespace ovms

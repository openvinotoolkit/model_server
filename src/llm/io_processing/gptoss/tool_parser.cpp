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

#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>
#include <regex>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../../../logging.hpp"
#include "../../../stringutils.hpp"
#include "tool_parser.hpp"
#include "harmony.hpp"
#include "../utils.hpp"

namespace ovms {

void GptToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    openai::Harmony harmony(tokenizer, generatedTokens);
    if (!harmony.parse()) {
        SPDLOG_LOGGER_INFO(llm_calculator_logger, "Harmony parsing failed");
        return;
    }

    // Yes, getContent is called twice, once in reasoning parser and once here, in tool parser.
    // This is because we have no guarantee that user will use both parsers, they might use only one of them.
    parsedOutput.content = harmony.getContent();
    parsedOutput.toolCalls = harmony.getToolCalls();
    for (const auto& toolCall : parsedOutput.toolCalls) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Unary | GPT Tool | id: [{}], name: [{}], arguments: [{}]", toolCall.id, toolCall.name, toolCall.arguments);
    }
}

/*
    Prepares document with {"arguments": "escaped_chunk"}
    String gets escaped automatically by rapidjson
*/
std::optional<rapidjson::Document> GptToolParser::wrapDeltaIntoDocument(const std::string& chunk) {
    rapidjson::Document newDelta;
    newDelta.SetObject();
    rapidjson::Value argumentsValue;
    argumentsValue.SetString(chunk.c_str(), static_cast<rapidjson::SizeType>(chunk.size()), newDelta.GetAllocator());
    newDelta.AddMember("arguments", argumentsValue, newDelta.GetAllocator());
    rapidjson::Document wrappedDelta;
    wrappedDelta.SetObject();
    rapidjson::Value toolCalls(rapidjson::kArrayType);
    rapidjson::Value toolCallObj(rapidjson::kObjectType);
    toolCallObj.AddMember("index", toolCallIndex, wrappedDelta.GetAllocator());
    rapidjson::Value functionObj(rapidjson::kObjectType);
    for (auto it = newDelta.MemberBegin(); it != newDelta.MemberEnd(); ++it) {
        rapidjson::Value key(it->name, wrappedDelta.GetAllocator());
        rapidjson::Value value(it->value, wrappedDelta.GetAllocator());
        functionObj.AddMember(key, value, wrappedDelta.GetAllocator());
    }
    toolCallObj.AddMember("function", functionObj, wrappedDelta.GetAllocator());
    toolCalls.PushBack(toolCallObj, wrappedDelta.GetAllocator());
    rapidjson::Value deltaWrapper(rapidjson::kObjectType);
    deltaWrapper.AddMember("tool_calls", toolCalls, wrappedDelta.GetAllocator());
    wrappedDelta.AddMember("delta", deltaWrapper, wrappedDelta.GetAllocator());
    return wrappedDelta;
}

void GptToolParser::clearState() {
    cache.clear();
    isStreamingFunctionName = false;
    functionNameCache.clear();
}

std::optional<rapidjson::Document> GptToolParser::parseChunk(const std::string& newChunk, ov::genai::GenerationFinishReason finishReason) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Tool | Processing Chunk [{}]", newChunk);

    std::string chunk = newChunk;
    std::optional<rapidjson::Document> result;

    if (chunk.find(getParsingStartTag()) != std::string::npos) {
        toolCallIndex++;  // starting with -1, first call will be 0
        return std::nullopt;
    }

    // This should only happen during channel read if model does not produce garbage
    if (chunk == openai::Harmony::TOKEN_CONSTRAIN) {
        // If previous state was channel, it means constrain was skipped
        // We can push function name in case there is some in cache
        if (streamState == StreamState::READING_CHANNEL) {
            if (functionNameCache.size()) {
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Tool | Sending Function Name [{}]", functionNameCache);
                result = wrapFirstDelta(functionNameCache, toolCallIndex);
            }
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Error: <|constrain|> appearance without previous <|channel|>, ignoring");
        }

        streamState = StreamState::READING_CONSTRAIN;
        clearState();
        return result;
    }

    // Message appears after channel and constrain, before actual message
    if (chunk == openai::Harmony::TOKEN_MESSAGE) {
        // If previous state was channel, it means constrain was skipped
        // We can push function name in case there is some in cache
        if (streamState == StreamState::READING_CHANNEL) {
            if (functionNameCache.size()) {
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Tool | Sending Function Name [{}]", functionNameCache);
                result = wrapFirstDelta(functionNameCache, toolCallIndex);
            }
        }
        // StreamState::READING_CONSTRAIN implement here if required

        streamState = StreamState::READING_MESSAGE;
        clearState();
        return result;
    }

    if (endsWith(chunk, openai::Harmony::TOKEN_CALL) || endsWith(chunk, openai::Harmony::TOKEN_END) || endsWith(chunk, openai::Harmony::TOKEN_RETURN)) {
        // find last <| and remove from chunk everything after it
        std::size_t pos = chunk.rfind("<|");
        if (pos != std::string::npos) {
            if (pos > 0) {
                std::string clearedChunk = chunk.substr(0, pos);
                if (!clearedChunk.empty()) {
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Tool | Sending Argument Part [{}]", clearedChunk);
                    result = wrapDeltaIntoDocument(clearedChunk);
                }
            }
        }

        streamState = StreamState::READING_CHANNEL;
        clearState();

        return result;
    }

    cache += chunk;

    switch (streamState) {
    case StreamState::READING_CHANNEL: {
        // Reading channel, but function name has not appeared yet
        if (!isStreamingFunctionName) {
            // Look ahead, and check if the function name reading state will begin now
            if (startsWith(cache, "functions.")) {
                isStreamingFunctionName = true;
                functionNameCache.clear();
                // Cut everything after first .
                // Remove and take only remaining part
                // The harmony format is: <|channel|>commentary to=functions.<function_name> <|constrain|>json<|message|>{...}<|call|>
                std::size_t pos = chunk.find('.');
                if (pos != std::string::npos) {
                    chunk = chunk.substr(pos + 1);
                }
            }
        }

        // If the function name reading state has begun, we are either reading function name or its end has been reached
        if (isStreamingFunctionName) {
            // Function names dont include space bars.
            // We can rely on this fact and simply decide if function name reading phase has finished.
            std::size_t pos = chunk.find(' ');
            if (pos != std::string::npos) {
                isStreamingFunctionName = false;
                chunk = chunk.substr(0, pos);
                cache.clear();
            }

            if (chunk.size()) {
                functionNameCache += chunk;
            }
        }
        break;
    }
    case StreamState::READING_CONSTRAIN:
        // Ignored, not needed for end user
        break;
    case StreamState::READING_MESSAGE: {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Streaming | GPT Tool | Sending Argument Part [{}]", chunk);
        return wrapDeltaIntoDocument(chunk);
    }
    }

    return std::nullopt;
}

const std::string GptToolParser::parsingStartTag = "<|channel|>commentary to=";
const std::string GptToolParser::parsingEndTag = "<|call|>";

}  // namespace ovms

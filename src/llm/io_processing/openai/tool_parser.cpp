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
    } else {
        //SPDLOG_LOGGER_INFO(llm_calculator_logger, "Parsed with harmony");
    }

    parsedOutput.content = harmony.getContent();  // what if someone has only reasoning parsers and no tool parser?
    parsedOutput.toolCalls = harmony.getToolCalls();
    for (auto& toolCall : parsedOutput.toolCalls) {
        SPDLOG_INFO("DEBUG Unary | GPT Tool Call | id: [{}], name: [{}], arguments: [{}]", toolCall.id, toolCall.name, toolCall.arguments);
    }
}

std::optional<rapidjson::Document> GptToolParser::wrapCustom(const std::string& chunk) {
    //SPDLOG_INFO("--------- OUT: [{}]", chunk);

    // prepare document with {"arguments": "escaped_chunk"}
    // It gets escaped automatically by rapidjson
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

std::optional<rapidjson::Document> GptToolParser::parseChunk(const std::string& c, ov::genai::GenerationFinishReason finishReason) {
    //SPDLOG_INFO("TOOL PARSER CHUNK [{}]", chunk);
    SPDLOG_INFO("DEBUG Streaming | GPT Tool | Chunk [{}]", c);

    std::string chunk = c;
    std::optional<rapidjson::Document> result;
    
    //if (chunk.find(getParsingStartTag()) != std::string::npos || chunk.find(getParsingEndTag()) != std::string::npos) {
    if (chunk.find(getParsingStartTag()) != std::string::npos) {
        toolCallIndex++;
        return std::nullopt;
    }

    
    if (chunk == "<|constrain|>") {
        if (streamState == StreamState::READING_CHANNEL) {
            //SPDLOG_INFO("CHANNEL READ COMPLETE [{}]", cache); 
            if (functionNameCache.size()) {
                SPDLOG_INFO("DEBUG Streaming | GPT Tool | Send Function Name [{}]", functionNameCache);
                result = wrapFirstDelta(functionNameCache, toolCallIndex);
            }
            cache.clear();
        }

        streamState = StreamState::READING_CONSTRAIN;
        isStreamingFunctionName = false;
        return result;
    }

    if (chunk == "<|message|>") {
        if (streamState == StreamState::READING_CHANNEL) {
            //SPDLOG_INFO("CHANNEL READ COMPLETE [{}]", cache);
            if (functionNameCache.size()) {
                SPDLOG_INFO("DEBUG Streaming | GPT Tool | Send Function Name [{}]", functionNameCache);
                result = wrapFirstDelta(functionNameCache, toolCallIndex);
            }
            cache.clear();
        }
        if (streamState == StreamState::READING_CONSTRAIN) {
            //SPDLOG_INFO("CONSTRAIN READ COMPLETE [{}]", cache);
            cache.clear();
        }

        streamState = StreamState::READING_MESSAGE;
        isStreamingFunctionName = false;
        return result;
    }
    
    if (endsWith(chunk, "<|call|>") || endsWith(chunk, "<|end|>") || endsWith(chunk, "<|return|>")) {
        // find last <| and remove from chunk everything after it
        std::size_t pos = chunk.rfind("<|");
        if (pos != std::string::npos) {
            if (pos > 0) {
                std::string toAdd = chunk.substr(0, pos);
                if (!toAdd.empty()) {
                    cache += toAdd;
                    //SPDLOG_INFO("READING MESSAGE STEP [{}]", toAdd);  //// ZZZZZZZZ
                    SPDLOG_INFO("DEBUG Streaming | GPT Tool | Send [{}]", toAdd);

                    result = wrapCustom(toAdd);
                }
            }
        }

        //SPDLOG_INFO("MESSAGE READ COMPLETE [{}]", cache);
        cache.clear();
        streamState = StreamState::READING_CHANNEL;
        isStreamingFunctionName = false;

        // print the json
        // use buffer writer
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        result.value().Accept(writer);
        //SPDLOG_INFO("WRAPPED DELTA [{}]", buffer.GetString());

        return result;
    }
    
    cache += chunk;

    switch (streamState) {
    case StreamState::READING_CHANNEL:
    {
        if (!isStreamingFunctionName) {
            std::string futureCache = cache + chunk;
            if (startsWith(futureCache, " to=functions.")) {
                isStreamingFunctionName = true;
                functionNameCache.clear();
                // cut everything after first .
                // remove and take only remaining part
                std::size_t pos = chunk.find('.');
                if (pos != std::string::npos) {
                    chunk = chunk.substr(pos + 1);
                }
            }
        }
        if (isStreamingFunctionName) {
            // search for space bar and cut everything after it, flag streaming function name end
            // dont include space bar
            std::size_t pos = chunk.find(' ');
            if (pos != std::string::npos) {
                isStreamingFunctionName = false;
                chunk = chunk.substr(0, pos);
                cache.clear();
            }

            if (chunk.size()) {
                //SPDLOG_INFO("FUNCTION NAME STEP [{}]", chunk);  // XXXXXXXXXXX
                functionNameCache += chunk;
            }
        } else {
            //SPDLOG_INFO("READING CHANNEL STEP [{}]", chunk);  /// XXXXXXXXXXX
        }
        break;
    }
    case StreamState::READING_CONSTRAIN:
        //SPDLOG_INFO("READING CONSTRAIN STEP [{}]", chunk);  // YYYYYYYYYY
        break;
    case StreamState::READING_MESSAGE:
    {
        //SPDLOG_INFO("READING MESSAGE STEP [{}]", chunk); /// ZZZZZZZZZZ
        SPDLOG_INFO("DEBUG Streaming | GPT Tool | Send [{}]", chunk);
        return wrapCustom(chunk);

    }
    }

    return std::nullopt;
}
}  // namespace ovms

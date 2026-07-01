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
#include <utility>

#include "src/port/rapidjson_document.hpp"

#include "../../../logging.hpp"
#include "tool_parser.hpp"
#include "../utils.hpp"
#include "src/stringutils.hpp"

namespace ovms {

void Llama3ToolParser::startNextToolCall() {
    lastJson.Clear();
    jsonBuilder.clear();
    toolCallIndex++;
    argumentsDelayWindow[0].clear();
    argumentsDelayWindow[1].clear();
}

static inline bool jsonHasArgumentsOrParameters(const rapidjson::Document& json) {
    return json.HasMember("arguments") || json.HasMember("parameters");
}

static inline void changeParametersToArguments(rapidjson::Document& json) {
    if (json.HasMember("parameters")) {
        // change key to "arguments"
        json.AddMember("arguments", json["parameters"], json.GetAllocator());
        json.RemoveMember("parameters");
    }
}

std::optional<rapidjson::Document> Llama3ToolParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason finishReason) {
    const bool hasPendingState =
        !argumentsDelayWindow[0].empty() ||
        !argumentsDelayWindow[1].empty() ||
        jsonHasArgumentsOrParameters(lastJson);
    if (chunk.empty() && !hasPendingState) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for Llama3ToolParser");
        return std::nullopt;
    }

    // <|python_tag|> boundary text (synthesised by OutputParser on token-ID detection)
    if (chunk.find("<|python_tag|>") != std::string::npos) {
        this->startNextToolCall();
        return std::nullopt;  // ignoring the special tag
    }

    // -1 means not started, we need to start it
    if (toolCallIndex < 0) {
        this->startNextToolCall();
    }

    // Cases to handle:
    // <|python_tag|>{ ... parameters ... } ; { ... parameters ... }
    // <|python_tag|>{ ... arguments ... } ; { ... arguments ... }
    // { ... parameters ... } ; { ... parameters ... }
    // { ... arguments ... } ; { ... arguments ... }

    bool isCurrentToolCallParsingFinished = false;

    // JSON already contains 'parameters'/'arguments' (they cannot be null at this point). Apply modifications to the input chunk if needed to keep the format valid.
    if (jsonHasArgumentsOrParameters(lastJson)) {
        std::string modifiedChunk = chunk;
        // Since inside a string, we need to escape characters like quotes, new lines, tabs, etc.
        escapeSpecialCharacters(modifiedChunk);

        // Handle the case when we are starting to collect parameters/arguments.
        // Force parameters/arguments string type and fill first element of the delay array.
        if (argumentsDelayWindow[0].empty()) {
            // If we are starting to collect parameters/arguments, we add opening quote before the first non-whitespace character
            size_t firstNonWhitespaceCharacter = modifiedChunk.find_first_not_of(" \t\n\r\f\v");
            if (firstNonWhitespaceCharacter != std::string::npos) {
                modifiedChunk.insert(firstNonWhitespaceCharacter, "\"");
            } else {
                // If the chunk is all whitespace, just insert closing quote at the end
                modifiedChunk.append("\"");
            }
            argumentsDelayWindow[0] = modifiedChunk;
            return std::nullopt;  // We don't return anything yet, we need to collect next chunk
        }

        if (!argumentsDelayWindow[1].empty()) {
            // We already have two chunks, so we can move delay window forward
            argumentsDelayWindow[0] = argumentsDelayWindow[1];
        }

        // If this is end of streaming, we need to manually add closing quote "
        // We need to place it right before last closing brace
        if (finishReason != ov::genai::GenerationFinishReason::NONE) {
            isCurrentToolCallParsingFinished = true;
            if (modifiedChunk.empty()) {
                // Empty STOP flush from streamer: finalize the delayed chunk in-place.
                size_t lastClosingBrace = argumentsDelayWindow[0].find_last_of('}');
                if (lastClosingBrace != std::string::npos) {
                    argumentsDelayWindow[0].insert(lastClosingBrace, "\"");
                }
            } else {
                size_t lastClosingBrace = modifiedChunk.find_last_of('}');
                if (lastClosingBrace != std::string::npos) {
                    modifiedChunk.insert(lastClosingBrace, "\"");
                }
                argumentsDelayWindow[0] += modifiedChunk;
            }
            // If this is end of one of the tool calls "in the middle" (; has been found), we need to manually add closing quote "
            // We need to place it right before last closing brace
        } else if (modifiedChunk.find(separator) != std::string::npos) {
            isCurrentToolCallParsingFinished = true;
            size_t lastClosingBrace = argumentsDelayWindow[0].find_last_of('}');
            if (lastClosingBrace != std::string::npos) {
                argumentsDelayWindow[0].insert(lastClosingBrace, "\"");
            }
        } else {
            argumentsDelayWindow[1] = modifiedChunk;
        }
    }

    rapidjson::Document newJson;
    // Push delayed chunk to the JSON builder
    try {
        if (!argumentsDelayWindow[0].empty()) {
            // Push delayed chunk to the JSON builder if we are processing parameters
            newJson = jsonBuilder.add(argumentsDelayWindow[0]);
        } else {
            // Otherwise just push the current chunk
            newJson = jsonBuilder.add(chunk);
        }
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call chunk partial parse failed: {}", e.what());
        // Throwing an error since at this point the JSON is broken and next chunks will not make it right.
        throw std::runtime_error("Generated tool call structure is not valid");  // re-throw
    }

    rapidjson::Document doc;
    // Case 1: 'parameters'/'arguments' has just appeared in the current chunk. If so, we return first delta.
    if (jsonHasArgumentsOrParameters(newJson) && !jsonHasArgumentsOrParameters(lastJson)) {
        std::string functionName;
        changeParametersToArguments(newJson);
        if (lastJson.HasMember("name") && lastJson["name"].IsString()) {
            functionName = lastJson["name"].GetString();
        } else if (newJson.HasMember("name") && newJson["name"].IsString()) {
            // We received big chunk with both full function name and parameters, so we get function name from the new JSON
            functionName = newJson["name"].GetString();
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool call name has not been generated and parameters already started");
            throw std::runtime_error("Tool call name is missing in generated output");
        }
        // Wrap first delta in {"tool_calls":[{"id":<id>,"type":"function","index":<toolCallIndex>,"function":{"name": <functionName>}}]}
        doc = wrapFirstDelta(functionName, toolCallIndex);
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
        return doc;
        // Case 2: 'parameters' already exists in the last JSON, we compute delta and return it.
    } else if (lastJson.HasMember("arguments") || lastJson.HasMember("parameters")) {
        changeParametersToArguments(newJson);
        rapidjson::Document delta = PartialJsonBuilder::computeDelta(lastJson, newJson);
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
        // If delta is empty or contains only null or empty string values, we don't stream anything.
        if (delta.ObjectEmpty()) {
            return std::nullopt;
        }
        for (auto it = delta.MemberBegin(); it != delta.MemberEnd(); ++it) {
            if (it->value.IsNull() || (it->value.IsString() && std::string(it->value.GetString()).empty())) {
                return std::nullopt;
            }
        }
        // Wrap delta in {"tool_calls":[{"index":<toolCallIndex>,"function":<delta>}]}
        doc = wrapDelta(delta, toolCallIndex);
        if (isCurrentToolCallParsingFinished) {
            this->startNextToolCall();
        }
        return doc;
        // Case 3: No 'parameters' exists or just appeared, so we keep building up until we have complete function name
    } else {
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
    }
    return std::nullopt;
}

}  // namespace ovms

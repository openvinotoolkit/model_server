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

#include <string>
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/port/rapidjson_document.hpp"
#include "src/logging.hpp"
#include "src/llm/io_processing/utils.hpp"
#include "src/stringutils.hpp"
#include "tool_parser.hpp"

namespace ovms {

// Static tag definitions using Unicode characters:
//   U+FF5C  ｜  FULLWIDTH VERTICAL LINE
//   U+2581  ▁  LOWER ONE EIGHTH BLOCK
const std::string DeepSeekToolParser::toolCallsBeginTag = "<\xEF\xBD\x9C" "tool\xE2\x96\x81" "calls\xE2\x96\x81" "begin\xEF\xBD\x9C>";
const std::string DeepSeekToolParser::toolCallBeginTag = "<\xEF\xBD\x9C" "tool\xE2\x96\x81" "call\xE2\x96\x81" "begin\xEF\xBD\x9C>";
const std::string DeepSeekToolParser::toolSepTag = "<\xEF\xBD\x9C" "tool\xE2\x96\x81" "sep\xEF\xBD\x9C>";
const std::string DeepSeekToolParser::toolCallEndTag = "<\xEF\xBD\x9C" "tool\xE2\x96\x81" "call\xE2\x96\x81" "end\xEF\xBD\x9C>";
const std::string DeepSeekToolParser::toolCallsEndTag = "<\xEF\xBD\x9C" "tool\xE2\x96\x81" "calls\xE2\x96\x81" "end\xEF\xBD\x9C>";
const std::string DeepSeekToolParser::parsingStartTag = DeepSeekToolParser::toolCallsBeginTag;
const std::string DeepSeekToolParser::parsingEndTag = DeepSeekToolParser::toolCallsEndTag;

std::string DeepSeekToolParser::extractArguments(const std::string& raw) {
    // The template wraps arguments in markdown code block:
    //   ```json\n{...}\n```
    // But the model may also produce raw JSON without the code block wrapper.
    std::string trimmed = raw;
    ovms::trim(trimmed);

    // Try to extract from markdown code block
    static const std::string jsonBlockStart = "```json\n";
    static const std::string blockEnd = "\n```";

    size_t jsonStart = trimmed.find(jsonBlockStart);
    if (jsonStart != std::string::npos) {
        jsonStart += jsonBlockStart.size();
        size_t jsonEnd = trimmed.find(blockEnd, jsonStart);
        if (jsonEnd != std::string::npos) {
            return trimmed.substr(jsonStart, jsonEnd - jsonStart);
        }
        // No closing ```, take everything after opening
        return trimmed.substr(jsonStart);
    }

    // Try plain ``` block (no "json" language tag)
    static const std::string plainBlockStart = "```\n";
    jsonStart = trimmed.find(plainBlockStart);
    if (jsonStart != std::string::npos) {
        jsonStart += plainBlockStart.size();
        size_t jsonEnd = trimmed.find(blockEnd, jsonStart);
        if (jsonEnd != std::string::npos) {
            return trimmed.substr(jsonStart, jsonEnd - jsonStart);
        }
        return trimmed.substr(jsonStart);
    }

    // No code block wrapper — assume raw JSON
    if (trimmed.empty()) {
        return "{}";
    }
    return trimmed;
}

void DeepSeekToolParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    // Expected format (may have multiple tool calls):
    //   content<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNC_NAME
    //   ```json
    //   {"param": "value"}
    //   ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    //
    // Tags are regular text, not special tokens.

    const std::string& content = parsedOutput.content;
    size_t callsBeginPos = content.find(toolCallsBeginTag);
    if (callsBeginPos == std::string::npos) {
        // No tool calls found — check for individual tool call tag without outer wrapper
        callsBeginPos = content.find(toolCallBeginTag);
        if (callsBeginPos == std::string::npos) {
            return;
        }
    } else {
        callsBeginPos += toolCallsBeginTag.size();
    }

    // Save content before tool calls
    std::string contentBeforeToolCalls;
    size_t firstTagPos = content.find(toolCallsBeginTag);
    if (firstTagPos == std::string::npos) {
        firstTagPos = content.find(toolCallBeginTag);
    }
    if (firstTagPos != std::string::npos && firstTagPos > 0) {
        contentBeforeToolCalls = content.substr(0, firstTagPos);
    }

    // Parse individual tool calls
    std::string searchArea = content.substr(callsBeginPos);
    size_t pos = 0;
    while (pos < searchArea.size()) {
        // Find next tool call begin tag
        size_t tcBegin = searchArea.find(toolCallBeginTag, pos);
        if (tcBegin == std::string::npos) {
            break;
        }
        tcBegin += toolCallBeginTag.size();

        // Find tool separator
        size_t sepPos = searchArea.find(toolSepTag, tcBegin);
        if (sepPos == std::string::npos) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "DeepSeek tool call missing separator tag");
            break;
        }

        // Skip "function" keyword between tool_call_begin and tool_sep
        size_t afterSep = sepPos + toolSepTag.size();

        // Find tool call end tag
        size_t tcEnd = searchArea.find(toolCallEndTag, afterSep);
        std::string toolCallContent;
        if (tcEnd != std::string::npos) {
            toolCallContent = searchArea.substr(afterSep, tcEnd - afterSep);
            pos = tcEnd + toolCallEndTag.size();
        } else {
            toolCallContent = searchArea.substr(afterSep);
            pos = searchArea.size();
        }

        // toolCallContent is: "FUNC_NAME\n```json\n{...}\n```" or "FUNC_NAME{...}"
        // Split at first newline to get function name and arguments
        ToolCall toolCall;
        size_t newlinePos = toolCallContent.find('\n');
        if (newlinePos != std::string::npos) {
            toolCall.name = toolCallContent.substr(0, newlinePos);
            std::string argsRaw = toolCallContent.substr(newlinePos + 1);
            toolCall.arguments = extractArguments(argsRaw);
        } else {
            // No newline - might be compact format: FUNC_NAME{...}
            size_t bracePos = toolCallContent.find('{');
            if (bracePos != std::string::npos) {
                toolCall.name = toolCallContent.substr(0, bracePos);
                toolCall.arguments = toolCallContent.substr(bracePos);
            } else {
                toolCall.name = toolCallContent;
                toolCall.arguments = "{}";
            }
        }
        ovms::trim(toolCall.name);
        toolCall.id = generateRandomId();
        parsedOutput.toolCalls.push_back(toolCall);
    }

    // Remove tool calls from content
    parsedOutput.content = contentBeforeToolCalls;
}

std::optional<rapidjson::Document> DeepSeekToolParser::parseChunk(const std::string& chunk, ov::genai::GenerationFinishReason finishReason) {
    /*
    Streaming format:
        <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNC_NAME
        ```json
        {"param": "value"}
        ```<｜tool▁call▁end｜>[<｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNC2...|<｜tool▁calls▁end｜>]

    States:
        AWAITING_SEP       - stripping prefix tags, waiting for tool_sep
        AWAITING_FUNC_NAME - found sep, waiting for newline to get function name
        PROCESSING_ARGS    - accumulating argument chunks and streaming them
    */

    streamContent += chunk;
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "DeepSeek chunk: '{}', accumulated: '{}', state: {}", chunk, streamContent, static_cast<int>(internalState));

    if (internalState == AWAITING_SEP) {
        // Strip outer calls begin tag if present
        size_t callsBeginPos = streamContent.find(toolCallsBeginTag);
        if (callsBeginPos != std::string::npos) {
            streamContent = streamContent.substr(callsBeginPos + toolCallsBeginTag.size());
        }

        // Strip tool call begin tag if present
        size_t tcBeginPos = streamContent.find(toolCallBeginTag);
        if (tcBeginPos != std::string::npos) {
            streamContent = streamContent.substr(tcBeginPos + toolCallBeginTag.size());
        }

        // Look for the separator tag
        size_t sepPos = streamContent.find(toolSepTag);
        if (sepPos == std::string::npos) {
            return std::nullopt;  // Wait for more data
        }

        // Skip "function" keyword and separator
        streamContent = streamContent.substr(sepPos + toolSepTag.size());
        internalState = AWAITING_FUNC_NAME;
        // Fall through to check if we already have the function name in this chunk
    }

    if (internalState == AWAITING_FUNC_NAME) {
        // Function name ends at the first newline
        size_t newlinePos = streamContent.find('\n');
        if (newlinePos == std::string::npos) {
            return std::nullopt;  // Wait for more data to get complete function name
        }

        toolName = streamContent.substr(0, newlinePos);
        ovms::trim(toolName);
        streamContent = streamContent.substr(newlinePos + 1);
        toolCallIndex++;
        internalState = PROCESSING_ARGS;

        return wrapFirstDelta(toolName, toolCallIndex);
    }

    if (internalState == PROCESSING_ARGS) {
        // Check for tool call end tag
        size_t endPos = streamContent.find(toolCallEndTag);
        bool toolCallEnded = (endPos != std::string::npos) || (finishReason != ov::genai::GenerationFinishReason::NONE);

        if (!toolCallEnded) {
            // Keep accumulating until we see the end tag or generation finishes
            return std::nullopt;
        }

        std::string arguments;
        if (endPos != std::string::npos) {
            arguments = streamContent.substr(0, endPos);
            std::string remaining = streamContent.substr(endPos + toolCallEndTag.size());
            streamContent.clear();
            // After completing a tool call, transition to AWAITING_SEP for potential next tool call
            internalState = AWAITING_SEP;

            if (!remaining.empty()) {
                streamContent = remaining;
            }
        } else {
            arguments = streamContent;
            streamContent.clear();
        }

        arguments = extractArguments(arguments);

        if (arguments.empty()) {
            arguments = "{}";
        }

        // Send arguments delta
        rapidjson::Document argumentsWrapper;
        argumentsWrapper.SetObject();
        rapidjson::Document::AllocatorType& allocator = argumentsWrapper.GetAllocator();
        rapidjson::Value argsValue(rapidjson::kStringType);
        argsValue.SetString(arguments.c_str(), allocator);
        argumentsWrapper.AddMember("arguments", argsValue, allocator);
        return wrapDelta(argumentsWrapper, toolCallIndex);
    }

    return std::nullopt;
}

}  // namespace ovms

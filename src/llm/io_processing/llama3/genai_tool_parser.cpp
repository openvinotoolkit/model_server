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

#include "genai_tool_parser.hpp"

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/genai/json_container.hpp>

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

#include "../../../logging.hpp"
#include "../base_output_parser.hpp"
#include "../utils.hpp"
#include "src/stringutils.hpp"

namespace ovms {

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string rapidjsonToString(const rapidjson::Document& doc) {
    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
    doc.Accept(writer);
    return sb.GetString();
}

static bool jsonHasArgumentsOrParameters(const rapidjson::Document& json) {
    return json.HasMember("arguments") || json.HasMember("parameters");
}

static void changeParametersToArguments(rapidjson::Document& json) {
    if (json.HasMember("parameters")) {
        json.AddMember("arguments", json["parameters"], json.GetAllocator());
        json.RemoveMember("parameters");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama3GenAIToolParser  (unary)
// ─────────────────────────────────────────────────────────────────────────────

void Llama3GenAIToolParser::parse(ov::genai::JsonContainer& message) {
    if (!message.contains("content") || !message["content"].is_string()) {
        return;
    }

    std::string content = message["content"].get_string();
    std::string toolCallsText;

    const std::string botTagStr(botTag);
    const size_t botPos = content.find(botTagStr);
    if (botPos != std::string::npos) {
        toolCallsText = content.substr(botPos + botTagStr.size());
        content = content.substr(0, botPos);
    } else if (!content.empty() && content[0] == '{') {
        // Model output starts with "{" — treat the whole content as tool calls
        toolCallsText = content;
        content.clear();
    } else {
        return;  // No tool calls present
    }

    // Split tool call JSON blobs by ";" separator
    std::vector<std::string> toolJsonStrings;
    const std::string sep(separator);
    size_t start = 0;
    size_t end;
    while ((end = toolCallsText.find(sep, start)) != std::string::npos) {
        std::string part = toolCallsText.substr(start, end - start);
        if (!part.empty()) {
            toolJsonStrings.push_back(std::move(part));
        }
        start = end + sep.size();
    }
    std::string lastPart = toolCallsText.substr(start);
    if (!lastPart.empty()) {
        toolJsonStrings.push_back(std::move(lastPart));
    }

    ov::genai::JsonContainer toolCallsArr = ov::genai::JsonContainer::array();

    for (const std::string& toolJson : toolJsonStrings) {
        rapidjson::Document toolDoc;
        toolDoc.Parse(toolJson.c_str());
        if (toolDoc.HasParseError()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Llama3GenAIToolParser: failed to parse tool call JSON");
            continue;
        }

        if (!toolDoc.HasMember("name") || !toolDoc["name"].IsString()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Llama3GenAIToolParser: tool call missing valid 'name' field");
            continue;
        }
        const std::string name = toolDoc["name"].GetString();

        const char* argsKey = nullptr;
        if (toolDoc.HasMember("arguments")) {
            argsKey = "arguments";
        } else if (toolDoc.HasMember("parameters")) {
            argsKey = "parameters";
        }
        if (!argsKey) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Llama3GenAIToolParser: tool call missing 'parameters'/'arguments' field");
            continue;
        }

        // Serialize the arguments/parameters object to a JSON string
        rapidjson::StringBuffer sb;
        rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
        toolDoc[argsKey].Accept(writer);
        const std::string arguments = sb.GetString();

        ov::genai::JsonContainer fn = ov::genai::JsonContainer::object();
        fn["name"] = name;
        fn["arguments"] = arguments;

        ov::genai::JsonContainer tc = ov::genai::JsonContainer::object();
        tc["id"] = generateRandomId();
        tc["type"] = "function";
        tc["function"] = fn;

        toolCallsArr.push_back(tc);
    }

    message["content"] = content;
    if (toolCallsArr.size() > 0) {
        message["tool_calls"] = toolCallsArr;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama3GenAIIncrementalToolParser  (streaming)
// ─────────────────────────────────────────────────────────────────────────────

void Llama3GenAIIncrementalToolParser::startNextToolCall() {
    lastJson.Clear();
    jsonBuilder.clear();
    toolCallIndex++;
    argumentsDelayWindow[0].clear();
    argumentsDelayWindow[1].clear();
}

void Llama3GenAIIncrementalToolParser::reset() {
    lastJson.Clear();
    jsonBuilder.clear();
    toolCallIndex = -1;
    argumentsDelayWindow[0].clear();
    argumentsDelayWindow[1].clear();
    escapeLevel = 0;
}

void Llama3GenAIIncrementalToolParser::flush(ov::genai::JsonContainer& delta_message) {
    if (toolCallIndex < 0 || argumentsDelayWindow[1].empty()) {
        return;
    }

    // Drain the last pending chunk from the delay window with the same
    // closing-quote logic that normally fires when ";" is detected.
    std::string lastChunk = argumentsDelayWindow[1];
    argumentsDelayWindow[1].clear();

    const size_t lastBrace = lastChunk.find_last_of('}');
    if (lastBrace != std::string::npos) {
        lastChunk.insert(lastBrace, "\"");
    } else {
        lastChunk += "\"";
    }

    rapidjson::Document newJson;
    try {
        newJson = jsonBuilder.add(lastChunk);
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger,
            "Llama3GenAIIncrementalToolParser::flush: chunk parse failed: {}", e.what());
        return;
    }

    if (!lastJson.HasMember("arguments") && !lastJson.HasMember("parameters")) {
        return;
    }

    changeParametersToArguments(newJson);
    rapidjson::Document delta = PartialJsonBuilder::computeDelta(lastJson, newJson);
    lastJson.CopyFrom(newJson, lastJson.GetAllocator());

    if (delta.ObjectEmpty()) {
        return;
    }
    for (auto it = delta.MemberBegin(); it != delta.MemberEnd(); ++it) {
        if (it->value.IsNull() ||
            (it->value.IsString() && std::string(it->value.GetString()).empty())) {
            return;
        }
    }

    const rapidjson::Document doc = BaseOutputParser::wrapDelta(delta, toolCallIndex);
    delta_message = ov::genai::JsonContainer::from_json_string(rapidjsonToString(doc));
}

std::string Llama3GenAIIncrementalToolParser::parse(
    ov::genai::JsonContainer& delta_message,
    std::string& delta_text,
    const std::optional<std::vector<int64_t>>& /*delta_tokens*/) {

    std::string chunk = delta_text;

    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Llama3GenAIIncrementalToolParser: received empty chunk");
        return "";
    }

    // <|python_tag|> signals start of tool calls — consume it, start first tool call
    if (chunk.find(botTag) != std::string::npos) {
        startNextToolCall();
        return "";
    }

    // If not yet started, start implicitly (bare "{" path)
    if (toolCallIndex < 0) {
        startNextToolCall();
    }

    bool isCurrentToolCallParsingFinished = false;

    if (jsonHasArgumentsOrParameters(lastJson)) {
        std::string modifiedChunk = chunk;
        escapeSpecialCharacters(modifiedChunk);

        // Starting to collect arguments — force string type by injecting opening quote
        if (argumentsDelayWindow[0].empty()) {
            const size_t firstNonWs = modifiedChunk.find_first_not_of(" \t\n\r\f\v");
            if (firstNonWs != std::string::npos) {
                modifiedChunk.insert(firstNonWs, "\"");
            } else {
                modifiedChunk.append("\"");
            }
            argumentsDelayWindow[0] = modifiedChunk;
            return "";
        }

        if (!argumentsDelayWindow[1].empty()) {
            argumentsDelayWindow[0] = argumentsDelayWindow[1];
        }

        // ";" detected: end of this tool call — close the arguments string
        if (modifiedChunk.find(separator) != std::string::npos) {
            isCurrentToolCallParsingFinished = true;
            const size_t lastBrace = argumentsDelayWindow[0].find_last_of('}');
            if (lastBrace != std::string::npos) {
                argumentsDelayWindow[0].insert(lastBrace, "\"");
            }
        } else {
            argumentsDelayWindow[1] = modifiedChunk;
        }
    }

    rapidjson::Document newJson;
    try {
        if (!argumentsDelayWindow[0].empty()) {
            newJson = jsonBuilder.add(argumentsDelayWindow[0]);
        } else {
            newJson = jsonBuilder.add(chunk);
        }
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Llama3GenAIIncrementalToolParser: chunk parse failed: {}", e.what());
        throw std::runtime_error("Generated tool call structure is not valid");
    }

    // Case 1: arguments/parameters just appeared — emit first delta with function name
    if (jsonHasArgumentsOrParameters(newJson) && !jsonHasArgumentsOrParameters(lastJson)) {
        std::string functionName;
        changeParametersToArguments(newJson);
        if (lastJson.HasMember("name") && lastJson["name"].IsString()) {
            functionName = lastJson["name"].GetString();
        } else if (newJson.HasMember("name") && newJson["name"].IsString()) {
            functionName = newJson["name"].GetString();
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Llama3GenAIIncrementalToolParser: function name missing when arguments appeared");
            throw std::runtime_error("Tool call name is missing in generated output");
        }
        const rapidjson::Document doc = BaseOutputParser::wrapFirstDelta(functionName, toolCallIndex);
        delta_message = ov::genai::JsonContainer::from_json_string(rapidjsonToString(doc));
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
        return "";

        // Case 2: arguments already present — compute and emit incremental delta
    } else if (lastJson.HasMember("arguments") || lastJson.HasMember("parameters")) {
        changeParametersToArguments(newJson);
        rapidjson::Document delta = PartialJsonBuilder::computeDelta(lastJson, newJson);
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());

        if (delta.ObjectEmpty()) {
            return "";
        }
        for (auto it = delta.MemberBegin(); it != delta.MemberEnd(); ++it) {
            if (it->value.IsNull() || (it->value.IsString() && std::string(it->value.GetString()).empty())) {
                return "";
            }
        }

        const rapidjson::Document doc = BaseOutputParser::wrapDelta(delta, toolCallIndex);
        delta_message = ov::genai::JsonContainer::from_json_string(rapidjsonToString(doc));

        if (isCurrentToolCallParsingFinished) {
            startNextToolCall();
        }
        return "";

        // Case 3: still accumulating function name — keep building
    } else {
        lastJson.CopyFrom(newJson, lastJson.GetAllocator());
    }

    return "";
}

}  // namespace ovms

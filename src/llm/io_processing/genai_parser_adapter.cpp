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

#include "genai_parser_adapter.hpp"

#include <string>
#include <utility>
#include <vector>

namespace ovms {

GenAIParserAdapter::GenAIParserAdapter(
    ov::genai::Tokenizer& tokenizer,
    std::shared_ptr<BaseGenAIParser> parser,
    std::shared_ptr<BaseGenAIIncrementalParser> incrementalParser) :
    BaseOutputParser(tokenizer),
    parser(std::move(parser)),
    incrementalParser(std::move(incrementalParser)) {}

// static
void GenAIParserAdapter::extractToolCalls(const ov::genai::JsonContainer& message, ToolCalls_t& toolCalls) {
    if (!message.contains("tool_calls") || !message["tool_calls"].is_array()) {
        return;
    }
    const ov::genai::JsonContainer toolCallsArr = message["tool_calls"];
    const size_t count = toolCallsArr.size();
    for (size_t i = 0; i < count; ++i) {
        const ov::genai::JsonContainer tc = toolCallsArr[i];
        ToolCall toolCall;
        if (tc.contains("id") && tc["id"].is_string()) {
            toolCall.id = tc["id"].get_string();
        }
        if (tc.contains("function") && tc["function"].is_object()) {
            const ov::genai::JsonContainer fn = tc["function"];
            if (fn.contains("name") && fn["name"].is_string()) {
                toolCall.name = fn["name"].get_string();
            }
            if (fn.contains("arguments") && fn["arguments"].is_string()) {
                toolCall.arguments = fn["arguments"].get_string();
            }
        }
        toolCalls.push_back(std::move(toolCall));
    }
}

// static
rapidjson::Document GenAIParserAdapter::jsonContainerToDocument(const ov::genai::JsonContainer& container) {
    const std::string jsonStr = container.to_json_string();
    rapidjson::Document doc;
    doc.Parse(jsonStr.c_str(), static_cast<rapidjson::SizeType>(jsonStr.size()));
    return doc;
}

void GenAIParserAdapter::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& /*generatedTokens*/) {
    ov::genai::JsonContainer message({{"content", parsedOutput.content}});

    parser->parse(message);

    if (message.contains("content") && message["content"].is_string()) {
        parsedOutput.content = message["content"].get_string();
    }

    extractToolCalls(message, parsedOutput.toolCalls);

    if (message.contains("reasoning_content") && message["reasoning_content"].is_string()) {
        parsedOutput.reasoning = message["reasoning_content"].get_string();
    }
}

std::optional<rapidjson::Document> GenAIParserAdapter::parseChunk(
    const std::string& chunkResponse,
    ov::genai::GenerationFinishReason finishReason) {

    ov::genai::JsonContainer deltaMessage = ov::genai::JsonContainer::object();
    std::string deltaText = chunkResponse;

    const std::string filteredContent = incrementalParser->parse(deltaMessage, deltaText);

    // On the final chunk, drain any content still buffered in the parser's delay window.
    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        ov::genai::JsonContainer flushedDelta = ov::genai::JsonContainer::object();
        incrementalParser->flush(flushedDelta);

        if (!flushedDelta.empty()) {
            rapidjson::Document flushedDoc = jsonContainerToDocument(flushedDelta);
            if (!flushedDoc.HasParseError() && flushedDoc.IsObject() && !flushedDoc.ObjectEmpty()) {
                // If parse() also produced a structured delta, combine argument strings
                // so a single document is returned.
                if (!deltaMessage.empty()) {
                    rapidjson::Document primaryDoc = jsonContainerToDocument(deltaMessage);
                    if (!primaryDoc.HasParseError() && primaryDoc.IsObject() && !primaryDoc.ObjectEmpty()) {
                        return combineArgumentDeltas(std::move(primaryDoc), std::move(flushedDoc));
                    }
                }
                return flushedDoc;
            }
        }
    }

    // Prefer structured deltas (tool calls, reasoning) if the incremental parser emitted any
    if (!deltaMessage.empty()) {
        rapidjson::Document doc = jsonContainerToDocument(deltaMessage);
        if (!doc.HasParseError() && doc.IsObject() && !doc.ObjectEmpty()) {
            return doc;
        }
    }

    // Fall back to plain content delta
    if (!filteredContent.empty()) {
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Value contentVal(
            filteredContent.c_str(),
            static_cast<rapidjson::SizeType>(filteredContent.size()),
            doc.GetAllocator());
        doc.AddMember("content", contentVal, doc.GetAllocator());
        return doc;
    }

    return std::nullopt;
}

// static
rapidjson::Document GenAIParserAdapter::combineArgumentDeltas(
    rapidjson::Document primary,
    rapidjson::Document flushed) {
    // Navigate primary: delta.tool_calls[0].function.arguments
    if (!primary.HasMember("delta") || !flushed.HasMember("delta")) {
        return primary;
    }
    auto& pDelta = primary["delta"];
    auto& fDelta = flushed["delta"];
    if (!pDelta.IsObject() || !pDelta.HasMember("tool_calls") ||
        !fDelta.IsObject() || !fDelta.HasMember("tool_calls")) {
        return primary;
    }
    auto& pTC = pDelta["tool_calls"];
    auto& fTC = fDelta["tool_calls"];
    if (!pTC.IsArray() || pTC.Empty() || !fTC.IsArray() || fTC.Empty()) {
        return primary;
    }
    auto& pEntry = pTC[rapidjson::SizeType(0)];
    auto& fEntry = fTC[rapidjson::SizeType(0)];
    if (!pEntry.IsObject() || !pEntry.HasMember("function") ||
        !fEntry.IsObject() || !fEntry.HasMember("function")) {
        return primary;
    }
    auto& pFunc = pEntry["function"];
    auto& fFunc = fEntry["function"];
    if (!pFunc.IsObject() || !pFunc.HasMember("arguments") ||
        !fFunc.IsObject() || !fFunc.HasMember("arguments")) {
        return primary;
    }
    if (!pFunc["arguments"].IsString() || !fFunc["arguments"].IsString()) {
        return primary;
    }
    const std::string combined =
        std::string(pFunc["arguments"].GetString()) +
        std::string(fFunc["arguments"].GetString());
    pFunc["arguments"].SetString(
        combined.c_str(),
        static_cast<rapidjson::SizeType>(combined.size()),
        primary.GetAllocator());
    return primary;
}

const std::vector<std::string>& GenAIParserAdapter::getParsingStartTags() const {
    return parser->getParsingStartTags();
}

const std::vector<std::string>& GenAIParserAdapter::getSpecialParsingStartTags() const {
    return parser->getSpecialParsingStartTags();
}

const std::string& GenAIParserAdapter::getParsingEndTag() const {
    return parser->getParsingEndTag();
}

bool GenAIParserAdapter::requiresStreamingWithSpecialTokens() const {
    return parser->requiresStreamingWithSpecialTokens() ||
           incrementalParser->requiresStreamingWithSpecialTokens();
}

}  // namespace ovms

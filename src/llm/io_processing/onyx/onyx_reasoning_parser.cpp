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

#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "src/logging.hpp"
#include "onyx_reasoning_parser.hpp"

namespace ovms {

void OnyxReasoningParser::parse(ParsedOutput& parsedOutput, const std::vector<int64_t>& generatedTokens) {
    // TODO @atobiszei overcomplicated I think? We just need t find recipient self & them <eom> cut that part out.
    // Case 1: private chain-of-thought turn (recipient="self") -> extract reasoning,
    // consume the whole segment (nothing meaningful is expected to follow it within the
    // same generate() call, see class comment).
    size_t selfPos = parsedOutput.content.find(selfRecipientTag);
    if (selfPos != std::string::npos) {
        size_t messagePos = parsedOutput.content.find(messageTag, selfPos);
        if (messagePos != std::string::npos) {
            size_t bodyStart = messagePos + messageTag.length();
            size_t endPos = parsedOutput.content.find(continuationEndTag, bodyStart);
            std::string body = (endPos != std::string::npos)
                                    ? parsedOutput.content.substr(bodyStart, endPos - bodyStart)
                                    : parsedOutput.content.substr(bodyStart);
            parsedOutput.reasoning = body;
            // Drop the leading " " before "to=" (rendered by the chat template) too, if present.
            size_t segmentStart = (selfPos > 0 && parsedOutput.content[selfPos - 1] == ' ') ? selfPos - 1 : selfPos;
            parsedOutput.content.erase(segmentStart);
            return;
        }
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Found '{}' without a following '{}', leaving content untouched", selfRecipientTag, messageTag);
        return;
    }

    // Case 2: tool-call turn (recipient="functions.<name>") -> leave untouched, OnyxToolParser
    // (which runs after this parser, see OutputParser::parse()) is responsible for it.
    if (parsedOutput.content.find(functionsRecipientTag) != std::string::npos) {
        return;
    }

    // Case 3: plain final answer (recipient="user" or absent) -> strip the generic
    // " to=user"? + "<|message|>" + "<|eot|>" envelope, leaving just the clean text.
    size_t messagePos = parsedOutput.content.find(messageTag);
    if (messagePos == std::string::npos) {
        // No framing found at all -- unexpected/malformed output, leave content as-is.
        return;
    }
    size_t bodyStart = messagePos + messageTag.length();
    size_t endPos = parsedOutput.content.find(turnFinalEndTag, bodyStart);
    std::string body = (endPos != std::string::npos)
                            ? parsedOutput.content.substr(bodyStart, endPos - bodyStart)
                            : parsedOutput.content.substr(bodyStart);
    parsedOutput.content = body;
}

std::optional<rapidjson::Document> OnyxReasoningParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason /*finishReason*/) {
    // TODO @atobiszei we need to stream between recipient=self & <eom>
    // TODO: streaming support is a first draft. It only forwards the chunk as
    // reasoning_content once we've seen the "to=self" start tag; stripping the
    // generic final-answer envelope (Case 3 above) in streaming mode is not
    // implemented yet and needs its own design (the envelope prefix/suffix can
    // straddle multiple chunks).
    if (chunk.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Received empty chunk for OnyxReasoningParser");
        return std::nullopt;
    }
    if (chunk.find(selfRecipientTag) != std::string::npos ||
        chunk.find(messageTag) != std::string::npos ||
        chunk.find(continuationEndTag) != std::string::npos) {
        return std::nullopt;
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.String("delta");
    writer.StartObject();
    writer.String("reasoning_content");
    writer.String(chunk.c_str());
    writer.EndObject();
    writer.EndObject();
    rapidjson::Document doc;
    doc.Parse(buffer.GetString());
    return doc;
}
}  // namespace ovms

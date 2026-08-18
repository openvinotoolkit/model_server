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
    // Step 1: Extract and remove ALL "to=self<|message|>...<|eom|>" reasoning segments.
    for (;;) {
        size_t selfPos = parsedOutput.content.find(selfRecipientTag);
        if (selfPos == std::string::npos)
            break;
        size_t messagePos = parsedOutput.content.find(messageTag, selfPos);
        if (messagePos == std::string::npos) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Found '{}' without a following '{}', leaving content untouched", selfRecipientTag, messageTag);
            break;
        }
        size_t bodyStart = messagePos + messageTag.length();
        size_t endPos = parsedOutput.content.find(continuationEndTag, bodyStart);
        std::string reasoning = (endPos != std::string::npos)
                                    ? parsedOutput.content.substr(bodyStart, endPos - bodyStart)
                                    : parsedOutput.content.substr(bodyStart);
        if (!parsedOutput.reasoning.empty())
            parsedOutput.reasoning += '\n';
        parsedOutput.reasoning += reasoning;
        // Erase the segment including the leading space before "to=" if present.
        size_t segmentStart = (selfPos > 0 && parsedOutput.content[selfPos - 1] == ' ') ? selfPos - 1 : selfPos;
        size_t eraseEnd = (endPos != std::string::npos) ? endPos + continuationEndTag.length() : parsedOutput.content.length();
        parsedOutput.content.erase(segmentStart, eraseEnd - segmentStart);
    }

    // Step 2: Remove all "<|start|>assistant" turn boundary markers (with optional trailing space).
    static const std::string turnBoundary = "<|start|>assistant";
    for (;;) {
        size_t pos = parsedOutput.content.find(turnBoundary);
        if (pos == std::string::npos)
            break;
        size_t eraseLen = turnBoundary.length();
        // Also consume one trailing space if present (before "to=").
        if (pos + eraseLen < parsedOutput.content.length() && parsedOutput.content[pos + eraseLen] == ' ')
            ++eraseLen;
        parsedOutput.content.erase(pos, eraseLen);
    }

    // Step 3: Strip envelope framing from remaining turns. Each non-self turn has
    // " to=<recipient><|message|>" before its body. Find each "<|message|>" tag, look
    // backwards for the closest "to=" prefix, and erase the envelope (including a
    // leading space if present). This preserves content between tool-call turns.
    static const std::string toPrefix = "to=";
    // The envelope (" to=<name><|message|>") is never longer than this.
    static constexpr size_t maxEnvelopeLen = 128;
    for (;;) {
        size_t messagePos = parsedOutput.content.find(messageTag);
        if (messagePos == std::string::npos)
            break;
        // Bound the backwards search to avoid matching "to=" in body content.
        size_t searchFrom = (messagePos > maxEnvelopeLen) ? messagePos - maxEnvelopeLen : 0;
        size_t toPos = parsedOutput.content.rfind(toPrefix, messagePos);
        size_t eraseStart;
        if (toPos != std::string::npos && toPos >= searchFrom && parsedOutput.content.find(messageTag, toPos) == messagePos) {
            // Include the leading space before "to=" if present.
            eraseStart = (toPos > 0 && parsedOutput.content[toPos - 1] == ' ') ? toPos - 1 : toPos;
        } else {
            // No "to=" found within the envelope window; erase just the tag itself.
            eraseStart = messagePos;
        }
        parsedOutput.content.erase(eraseStart, messagePos + messageTag.length() - eraseStart);
    }

    // Step 4: Remove all remaining terminators.
    for (const auto& term : {continuationEndTag, turnFinalEndTag}) {
        for (;;) {
            size_t pos = parsedOutput.content.find(term);
            if (pos == std::string::npos)
                break;
            parsedOutput.content.erase(pos, term.length());
        }
    }
}

std::optional<rapidjson::Document> OnyxReasoningParser::parseChunk(const std::string& chunk, const std::vector<int64_t>& /*tokens*/, ov::genai::GenerationFinishReason /*finishReason*/) {
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

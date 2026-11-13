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

#include "openai_transcriptions.hpp"

#include <string>
#include <string_view>
#include <vector>

#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

namespace ovms {

namespace {

std::string serializeTranscriptionResponse(const std::string& text) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("text");
    writer.String(text.c_str());
    writer.EndObject();
    return buffer.GetString();
}

}  // namespace

absl::Status OpenAITranscriptionsHandler::parseRequestImpl(std::optional<uint32_t> maxTokensLimit,
    uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength,
    std::optional<std::string>, std::optional<std::vector<std::string>>) {
    auto status = parseCommonPart(maxTokensLimit, bestOfLimit, maxModelLength);
    if (!status.ok()) {
        return status;
    }
    request.stream = false;
    const auto prompt = doc.FindMember("prompt");
    if (prompt != doc.MemberEnd()) {
        if (!prompt->value.IsString()) {
            return absl::InvalidArgumentError("prompt must be a string");
        }
        request.chatHistory.push_back({
            {"role", "system"},
            {"content", std::string(prompt->value.GetString(), prompt->value.GetStringLength())},
        });
    }
    request.chatHistory.push_back({
        {"role", "user"},
        {"content", ov::genai::JsonContainer::from_json_string(R"([{"type":"audio"}])")},
    });
    return absl::OkStatus();
}

std::string OpenAITranscriptionsHandler::transcriptionText(const std::vector<Delta>& deltas) {
    const std::string generatedText = parsedOutputFromDeltas(deltas).content;
    constexpr std::string_view asrTextMarker = "<asr_text>";
    const size_t markerPosition = generatedText.find(asrTextMarker);
    return markerPosition == std::string::npos ? generatedText : generatedText.substr(markerPosition + asrTextMarker.size());
}

std::string OpenAITranscriptionsHandler::serializeUnaryResponse(const std::vector<Delta>& deltas,
    ov::genai::GenerationFinishReason) {
    return serializeTranscriptionResponse(transcriptionText(deltas));
}

std::string OpenAITranscriptionsHandler::serializeUnaryResponse(const std::vector<std::vector<Delta>>& allDeltas,
    const std::vector<ov::genai::GenerationFinishReason>& finishReasons,
    const std::vector<UnaryChoiceLogprobs>&) {
    static const std::vector<Delta> emptyDeltas;
    return serializeUnaryResponse(allDeltas.empty() ? emptyDeltas : allDeltas.front(),
        finishReasons.empty() ? ov::genai::GenerationFinishReason::STOP : finishReasons.front());
}

std::string OpenAITranscriptionsHandler::serializeStreamingChunk(Delta delta,
    ov::genai::GenerationFinishReason) {
    return "";
}

std::string OpenAITranscriptionsHandler::serializeStreamingUsageChunk() {
    return "";
}

std::string OpenAITranscriptionsHandler::serializeStreamingHandshakeChunk() {
    return "";
}

}  // namespace ovms

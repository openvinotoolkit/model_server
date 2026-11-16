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

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/http_payload.hpp"
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

std::string serializeStreamingTranscriptionDelta(const std::string& delta) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.String("type");
    writer.String("transcript.text.delta");
    writer.String("delta");
    writer.String(delta.c_str());
    writer.String("logprobs");
    writer.StartArray();
    writer.EndArray();
    writer.EndObject();
    return buffer.GetString();
}

std::string serializeStreamingTranscriptionDone(const std::string& text) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.String("type");
    writer.String("transcript.text.done");
    writer.String("text");
    writer.String(text.c_str());
    writer.String("logprobs");
    writer.StartArray();
    writer.EndArray();
    writer.EndObject();
    return buffer.GetString();
}

}  // namespace

absl::Status OpenAITranscriptionsHandler::normalizeMultipartRequest(HttpPayload& payload) {
    if (!payload.multipartParser || payload.multipartParser->hasParseError()) {
        return absl::InvalidArgumentError("Failed to parse multipart data");
    }

    const std::string model = payload.multipartParser->getFieldByName("model");
    if (model.empty()) {
        return absl::InvalidArgumentError("model is missing");
    }
    if (payload.multipartParser->getFileContentByFieldName("file").empty()) {
        return absl::InvalidArgumentError("file is missing");
    }

    auto request = std::make_shared<rapidjson::Document>();
    request->SetObject();
    auto& allocator = request->GetAllocator();
    request->AddMember("model", rapidjson::Value(model.c_str(), allocator), allocator);

    const std::string prompt = payload.multipartParser->getFieldByName("prompt");
    if (!prompt.empty()) {
        request->AddMember("prompt", rapidjson::Value(prompt.c_str(), allocator), allocator);
    }

    const std::string stream = payload.multipartParser->getFieldByName("stream");
    if (!stream.empty()) {
        if (stream == "true") {
            request->AddMember("stream", true, allocator);
        } else if (stream == "false") {
            request->AddMember("stream", false, allocator);
        } else {
            return absl::InvalidArgumentError("stream must be true or false");
        }
    }

    rapidjson::Value audioPart(rapidjson::kObjectType);
    audioPart.AddMember("type", "audio", allocator);

    rapidjson::Value content(rapidjson::kArrayType);
    content.PushBack(audioPart, allocator);
    rapidjson::Value message(rapidjson::kObjectType);
    message.AddMember("role", "user", allocator);
    message.AddMember("content", content, allocator);
    rapidjson::Value messages(rapidjson::kArrayType);
    messages.PushBack(message, allocator);
    request->AddMember("messages", messages, allocator);

    const std::string temperature = payload.multipartParser->getFieldByName("temperature");
    if (!temperature.empty()) {
        char* parseEnd = nullptr;
        const float parsedTemperature = std::strtof(temperature.c_str(), &parseEnd);
        if (parseEnd == temperature.c_str() || *parseEnd != '\0' ||
            !std::isfinite(parsedTemperature) || parsedTemperature < 0.0f || parsedTemperature > 2.0f) {
            return absl::InvalidArgumentError("temperature must be finite and in range [0.0, 2.0]");
        }
        request->AddMember("temperature", parsedTemperature, allocator);
    }

    payload.parsedJson = std::move(request);
    return absl::OkStatus();
}

absl::Status OpenAITranscriptionsHandler::parseRequestImpl(std::optional<uint32_t> maxTokensLimit,
    uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength,
    std::optional<std::string>, std::optional<std::vector<std::string>>) {
    auto status = parseCommonPart(maxTokensLimit, bestOfLimit, maxModelLength);
    if (!status.ok()) {
        return status;
    }
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
    return parsedOutputFromDeltas(deltas).content;
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
    ov::genai::GenerationFinishReason finishReason) {
    if (const auto* contentDelta = std::get_if<ContentDelta>(&delta)) {
        streamingText += contentDelta->text;
        return serializeStreamingTranscriptionDelta(contentDelta->text);
    }
    if (finishReason != ov::genai::GenerationFinishReason::NONE) {
        return serializeStreamingTranscriptionDone(streamingText);
    }
    return {};
}

std::string OpenAITranscriptionsHandler::serializeStreamingUsageChunk() {
    return {};
}

std::string OpenAITranscriptionsHandler::serializeStreamingHandshakeChunk() {
    return {};
}

}  // namespace ovms

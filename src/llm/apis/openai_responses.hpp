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
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "openai_api_handler.hpp"

namespace ovms {

// Encapsulates all mutable state accumulated during Responses API streaming.
struct ResponsesStreamingState {
    size_t sequenceNumber = 1;
    bool createdSent = false;
    bool inProgressSent = false;
    bool reasoningInitialized = false;
    bool reasoningCompleted = false;
    bool messageInitialized = false;
    std::string outputText;
    std::string reasoningText;
    ToolCalls_t toolCalls;
};

// Handler for OpenAI Responses API (/v3/responses).
// Implements Responses-specific request parsing and response serialization.
class OpenAIResponsesHandler : public OpenAIApiHandler {
    ResponsesStreamingState responsesState;

    // Responses-specific request parsing
    absl::Status parseInput(std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains);
    absl::Status parseResponsesPart(std::optional<uint32_t> maxTokensLimit, std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains);

    // Responses serialization helpers
    void serializeToolChoice(Writer<StringBuffer>& writer) const;
    void serializeTools(Writer<StringBuffer>& writer) const;
    void serializeCommonResponseParameters(Writer<StringBuffer>& writer, const std::string& responseId, int64_t createdAt,
        const std::string& status,
        const std::optional<std::string>& incompleteReason = std::nullopt, const std::optional<std::string>& errorMessage = std::nullopt, ResponsesErrorCode errorCode = ResponsesErrorCode::SERVER_ERROR) const;
    void serializeResponseObject(Writer<StringBuffer>& writer, const std::string& responseId, int64_t createdAt,
        const std::string& status, const std::string& fullOutputText, bool includeUsage,
        const std::optional<std::string>& incompleteReason = std::nullopt, const std::optional<std::string>& errorMessage = std::nullopt, ResponsesErrorCode errorCode = ResponsesErrorCode::SERVER_ERROR) const;
    static void serializeOutputItem(Writer<StringBuffer>& writer, const std::string& outputItemId,
        const std::string& text, const std::string& status);
    static void serializeTextPart(Writer<StringBuffer>& writer, const std::string& text);
    std::string serializeUnaryResponseImpl(const std::vector<ParsedOutput>& parsedOutputs,
        ov::genai::GenerationFinishReason finishReason = ov::genai::GenerationFinishReason::STOP) const;

    // Streaming event building blocks
    void writeEventHeader(Writer<StringBuffer>& writer, const char* eventType);
    static void writeContentLocation(Writer<StringBuffer>& writer, const std::string& itemId, uint64_t outputIndex = 0);
    static void writeReasoningLocation(Writer<StringBuffer>& writer, const std::string& itemId);

    // Individual streaming event serializers
    std::string serializeOutputItemAddedEvent(const std::string& outputItemId, uint64_t outputIndex = 0);
    std::string serializeContentPartAddedEvent(const std::string& outputItemId, uint64_t outputIndex = 0);
    std::string serializeOutputTextDeltaEvent(const std::string& outputItemId, const std::string& delta, uint64_t outputIndex = 0);
    std::string serializeOutputTextDoneEvent(const std::string& outputItemId, uint64_t outputIndex = 0);
    std::string serializeContentPartDoneEvent(const std::string& outputItemId, uint64_t outputIndex = 0);
    std::string serializeOutputItemDoneEvent(const std::string& outputItemId, ov::genai::GenerationFinishReason finishReason, uint64_t outputIndex = 0);
    std::string serializeCompletedEvent(const std::string& responseId, int64_t createdAt, ov::genai::GenerationFinishReason finishReason);
    std::string serializeFailedEventBody(const std::string& responseId, int64_t createdAt, const std::string& errorMessage, ResponsesErrorCode errorCode);

    // Reasoning streaming event serializers
    std::string serializeReasoningOutputItemAddedEvent(const std::string& reasoningItemId);
    std::string serializeReasoningSummaryPartAddedEvent(const std::string& reasoningItemId);
    std::string serializeReasoningSummaryTextDeltaEvent(const std::string& reasoningItemId, const std::string& delta);
    std::string serializeReasoningSummaryTextDoneEvent(const std::string& reasoningItemId);
    std::string serializeReasoningSummaryPartDoneEvent(const std::string& reasoningItemId);
    std::string serializeReasoningOutputItemDoneEvent(const std::string& reasoningItemId);

    // Function call streaming event serializers
    std::string serializeFunctionCallOutputItemAddedEvent(const ToolCall& toolCall, uint64_t outputIndex);
    std::string serializeFunctionCallArgumentsDeltaEvent(const std::string& callId, const std::string& delta, uint64_t outputIndex);
    std::string serializeFunctionCallArgumentsDoneEvent(const ToolCall& toolCall, uint64_t outputIndex);
    std::string serializeFunctionCallOutputItemDoneEvent(const ToolCall& toolCall, ov::genai::GenerationFinishReason finishReason, uint64_t outputIndex);

public:
    using OpenAIApiHandler::OpenAIApiHandler;  // Inherit constructors

    absl::Status parseRequest(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength,
        std::optional<std::string> allowedLocalMediaPath = std::nullopt, std::optional<std::vector<std::string>> allowedMediaDomains = std::nullopt) override;

    std::string serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs) override;
    std::string serializeUnaryResponse(ov::genai::EncodedResults& results) override;
    std::string serializeUnaryResponse(ov::genai::VLMDecodedResults& results) override;
    std::string serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason) override;
    std::string serializeStreamingUsageChunk() override;
    std::string serializeStreamingHandshakeChunk() override;
    std::string serializeStreamingCreatedEvent() override;
    std::string serializeStreamingInProgressEvent() override;
    std::string serializeFailedEvent(const std::string& errorMessage, ResponsesErrorCode errorCode = ResponsesErrorCode::SERVER_ERROR) override;
};

}  // namespace ovms

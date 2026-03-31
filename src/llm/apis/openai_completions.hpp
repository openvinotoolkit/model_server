//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include <vector>

#include "openai_api_handler.hpp"

namespace ovms {

// Handler for OpenAI Completions (/v3/completions) and Chat Completions (/v3/chat/completions) APIs.
class OpenAIChatCompletionsHandler : public OpenAIApiHandler {
    bool toolCallsDetectedInStream = false;  // tracks whether tool calls were detected in any streaming chunk
    size_t processedTokens = 0;              // tracks overall number of tokens processed by the pipeline (echo-aware)

    absl::Status parseCompletionsPart();
    absl::Status parseChatCompletionsPart(std::optional<uint32_t> maxTokensLimit, std::optional<std::string> allowedLocalMediaPath, std::optional<std::vector<std::string>> allowedMediaDomains);

public:
    using OpenAIApiHandler::OpenAIApiHandler;  // Inherit constructors

    absl::Status parseRequest(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength,
        std::optional<std::string> allowedLocalMediaPath = std::nullopt, std::optional<std::vector<std::string>> allowedMediaDomains = std::nullopt) override;
    absl::Status parseMessages(std::optional<std::string> allowedLocalMediaPath = std::nullopt, std::optional<std::vector<std::string>> allowedMediaDomains = std::nullopt);

    std::string serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs) override;
    std::string serializeUnaryResponse(ov::genai::EncodedResults& results) override;
    std::string serializeUnaryResponse(ov::genai::VLMDecodedResults& results) override;
    std::string serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason) override;
    std::string serializeStreamingUsageChunk() override;
    std::string serializeStreamingHandshakeChunk() override;
    void incrementProcessedTokens(size_t numTokens = 1) override;
};
}  // namespace ovms

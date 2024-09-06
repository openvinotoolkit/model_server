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

#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/generation_handle.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <rapidjson/document.h>

#include "absl/status/status.h"

using namespace rapidjson;

namespace ovms {

struct StreamOptions {
    bool includeUsage = false;
};

using chat_entry_t = std::unordered_map<std::string, std::string>;
using chat_t = std::vector<chat_entry_t>;

#define IGNORE_EOS_MAX_TOKENS_LIMIT 4000

enum class Endpoint {
    CHAT_COMPLETIONS,
    COMPLETIONS,
};

struct CompletionUsageStatistics {
    size_t promptTokens = 0;
    size_t completionTokens = 0;

    size_t calculateTotalTokens() const {
        return promptTokens + completionTokens;
    }
};

struct OpenAIChatCompletionsRequest;

// Class that wraps OpenAI request, holds and processes raw JSON, provides methods for serialization and keeps track of usage.
// It is used in the calculator.
class OpenAIChatCompletionsHandler {
    Document& doc;
    Endpoint endpoint;
    CompletionUsageStatistics usage;
    std::unique_ptr<OpenAIChatCompletionsRequest> request{nullptr};
    std::chrono::time_point<std::chrono::system_clock> created;
    ov::genai::Tokenizer tokenizer;

    absl::Status parseCompletionsPart();
    absl::Status parseChatCompletionsPart();
    absl::Status parseCommonPart(uint32_t maxTokensLimit, uint32_t bestOfLimit);

public:
    OpenAIChatCompletionsHandler(Document& doc, Endpoint endpoint, std::chrono::time_point<std::chrono::system_clock> creationTime,
        ov::genai::Tokenizer tokenizer);

    ~OpenAIChatCompletionsHandler();

    std::optional<std::string> getPrompt() const;
    std::optional<int> getNumReturnSequences() const;
    StreamOptions getStreamOptions() const;

    bool isStream() const;
    std::string getModel() const;

    void setPromptTokensUsage(int promptTokens);

    void incrementCompletionTokensUsage();

    ov::genai::GenerationConfig createGenerationConfig() const;

    absl::Status parseRequest(uint32_t maxTokensLimit, uint32_t bestOfLimit);

    std::string serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs);
    std::string serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason);
    std::string serializeStreamingUsageChunk();
};
}  // namespace ovms

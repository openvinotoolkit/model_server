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

#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/generation_handle.hpp>
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

// Class that maps OpenAI request content and provides methods to create GenerationConfig from it.
// It's supposed to be used only by OpenAIChatCompletionsHandler class
struct OpenAIChatCompletionsRequest {
    chat_t messages;
    std::optional<std::string> prompt{std::nullopt};
    bool stream{false};
    StreamOptions streamOptions;
    std::string model;
    std::optional<int> maxTokens{std::nullopt};
    std::optional<float> frequencyPenalty{std::nullopt};
    std::optional<float> presencePenalty{std::nullopt};
    std::optional<float> diversityPenalty{std::nullopt};
    std::optional<float> repetitionPenalty{std::nullopt};
    std::optional<float> lengthPenalty{std::nullopt};
    std::optional<int> numReturnSequences{std::nullopt};
    std::optional<float> temperature{std::nullopt};
    std::optional<float> topP{std::nullopt};
    std::optional<int> topK{std::nullopt};
    std::optional<int> seed{std::nullopt};
    std::optional<std::set<std::string>> stop{std::nullopt};
    std::optional<bool> includeStopStrInOutput{std::nullopt};
    std::optional<int> bestOf{std::nullopt};
    std::optional<bool> ignoreEOS{std::nullopt};

    OpenAIChatCompletionsRequest() = default;

    ov::genai::GenerationConfig createGenerationConfig() const;
};

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

// Class that wraps OpenAI request, holds and processes raw JSON, provides methods for serialization and keeps track of usage.
// It is used in the calculator.
class OpenAIChatCompletionsHandler {
    Document& doc;
    Endpoint endpoint;
    CompletionUsageStatistics usage;
    OpenAIChatCompletionsRequest request;
    std::chrono::time_point<std::chrono::system_clock> created;

    absl::Status processCompletionsPart();
    absl::Status processChatCompletionsPart();
    absl::Status processCommonPart(uint32_t maxTokensLimit, uint32_t bestOfLimit);

public:
    OpenAIChatCompletionsHandler(Document& doc, Endpoint endpoint, std::chrono::time_point<std::chrono::system_clock> creationTime) :
        doc(doc),
        endpoint(endpoint),
        created(creationTime) {}

    std::optional<std::string> getPrompt() const;
    std::optional<int> getNumReturnSequences() const;
    StreamOptions getStreamOptions() const;

    bool isStream() const;
    std::string getModel() const;

    void setPromptTokensUsage(int promptTokens);

    void incrementCompletionTokensUsage();

    ov::genai::GenerationConfig createGenerationConfig() const;

    absl::Status processRequest(uint32_t maxTokensLimit, uint32_t bestOfLimit);

    std::string serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs, ov::genai::Tokenizer tokenizer);
    std::string serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason);
    std::string serializeStreamingUsageChunk();
};
}  // namespace ovms

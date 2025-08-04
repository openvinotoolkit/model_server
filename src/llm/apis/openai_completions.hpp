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

#include <limits>
#include <optional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/generation_handle.hpp>
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#pragma warning(pop)
#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/status/status.h"
#pragma warning(pop)
#include "../io_processing/output_parser.hpp"
#include "openai_request.hpp"

using namespace rapidjson;

namespace ovms {

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
    ov::genai::Tokenizer tokenizer;
    size_t processedTokens = 0;  // tracks overall number of tokens processed by the pipeline

    // Output parser is used to parse chat completions response to extract specific fields like tool calls and reasoning.
    std::unique_ptr<OutputParser> outputParser = nullptr;

    absl::Status parseCompletionsPart();
    absl::Status parseChatCompletionsPart(std::optional<uint32_t> maxTokensLimit, std::optional<std::string> allowedLocalMediaPath);
    absl::Status parseCommonPart(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength);

    ParsedOutput parseOutputIfNeeded(const std::vector<int64_t>& generatedIds);

public:
    OpenAIChatCompletionsHandler(Document& doc, Endpoint endpoint, std::chrono::time_point<std::chrono::system_clock> creationTime,
        ov::genai::Tokenizer tokenizer, const std::string& toolParserName = "", const std::string& reasoningParserName = "") :
        doc(doc),
        endpoint(endpoint),
        created(creationTime),
        tokenizer(tokenizer) {
        if (!toolParserName.empty() || !reasoningParserName.empty()) {
            outputParser = std::make_unique<OutputParser>(tokenizer, toolParserName, reasoningParserName);
        }
    }

    const OpenAIChatCompletionsRequest& getRequest() const;
    std::optional<std::string> getPrompt() const;
    std::optional<int> getNumReturnSequences() const;
    StreamOptions getStreamOptions() const;
    const std::string& getProcessedJson() const;
    const ImageHistory& getImageHistory() const;
    // User input might be modified by the servable logic, so it is not const
    ov::genai::ChatHistory& getChatHistory();
    std::optional<int> getMaxTokens() const;
    std::optional<std::string> getResponseSchema() const;

    bool isStream() const;
    std::string getModel() const;

    void setPromptTokensUsage(size_t promptTokens);

    void incrementProcessedTokens(size_t numTokens = 1);

    absl::Status parseRequest(std::optional<uint32_t> maxTokensLimit, uint32_t bestOfLimit, std::optional<uint32_t> maxModelLength, std::optional<std::string> allowedLocalMediaPath = std::nullopt);
    absl::Status parseMessages(std::optional<std::string> allowedLocalMediaPath = std::nullopt);
    absl::Status parseTools();
    const bool areToolsAvailable() const;

    std::string serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs);
    std::string serializeUnaryResponse(const ov::genai::EncodedResults& results);
    // VLMDecodedResults does not contain tokens that we can count, so we need to pass completionTokens in order to provide correct usage statistics
    std::string serializeUnaryResponse(const ov::genai::VLMDecodedResults& results, size_t completionTokens);
    std::string serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason);
    std::string serializeStreamingUsageChunk();
};
}  // namespace ovms

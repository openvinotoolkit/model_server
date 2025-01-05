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
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/generation_handle.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/runtime/tensor.hpp>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>

#include "absl/status/status.h"

using namespace rapidjson;

namespace ovms {

struct StreamOptions {
    bool includeUsage = false;
};

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

// Class that maps OpenAI request content and provides methods to create GenerationConfig from it.
struct OpenAIChatCompletionsRequest {
    std::string processedJson;
    std::vector<ov::Tensor> images;
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
    int logprobs = 0;
    bool logprobschat = false;
    bool echo{false};

    OpenAIChatCompletionsRequest() = default;
    ~OpenAIChatCompletionsRequest() = default;

    ov::genai::GenerationConfig createGenerationConfig(ov::genai::Adapter adapter) const {
        ov::genai::GenerationConfig config;
        ov::genai::AdaptersProperty test;

        // Generic
        if (maxTokens.has_value())
            config.max_new_tokens = maxTokens.value();
        // TODO: max_length = ?
        if (ignoreEOS.has_value())
            config.ignore_eos = ignoreEOS.value();

        config.echo = echo;

        // Beam search specific
        config.num_beam_groups = 1;  // OpenAI hardcoded
        config.num_beams = 1;        // OpenAI hardcoded
        config.no_repeat_ngram_size = std::numeric_limits<size_t>::max();

        if (bestOf.has_value())
            config.num_beams = bestOf.value();

        if (diversityPenalty.has_value())
            config.diversity_penalty = diversityPenalty.value();  // TODO: Not available in OpenAI nor vLLM
        // TODO: stop_criteria = ?
        if (numReturnSequences.has_value())
            config.num_return_sequences = numReturnSequences.value();
        if (repetitionPenalty.has_value())
            config.repetition_penalty = repetitionPenalty.value();
        if (lengthPenalty.has_value())
            config.length_penalty = lengthPenalty.value();
        // TODO: no_repeat_ngram_size = ?
        // TODO: early_finish = ?
        // TODO use_beam_search is unused ?

        // Multinomial specific
        if (temperature.has_value())
            config.temperature = temperature.value();
        if (topK.has_value())
            config.top_k = topK.value();
        if (topP.has_value())
            config.top_p = topP.value();
        if (seed.has_value())
            config.rng_seed = seed.value();
        if (stop.has_value())
            config.stop_strings = stop.value();
        if (includeStopStrInOutput.has_value())
            config.include_stop_str_in_output = includeStopStrInOutput.value();
        if (frequencyPenalty.has_value())
            config.frequency_penalty = frequencyPenalty.value();
        if (presencePenalty.has_value())
            config.presence_penalty = presencePenalty.value();
        config.do_sample = config.temperature > 0.0f && config.num_beams == 1;

        if (logprobschat || logprobs > 0)
            config.logprobs = 1;    
        //config.adapters = ov::genai::AdapterConfig(adapter, 1, ov::genai::AdapterConfig::Mode::MODE_STATIC);
        //config.adapters = std::nullptr;
        return config;
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

    absl::Status parseCompletionsPart();
    absl::Status parseChatCompletionsPart(uint32_t maxTokensLimit);
    absl::Status parseCommonPart(uint32_t maxTokensLimit, uint32_t bestOfLimit);

public:
    OpenAIChatCompletionsHandler(Document& doc, Endpoint endpoint, std::chrono::time_point<std::chrono::system_clock> creationTime,
        ov::genai::Tokenizer tokenizer) :
        doc(doc),
        endpoint(endpoint),
        created(creationTime),
        tokenizer(tokenizer) {}

    std::optional<std::string> getPrompt() const;
    std::optional<int> getNumReturnSequences() const;
    StreamOptions getStreamOptions() const;
    const std::string& getProcessedJson() const;
    const std::vector<ov::Tensor> getImages() const;

    bool isStream() const;
    std::string getModel() const;

    void setPromptTokensUsage(int promptTokens);

    void incrementProcessedTokens(int numTokens = 1);

    ov::genai::GenerationConfig createGenerationConfig(ov::genai::Adapter) const;

    absl::Status parseRequest(uint32_t maxTokensLimit, uint32_t bestOfLimit);
    absl::Status parseMessages();

    std::string serializeUnaryResponse(const std::vector<ov::genai::GenerationOutput>& generationOutputs);
    std::string serializeStreamingChunk(const std::string& chunkResponse, ov::genai::GenerationFinishReason finishReason);
    std::string serializeStreamingUsageChunk();
    static void writeLogprob(Writer<StringBuffer>& writer, float logprob);
};
}  // namespace ovms

//*****************************************************************************
// Copyright 2025 Intel Corporation
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

// Type that holds vector of pairs where first element is chat turn index and second is image tensor
// this way we store information about which image is associated with which chat turn
#pragma once
#include <map>
#include <optional>
#include <string>
#include <set>
#include <utility>
#include <vector>

#include <openvino/runtime/tensor.hpp>
#include <openvino/genai/tokenizer.hpp>

namespace ovms {
using ImageHistory = std::vector<std::pair<size_t, ov::Tensor>>;

struct StreamOptions {
    bool includeUsage = false;
};

using ToolsSchemas_t = std::map<std::string, std::string>;
// Class that maps OpenAI request content.
struct OpenAIChatCompletionsRequest {
    ov::genai::ChatHistory chatHistory;
    std::string processedJson;
    ImageHistory imageHistory;
    std::optional<std::string> prompt{std::nullopt};
    bool stream{false};
    StreamOptions streamOptions;
    std::string model;
    std::optional<int> maxTokens{std::nullopt};
    bool logprobs{false};
    int logprobschat{0};
    bool echo{false};
    std::optional<bool> ignoreEOS{std::nullopt};
    std::optional<std::set<std::string>> stop{std::nullopt};
    std::optional<bool> includeStopStrInOutput{std::nullopt};
    std::optional<int> numReturnSequences{std::nullopt};  // effective for beam search and multinomial decoding
    // Multinomial decoding specific
    std::optional<float> temperature{std::nullopt};
    std::optional<float> topP{std::nullopt};
    std::optional<int> topK{std::nullopt};
    std::optional<int> seed{std::nullopt};
    std::optional<float> frequencyPenalty{std::nullopt};
    std::optional<float> presencePenalty{std::nullopt};
    std::optional<float> repetitionPenalty{std::nullopt};
    // Beam search specific
    std::optional<int> bestOf{std::nullopt};
    std::optional<float> lengthPenalty{std::nullopt};

    // Assisted decoding specific (only with speculative decoding or prompt lookup pipeline)
    std::optional<int> numAssistantTokens{std::nullopt};
    std::optional<float> assistantConfidenceThreshold{std::nullopt};
    std::optional<int> maxNgramSize{std::nullopt};

    std::optional<uint32_t> maxModelLength;

    // Guided generation specific
    // Schema for response_format handling
    std::optional<std::string> responseSchema{std::nullopt};
    // Map that holds tool names and schemas for their arguments
    ToolsSchemas_t toolNameSchemaMap;
    // Holds value for tool_choice field as described in https://platform.openai.com/docs/api-reference/chat/create#chat_create-tool_choice
    std::string toolChoice;

    OpenAIChatCompletionsRequest() = default;
    ~OpenAIChatCompletionsRequest() = default;
};
}  // namespace ovms

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
#include "openvino/genai/streamer_base.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../logging.hpp"
#include "../stringutils.hpp"
#include "../http_payload.hpp"
#include "apis/openai_completions.hpp"

// Text streamer implementation copied from GenAI. Use GenAI directly when it's moved to the interface.
namespace ov {
namespace genai {
class TextCallbackStreamer : public StreamerBase {
protected:
    Tokenizer m_tokenizer;
    std::vector<int64_t> m_tokens_cache;
    std::vector<int64_t> m_decoded_lengths;
    size_t m_printed_len = 0;

public:
    bool put(int64_t token) {
        std::stringstream res;
        m_tokens_cache.push_back(token);
        std::string text = m_tokenizer.decode(m_tokens_cache);
        m_decoded_lengths.push_back(text.length());

        if (!text.empty() && '\n' == text.back() && text.size() > m_printed_len) {
            // Flush the cache after the new line symbol
            res << std::string_view{text.data() + m_printed_len, text.size() - m_printed_len};
            m_tokens_cache.clear();
            m_decoded_lengths.clear();
            m_printed_len = 0;
            return on_finalized_subword_callback(res.str());
        }

        constexpr size_t delay_n_tokens = 3;
        // In some cases adding the next token can shorten the text,
        // e.g. when apostrophe removing regex had worked after adding new tokens.
        // Printing several last tokens is delayed.
        if (m_decoded_lengths.size() < delay_n_tokens) {
            return on_finalized_subword_callback(res.str());
        }
        constexpr char replacement[] = "\xef\xbf\xbd";  // MSVC with /utf-8 fails to compile <UNK> directly with newline in string literal error.
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, replacement) == 0) {
            m_decoded_lengths[m_decoded_lengths.size() - 1] = -1;
            // Don't print incomplete text
            return on_finalized_subword_callback(res.str());
        }
        int64_t print_until = m_decoded_lengths[m_decoded_lengths.size() - delay_n_tokens];
        if (print_until != -1 && print_until > static_cast<int64_t>(m_printed_len)) {
            // It is possible to have a shorter text after adding new token.
            // Print to output only if text length is increaesed.
            res << std::string_view{text.data() + m_printed_len, print_until - m_printed_len} << std::flush;
            m_printed_len = print_until;
        }

        return on_finalized_subword_callback(res.str());
    }

    void end() {
        std::stringstream res;
        std::string text = m_tokenizer.decode(m_tokens_cache);
        if (text.size() <= m_printed_len)
            return;
        res << std::string_view{text.data() + m_printed_len, text.size() - m_printed_len} << std::flush;
        m_tokens_cache.clear();
        m_decoded_lengths.clear();
        m_printed_len = 0;
        on_finalized_subword_callback(res.str());
        return;
    }

    TextCallbackStreamer(const Tokenizer& tokenizer, std::function<bool(std::string)> callback) {
        m_tokenizer = tokenizer;
        on_finalized_subword_callback = callback;
    }

    std::function<bool(std::string)> on_finalized_subword_callback = [](std::string words) -> bool { return false; };
};
}  // namespace genai
}  // namespace ov
// End of text streamer implementation copy

namespace ovms {
// Some pipelines internals rely on request_id, so for now we provide increasing ID
static std::atomic<uint64_t> currentRequestId = 0;

class Status;

// Basic container with members required by all pipelines
// Specific pipelines should extend it as needed.
struct BasicExecutionContext {
    ovms::HttpPayload payload;
    Endpoint endpoint;
    std::shared_ptr<OpenAIChatCompletionsHandler> apiHandler;
    std::string inputText;
    std::string response;
    std::shared_ptr<ov::genai::TextCallbackStreamer> textStreamer;
    bool sendLoopbackSignal = false;
    std::string lastStreamerCallbackOutput;
};

struct LLMNodeProperties {
    std::string modelsPath;
    std::string device;
    ov::AnyMap pluginConfig;
    ov::AnyMap tokenizerPluginConfig;
    uint32_t maxTokensLimit;
    uint32_t bestOfLimit;
};

class LLMNodeResources {
public:
    std::shared_ptr<LLMNodeProperties> properties;

    LLMNodeResources() = default;
    LLMNodeResources(LLMNodeResources&&) = default;
    LLMNodeResources& operator=(LLMNodeResources&&) = default;
    LLMNodeResources(const LLMNodeResources&) = delete;
    LLMNodeResources& operator=(const LLMNodeResources&) = delete;
    virtual ~LLMNodeResources() = default;

    // ----- Common implementation for all derived classes

    // Loads payload into the execution context
    absl::Status loadRequest(std::shared_ptr<BasicExecutionContext>& executionContext, const ovms::HttpPayload& payload) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", payload.body);
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
        if (payload.uri == "/v3/chat/completions" || payload.uri == "/v3/v1/chat/completions") {
            executionContext->endpoint = Endpoint::CHAT_COMPLETIONS;
        } else if (payload.uri == "/v3/completions" || payload.uri == "/v3/v1/completions") {
            executionContext->endpoint = Endpoint::COMPLETIONS;
        } else {
            return absl::InvalidArgumentError("Wrong endpoint. Allowed endpoints: /v3/chat/completions, /v3/completions");
        }
        executionContext->payload = payload;
        return absl::OkStatus();
    };

    // ----- Interface for derived classes, TODO: add description when completed

    // Called in OVMS core hence returning ovms::Status
    virtual ovms::Status initialize() = 0;

    // All below methods are called from the calculator, hence returning absl::Status

    virtual std::shared_ptr<BasicExecutionContext> createExecutionContext() = 0;

    // Consider merging
    virtual absl::Status createApiHandler(std::shared_ptr<BasicExecutionContext>& executionContext) = 0;
    
    virtual absl::Status parseRequest(std::shared_ptr<BasicExecutionContext>& executionContext) = 0;
    // ---

    // Fill executionContext with data necessary to start the generation
    virtual absl::Status preparePipelineInput(std::shared_ptr<BasicExecutionContext>& executionContext) = 0;

    // This method should implement any necessary queueing mechanism or start asynchronous execution.
    // Execution context in such case may contain handles futures or other objects that will be used to track the execution.
    // If none of that is necessary, the implementation can simply return OK status.
    virtual absl::Status schedulePipelineExecution(std::shared_ptr<BasicExecutionContext>& executionContext) = 0;

    // This method should implement reading the results of the execution and filling the executionContext with the results.
    // If interacting with the pipeline is not asynchronous and does not require any queuing (schedulePipelineExecution implementation is essenatially void),
    // then this method should run entire execution.
    virtual absl::Status readCompleteExecutionResults(std::shared_ptr<BasicExecutionContext>& executionContext) = 0;
    virtual absl::Status readPartialExecutionResults(std::shared_ptr<BasicExecutionContext>& executionContext) = 0;

};

using LLMNodeResourcesMap = std::unordered_map<std::string, std::shared_ptr<LLMNodeResources>>;
}  // namespace ovms

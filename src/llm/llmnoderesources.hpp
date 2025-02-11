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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/genai/scheduler_config.hpp>
#include <openvino/openvino.hpp>

#include "openvino/genai/streamer_base.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>
#pragma warning(pop)

#include "../logging.hpp"
#include "../stringutils.hpp"
#include "src/llm/llm_calculator.pb.h"
#include "src/python/utils.hpp"
#include "text_processor.hpp"

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

class LLMExecutorWrapper;

class Status;

using plugin_config_t = std::map<std::string, ov::Any>;

#pragma GCC visibility push(hidden)

struct LLMNodeResources {
public:
    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> cbPipe = nullptr;
    bool isSpeculativePipeline{false};
    std::string modelsPath;
    std::string device;
    plugin_config_t pluginConfig;
    ov::genai::SchedulerConfig schedulerConfig;
    TextProcessor textProcessor;
    int maxTokensLimit;
    int bestOfLimit;

    static Status initializeLLMNodeResources(LLMNodeResources& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNode, std::string graphPath);
    static void loadTextProcessor(LLMNodeResources& nodeResources, const std::string& chatTemplateDirectory);

    LLMNodeResources(const LLMNodeResources&) = delete;
    LLMNodeResources& operator=(LLMNodeResources&) = delete;
    LLMNodeResources() = default;
    virtual ~LLMNodeResources() = default;

    virtual void initiateGeneration();

    void notifyExecutorThread();

private:
    std::unique_ptr<LLMExecutorWrapper> llmExecutorWrapper;
    static std::unordered_map<std::string, std::string> prepareLLMNodeInitializeArguments(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, std::string basePath);
    static ov::genai::SchedulerConfig prepareDraftModelSchedulerConfig(const mediapipe::LLMCalculatorOptions& nodeOptions);

public:
    virtual void initializeContinuousBatchingPipeline(
        const std::string& basePath,
        const ov::genai::SchedulerConfig& schedulerConfig,
        const std::string& device,
        const plugin_config_t& pluginConfig,
        const plugin_config_t& tokenizerPluginConfig);
};
#pragma GCC visibility pop
using LLMNodeResourcesMap = std::unordered_map<std::string, std::shared_ptr<LLMNodeResources>>;

}  // namespace ovms

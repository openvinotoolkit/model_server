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
#include <openvino/genai/lora_adapter.hpp>
#include <openvino/openvino.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop

#include <pybind11/embed.h>  // everything needed for embedding
#include <pybind11/stl.h>

#include "../logging.hpp"
#include "../stringutils.hpp"
#include "src/python/utils.hpp"
#include "text_processor.hpp"

namespace ovms {

class LLMExecutorWrapper;

// TODO: To be moved to CB library.
class TextStreamer {
    std::shared_ptr<ov::genai::Tokenizer> tokenizer;
    std::vector<int64_t> tokenCache;
    size_t printLen{0};

public:
    TextStreamer(std::shared_ptr<ov::genai::Tokenizer> tokenizer) :
        tokenizer(std::move(tokenizer)) {}

    std::optional<std::string> put(std::vector<int64_t>& tokens) {
        for (auto token : tokens) {
            tokenCache.push_back(token);
        }
        std::string text = tokenizer->decode(tokenCache);
        if (!text.empty() && '\n' == text.back() && text.size() > printLen) {
            // The chunk is ready if the generated text ends with new line.
            // Also, clear the cache.
            std::string chunk = std::string{text.data() + printLen, text.size() - printLen};
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", tokenCache);
            tokenCache.clear();
            printLen = 0;
            return chunk;
        } else if (!isValidUtf8(text)) {
            return std::nullopt;
        } else if (text.size() > printLen) {
            // The chunk is ready if the new text in the cache contains space.
            // The chunk is constructed from the new text, however only up to the last space character (including it)
            // Does not clear the cache.
            auto lastSpacePos = text.rfind(' ');
            if (lastSpacePos == std::string::npos || lastSpacePos < printLen) {
                return std::nullopt;
            }
            std::string chunk = std::string{text.data() + printLen, lastSpacePos - printLen + 1};
            printLen = lastSpacePos + 1;
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", tokenCache);
            return chunk;
        }
        return std::nullopt;
    }
    std::string end() {
        if (tokenCache.size() > 0) {
            std::string text = tokenizer->decode(tokenCache);
            std::string chunk = std::string{text.data() + printLen, text.size() - printLen};
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", tokenCache);
            tokenCache.clear();
            printLen = 0;
            return chunk;
        }
        return "";
    }
};

class Status;

using plugin_config_t = std::map<std::string, ov::Any>;

#pragma GCC visibility push(hidden)

struct LLMNodeResources {
public:
    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> cbPipe = nullptr;
    std::string modelsPath;
    std::string device;
    plugin_config_t pluginConfig;
    ov::genai::SchedulerConfig schedulerConfig;
    TextProcessor textProcessor;
    ov::genai::AdapterConfig adapters;
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

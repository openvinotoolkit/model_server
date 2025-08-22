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

#pragma once
#include <memory>
#include <iostream>
#include <string>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/tokenizer.hpp>
#include "base_generation_config_builder.hpp"
#include "phi4/generation_config_builder.hpp"
#include "llama3/generation_config_builder.hpp"
#include "hermes3/generation_config_builder.hpp"
#include "../apis/openai_request.hpp"
#include "../../logging.hpp"

namespace ovms {
class GenerationConfigBuilder {
    std::unique_ptr<BaseGenerationConfigBuilder> builder_impl;

public:
    GenerationConfigBuilder() = delete;
    // Using tool parser name to select appropriate builder implementation to avoid introducing additional parameters. Might be insufficient in the future.
    explicit GenerationConfigBuilder(ov::genai::GenerationConfig baseConfig, std::string toolParserName = "", bool enableToolGuidedGeneration = false) {
        if (!enableToolGuidedGeneration) {
            builder_impl = std::make_unique<BaseGenerationConfigBuilder>(baseConfig);
            return;
        }

        if (toolParserName == "llama3") {
            builder_impl = std::make_unique<Llama3GenerationConfigBuilder>(baseConfig);
        } else if (toolParserName == "qwen3") {
            // Qwen3 and Hermes3 share the same mechanism for generating tool calls, so we can use Hermes3GenerationConfigBuilder
            builder_impl = std::make_unique<Hermes3GenerationConfigBuilder>(baseConfig);
        } else if (toolParserName == "hermes3") {
            builder_impl = std::make_unique<Hermes3GenerationConfigBuilder>(baseConfig);
        } else if (toolParserName == "phi4") {
            builder_impl = std::make_unique<Phi4GenerationConfigBuilder>(baseConfig);
        } else {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Option enable_tool_guided_generation is set, but will not be effective since no valid tool parser has been provided.");
            builder_impl = std::make_unique<BaseGenerationConfigBuilder>(baseConfig);
        }
    }

    ov::genai::GenerationConfig& getConfig() {
        return builder_impl->getConfig();
    }

    // Validates the structured output configuration, if exists.
    // Throws exception if validation fails.
    void validateStructuredOutputConfig(ov::genai::Tokenizer& tokenizer) {
        builder_impl->validateStructuredOutputConfig(tokenizer);
    }

    // Fills generation config with values read from OpenAI request
    void parseConfigFromRequest(const OpenAIChatCompletionsRequest& request) {
        builder_impl->parseConfigFromRequest(request);
    }
};
}  // namespace ovms

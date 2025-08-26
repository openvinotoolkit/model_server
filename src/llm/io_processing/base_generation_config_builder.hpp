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
#include <string>

#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/tokenizer.hpp>
#include "../apis/openai_request.hpp"

namespace ovms {

/*
 * BaseGenerationConfigBuilder is a class that helps in building the base generation configuration
 * for OpenVINO GenAI pipeline based on OpenAI API request. 
 * This class provides functionalities common for different models and pipeline types.
 * It is designed to be extended by specific configuration builders for different models or pipeline types.
 */
class BaseGenerationConfigBuilder {
protected:
    ov::genai::GenerationConfig config;
    void setStructuralTagsConfig(const ov::genai::StructuralTagsConfig& structuralTagsConfig);

public:
    BaseGenerationConfigBuilder() = delete;
    // Initializes the builder with a base generation config read from model generation_config.json
    explicit BaseGenerationConfigBuilder(ov::genai::GenerationConfig& baseConfig) :
        config(baseConfig) {}
    virtual ~BaseGenerationConfigBuilder() = default;

    ov::genai::GenerationConfig& getConfig() { return config; }

    void addStopString(const std::string& decodedStopString);

    // Validates the structured output configuration, if exists.
    // Throws exception if validation fails.
    void validateStructuredOutputConfig(ov::genai::Tokenizer& tokenizer);

    /*
     * Fills generation config with values read from OpenAI request.
     * If extended, model specific implementation should call base class method first to fill in common configuration
     * and then set model specific parameters.
     */
    virtual void parseConfigFromRequest(const OpenAIChatCompletionsRequest& request);
};
}  // namespace ovms

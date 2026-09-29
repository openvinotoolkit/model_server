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
* DecodingMethod enum is used to properly set defaults and validate GenerationConfig depending on whether pipeline has been
* configured to use standard sampling strategies like greedy, beam search or multinomial or non-standard strategies like 
* speculative decoding with draft model or prompt lookup technique.
*
* STANDARD: Standard decoding methods such as greedy, beam search, and multinomial sampling. No special pipeline configuration.
* FAST_DRAFT: Classic two-model speculative decoding — a smaller off-the-shelf LLM drafts tokens that the main model verifies.
*             Pipeline configured with draft_models_path.
* EAGLE3: EAGLE3 speculative decoding — draft head conditioned on main model hidden states; supports tree drafting.
*         Auto-detected from model rt_info (eagle3_mode=true). Greedy decoding only.
* DFLASH: DFlash speculative decoding — hidden-state-conditioned head using linear attention.
*         Auto-detected from model rt_info (dflash_mode=true). Supports VLM and multinomial sampling.
* MTP: Multi-Token Prediction — bundled prediction head (openvino_mtp_model.xml) inside the draft model directory.
*      Auto-detected from model artifacts; no extra flag required. Multinomial sampling supported.
* PROMPT_LOOKUP: N-gram prompt lookup decoding; no draft model required.
*                Pipeline configured with prompt_lookup=true in pluginConfig.
*/
enum DecodingMethod {
    STANDARD,
    FAST_DRAFT,
    EAGLE3,
    DFLASH,
    MTP,
    PROMPT_LOOKUP
};

/*
 * BaseGenerationConfigBuilder is a class that helps in building the base generation configuration
 * for OpenVINO GenAI pipeline based on OpenAI API request. 
 * This class provides functionalities common for different models and pipeline types.
 * It is designed to be extended by specific configuration builders for different models or pipeline types.
 */
class BaseGenerationConfigBuilder {
protected:
    ov::genai::GenerationConfig config;
    const bool enableToolGuidedGeneration;
    DecodingMethod decodingMethod;
    void setStructuralTagsConfig(const ov::genai::StructuredOutputConfig::StructuralTag& structuralTag);

public:
    BaseGenerationConfigBuilder() = delete;
    // Initializes the builder with a base generation config read from model generation_config.json
    explicit BaseGenerationConfigBuilder(const ov::genai::GenerationConfig& baseConfig, bool enableToolGuidedGeneration, DecodingMethod decodingMethod) :
        config(baseConfig),
        enableToolGuidedGeneration(enableToolGuidedGeneration),
        decodingMethod(decodingMethod) {}
    virtual ~BaseGenerationConfigBuilder() = default;

    ov::genai::GenerationConfig& getConfig() { return config; }

    /*
    * Adjusts generation config based on the decoding method used in the pipeline.
    * This includes setting defaults for parameters required by the selected decoding method if they are not already set.
    */
    void adjustConfigForDecodingMethod();

    /*
    * Add stop string to generation config. Used when model server needs to add additional stop string that has not been provided in the request.
    */
    void addStopString(const std::string& decodedStopString);

    /*
    * Validates the structured output configuration, if exists.
    * Throws exception if validation fails.
    */
    void validateStructuredOutputConfig(ov::genai::Tokenizer& tokenizer);

    /*
     * Unsets the structured output configuration, effectively disabling guided generation.
     * Should be used when validateStructuredOutputConfig throws and we want to allow
     * the request to proceed without guided generation.
     */
    void unsetStructuredOutputConfig();

    /*
     * Fills generation config with values read from OpenAI request.
     * If extended, model specific implementation should call base class method first to fill in common configuration
     * and then set model specific parameters.
     */
    virtual void parseConfigFromRequest(const OpenAIRequest& request);
};
}  // namespace ovms

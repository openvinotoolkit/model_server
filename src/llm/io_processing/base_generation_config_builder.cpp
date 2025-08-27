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

#include <limits>
#include <string>
#include <openvino/genai/generation_config.hpp>
#include "base_generation_config_builder.hpp"

namespace ovms {
void BaseGenerationConfigBuilder::setStructuralTagsConfig(const ov::genai::StructuralTagsConfig& structuralTagsConfig) {
    if (config.structured_output_config) {
        config.structured_output_config->structural_tags_config = structuralTagsConfig;
    } else {
        ov::genai::StructuredOutputConfig structuredOutputConfig;
        structuredOutputConfig.structural_tags_config = structuralTagsConfig;
        config.structured_output_config = structuredOutputConfig;
    }
}

void BaseGenerationConfigBuilder::addStopString(const std::string& decodedStopString) {
    config.stop_strings.insert(decodedStopString);
}

void BaseGenerationConfigBuilder::validateStructuredOutputConfig(ov::genai::Tokenizer& tokenizer) {
    if (config.structured_output_config.has_value()) {
        config.structured_output_config.value().validate(tokenizer);
    }
}

void BaseGenerationConfigBuilder::unsetStructuredOutputConfig() {
    config.structured_output_config.reset();
}

void BaseGenerationConfigBuilder::parseConfigFromRequest(const OpenAIChatCompletionsRequest& request) {
    // Generic
    config.apply_chat_template = false;  // template is applied on the serving side
    if (request.maxTokens.has_value())
        config.max_new_tokens = request.maxTokens.value();
    if (request.maxModelLength.has_value())
        config.max_length = request.maxModelLength.value();
    if (request.ignoreEOS.has_value())
        config.ignore_eos = request.ignoreEOS.value();

    config.echo = request.echo;

    // Beam search specific
    config.num_beam_groups = 1;  // OpenAI hardcoded
    config.num_beams = 1;        // OpenAI hardcoded
    config.no_repeat_ngram_size = std::numeric_limits<size_t>::max();

    if (request.bestOf.has_value())
        config.num_beams = request.bestOf.value();

    // TODO: stop_criteria = ?
    if (request.numReturnSequences.has_value())
        config.num_return_sequences = request.numReturnSequences.value();
    if (request.repetitionPenalty.has_value())
        config.repetition_penalty = request.repetitionPenalty.value();
    if (request.lengthPenalty.has_value())
        config.length_penalty = request.lengthPenalty.value();
    // TODO: no_repeat_ngram_size = ?
    // TODO: early_finish = ?

    // Multinomial sampling specific
    if (request.temperature.has_value())
        config.temperature = request.temperature.value();
    if (request.topK.has_value())
        config.top_k = request.topK.value();
    if (request.topP.has_value())
        config.top_p = request.topP.value();
    if (request.seed.has_value())
        config.rng_seed = request.seed.value();
    if (request.stop.has_value())
        config.stop_strings = request.stop.value();
    if (request.includeStopStrInOutput.has_value())
        config.include_stop_str_in_output = request.includeStopStrInOutput.value();
    if (request.frequencyPenalty.has_value())
        config.frequency_penalty = request.frequencyPenalty.value();
    if (request.presencePenalty.has_value())
        config.presence_penalty = request.presencePenalty.value();
    config.do_sample = config.temperature > 0.0f && config.num_beams == 1;

    if (request.logprobschat || request.logprobs)
        config.logprobs = 1;
    // Assisted decoding specific
    if (request.numAssistantTokens.has_value())
        config.num_assistant_tokens = request.numAssistantTokens.value();
    if (request.assistantConfidenceThreshold.has_value())
        config.assistant_confidence_threshold = request.assistantConfidenceThreshold.value();
    if (request.maxNgramSize.has_value())
        config.max_ngram_size = request.maxNgramSize.value();

    // Response format handling
    if (request.responseSchema.has_value()) {
        ov::genai::StructuredOutputConfig structuredOutputConfig;
        structuredOutputConfig.json_schema = request.responseSchema.value();
        config.structured_output_config = structuredOutputConfig;
        config.stop_strings.insert("#");
    }
}
}  // namespace ovms

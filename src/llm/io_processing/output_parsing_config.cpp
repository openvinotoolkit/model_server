//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <algorithm>

#include "src/stringutils.hpp"
#include "output_parsing_config.hpp"
#include "parser_config_validation.hpp"
#include "qwen3/reasoning_parser.hpp"
#include "gemma4/gemma4_reasoning_parser.hpp"
#include "gptoss/reasoning_parser.hpp"
#include "minicpm5/minicpm5_reasoning_parser.hpp"
#include "lfm2/lfm25_reasoning_parser.hpp"
#include "onyx/onyx_reasoning_parser.hpp"

namespace ovms {

OutputParsingConfig getDefaultReasoningParsingConfig(const std::string& parserName) {
    if (parserName.empty()) {
        return {};
    }
    if (parserName == "qwen3") {
        return Qwen3ReasoningParser::getDefaultReasoningParsingConfig();
    }
    if (parserName == "gemma4") {
        return Gemma4ReasoningParser::getDefaultReasoningParsingConfig();
    }
    if (parserName == "gptoss") {
        return GptOssReasoningParser::getDefaultReasoningParsingConfig();
    }
    if (parserName == "minicpm5") {
        return Minicpm5ReasoningParser::getDefaultReasoningParsingConfig();
    }
    if (parserName == "lfm2") {
        return Lfm25ReasoningParser::getDefaultReasoningParsingConfig();
    }
    if (parserName == "onyx") {
        return OnyxReasoningParser::getDefaultReasoningParsingConfig();
    }
    throw std::runtime_error("Unsupported reasoning parser: \"" + parserName +
                             "\". Supported reasoning parsers are: " + getSupportedReasoningParserNamesAsString());
}

ReasoningGenerationContext determineReasoningContext(const OutputParsingConfig& config, const std::string& renderedPrompt) {
    ReasoningGenerationContext context;
    if (config.startTags.empty() && config.endTags.empty()) {
        return context;
    }
    std::string trimmed = renderedPrompt;
    rtrim(trimmed);
    const auto start = std::find_if(config.startTags.begin(), config.startTags.end(),
        [&](const std::string& tag) { return !tag.empty() && endsWith(trimmed, tag); });
    context.implicitStart = start != config.startTags.end();
    if (context.implicitStart) {
        context.begin = *start;
    } else if (config.startTags.size() == 1) {
        context.begin = config.startTags.front();
    }
    if (config.endTags.size() == 1) {
        context.end = config.endTags.front();
    }
    context.completedInPrompt = std::any_of(config.endTags.begin(), config.endTags.end(),
        [&](const std::string& tag) { return !tag.empty() && endsWith(trimmed, tag); });
    return context;
}

}  // namespace ovms

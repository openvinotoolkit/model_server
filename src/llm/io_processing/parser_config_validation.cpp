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
#include "parser_config_validation.hpp"

#include <algorithm>

#include "../../stringutils.hpp"

namespace ovms {

const std::vector<std::string>& getSupportedToolParserNames() {
    static const std::vector<std::string> names = {
        "llama3",
        "hermes3",
        "phi4",
        "mistral",
        "gptoss",
        "qwen3coder",
        "devstral",
        "lfm2",
        "lfm2.5",
        "gemma4",
    };
    return names;
}

const std::vector<std::string>& getSupportedReasoningParserNames() {
    static const std::vector<std::string> names = {
        "qwen3",
        "gemma4",
        "gptoss",
        "lfm2.5",
    };
    return names;
}

bool isSupportedToolParserName(const std::string& name) {
    const auto& names = getSupportedToolParserNames();
    return std::find(names.begin(), names.end(), name) != names.end();
}

bool isSupportedReasoningParserName(const std::string& name) {
    const auto& names = getSupportedReasoningParserNames();
    return std::find(names.begin(), names.end(), name) != names.end();
}

std::string getSupportedToolParserNamesAsString() {
    return joins(getSupportedToolParserNames(), ", ");
}

std::string getSupportedReasoningParserNamesAsString() {
    return joins(getSupportedReasoningParserNames(), ", ");
}

}  // namespace ovms

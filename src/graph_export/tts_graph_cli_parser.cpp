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
#include "tts_graph_cli_parser.hpp"

#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../capi_frontend/server_settings.hpp"
#include "../ovms_exit_codes.hpp"
#include "../status.hpp"

namespace ovms {

TextToSpeechGraphSettingsImpl& TextToSpeechGraphCLIParser::defaultGraphSettings() {
    static TextToSpeechGraphSettingsImpl instance;
    return instance;
}

void TextToSpeechGraphCLIParser::createOptions() {
    this->options = std::make_unique<cxxopts::Options>("ovms --pull [PULL OPTIONS ... ]", "-pull --task text_to_speech graph options");
    options->allow_unrecognised_options();

    // clang-format off
    options->add_options("TextToSpeech")
        ("num_streams",
            "The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems.",
            cxxopts::value<uint32_t>()->default_value("1"),
            "NUM_STREAMS");
}

void TextToSpeechGraphCLIParser::printHelp() {
    if (!this->options) {
        this->createOptions();
    }
    std::cout << options->help({"TextToSpeech"}) << std::endl;
}

std::vector<std::string> TextToSpeechGraphCLIParser::parse(const std::vector<std::string>& unmatchedOptions) {
    if (!this->options) {
        this->createOptions();
    }
    std::vector<const char*> cStrArray;
    cStrArray.reserve(unmatchedOptions.size() + 1);
    cStrArray.push_back("ovms graph");
    std::transform(unmatchedOptions.begin(), unmatchedOptions.end(), std::back_inserter(cStrArray), [](const std::string& str) { return str.c_str(); });
    const char* const* args = cStrArray.data();
    result = std::make_unique<cxxopts::ParseResult>(options->parse(cStrArray.size(), args));

    return  result->unmatched();
}

void TextToSpeechGraphCLIParser::prepare(OvmsServerMode serverMode, HFSettingsImpl& hfSettings, const std::string& modelName) {
    TextToSpeechGraphSettingsImpl textToSpeechGraphSettings = TextToSpeechGraphCLIParser::defaultGraphSettings();
    textToSpeechGraphSettings.targetDevice = hfSettings.exportSettings.targetDevice;
    if (modelName != "") {
        textToSpeechGraphSettings.modelName = modelName;
    } else {
        textToSpeechGraphSettings.modelName = hfSettings.sourceModel;
    }
    if (nullptr == result) {
        // Pull with default arguments - no arguments from user
        if (serverMode != HF_PULL_MODE && serverMode != HF_PULL_AND_START_MODE) {
            throw std::logic_error("Tried to prepare server and model settings without graph parse result");
        }
    } else {
        textToSpeechGraphSettings.numStreams = result->operator[]("num_streams").as<uint32_t>();
    }
    hfSettings.graphSettings = std::move(textToSpeechGraphSettings);
}

}  // namespace ovms

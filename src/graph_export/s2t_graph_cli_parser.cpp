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
#include "s2t_graph_cli_parser.hpp"

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

SpeechToTextGraphSettingsImpl& SpeechToTextGraphCLIParser::defaultGraphSettings() {
    static SpeechToTextGraphSettingsImpl instance;
    return instance;
}

void SpeechToTextGraphCLIParser::createOptions() {
    this->options = std::make_unique<cxxopts::Options>("ovms --pull [PULL OPTIONS ... ]", "-pull --task speech2text graph options");
    options->allow_unrecognised_options();

    // clang-format off
    options->add_options("SpeechToText")
        ("num_streams",
            "The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems.",
            cxxopts::value<uint32_t>()->default_value("1"),
            "NUM_STREAMS");
}

void SpeechToTextGraphCLIParser::printHelp() {
    if (!this->options) {
        this->createOptions();
    }
    std::cout << options->help({"SpeechToText"}) << std::endl;
}

std::vector<std::string> SpeechToTextGraphCLIParser::parse(const std::vector<std::string>& unmatchedOptions) {
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

void SpeechToTextGraphCLIParser::prepare(OvmsServerMode serverMode, HFSettingsImpl& hfSettings, const std::string& modelName) {
    SpeechToTextGraphSettingsImpl speechToTextGraphSettings = SpeechToTextGraphCLIParser::defaultGraphSettings();
    hfSettings.exportSettings.targetDevice = hfSettings.exportSettings.targetDevice;
    if (modelName != "") {
        hfSettings.exportSettings.modelName = modelName;
    } else {
        hfSettings.exportSettings.modelName = hfSettings.sourceModel;
    }
    if (nullptr == result) {
        // Pull with default arguments - no arguments from user
        if (serverMode != HF_PULL_MODE && serverMode != HF_PULL_AND_START_MODE) {
            throw std::logic_error("Tried to prepare server and model settings without graph parse result");
        }
    } else {
        hfSettings.exportSettings.pluginConfig.numStreams = result->operator[]("num_streams").as<uint32_t>();
    }
    hfSettings.graphSettings = std::move(speechToTextGraphSettings);
}

}  // namespace ovms

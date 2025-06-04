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
#include "embeddings_graph_cli_parser.hpp"

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

EmbeddingsGraphSettingsImpl& EmbeddingsGraphCLIParser::defaultGraphSettings() {
    static EmbeddingsGraphSettingsImpl instance;
    return instance;
}

void EmbeddingsGraphCLIParser::createOptions() {
    this->options = std::make_unique<cxxopts::Options>("ovms --pull [PULL OPTIONS ... ]", "-pull --task embeddings graph options");
    options->allow_unrecognised_options();

    // clang-format off
    options->add_options("embeddings")
        ("num_streams",
            "The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems.",
            cxxopts::value<uint32_t>()->default_value("1"),
            "NUM_STREAMS")
        ("normalize",
            "Normalize the embeddings.",
            cxxopts::value<std::string>()->default_value("true"),
            "NORMALIZE")
        ("mean_pooling",
            "Mean pooling option.",
            cxxopts::value<std::string>()->default_value("false"),
            "MEAN_POOLING");
}

void EmbeddingsGraphCLIParser::printHelp() {
    if (!this->options) {
        this->createOptions();
    }
    std::cout << options->help({"embeddings"}) << std::endl;
}

std::vector<std::string> EmbeddingsGraphCLIParser::parse(const std::vector<std::string>& unmatchedOptions) {
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

void EmbeddingsGraphCLIParser::prepare(OvmsServerMode serverMode, HFSettingsImpl& hfSettings, const std::string& modelName) {
    EmbeddingsGraphSettingsImpl embeddingsGraphSettings = EmbeddingsGraphCLIParser::defaultGraphSettings();
    embeddingsGraphSettings.targetDevice = hfSettings.targetDevice;
    if (modelName != "") {
        embeddingsGraphSettings.modelName = modelName;
    } else {
        embeddingsGraphSettings.modelName = hfSettings.sourceModel;
    }
    if (nullptr == result) {
        // Pull with default arguments - no arguments from user
        if (serverMode != HF_PULL_MODE && serverMode != HF_PULL_AND_START_MODE) {
            throw std::logic_error("Tried to prepare server and model settings without graph parse result");
        }
    } else {
        embeddingsGraphSettings.numStreams = result->operator[]("num_streams").as<uint32_t>();
        embeddingsGraphSettings.normalize = result->operator[]("normalize").as<std::string>();
        embeddingsGraphSettings.meanPooling = result->operator[]("mean_pooling").as<std::string>();
    }
    hfSettings.graphSettings = std::move(embeddingsGraphSettings);
}

}  // namespace ovms

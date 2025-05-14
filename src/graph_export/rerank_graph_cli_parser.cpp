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
#include "rerank_graph_cli_parser.hpp"

#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../capi_frontend/server_settings.hpp"
#include "../ovms_exit_codes.hpp"
#include "../status.hpp"

namespace ovms {

RerankGraphSettingsImpl& RerankGraphCLIParser::defaultGraphSettings() {
    static RerankGraphSettingsImpl instance;
    return instance;
}

void RerankGraphCLIParser::createOptions() {
    this->options = std::make_unique<cxxopts::Options>("ovms --pull [PULL OPTIONS ... ]", "-pull --task rerank graph options");

    // clang-format off
    options->add_options("rerank")
        ("graph_target_device",
            "CPU, GPU, NPU or HETERO, default is CPU.",
            cxxopts::value<std::string>()->default_value("CPU"),
            "GRAPH_TARGET_DEVICE")
        ("num_streams",
            "The number of parallel execution streams to use for the model. Use at least 2 on 2 socket CPU systems.",
            cxxopts::value<uint32_t>()->default_value("1"),
            "NUM_STREAMS")
        ("max_doc_length",
            "Maximum length of input documents in tokens.",
            cxxopts::value<uint32_t>()->default_value("16000"),
            "MAX_DOC_LENGTH")
        ("version",
            "Version of the model.",
            cxxopts::value<uint32_t>()->default_value("1"),
            "VERSION");
}

void RerankGraphCLIParser::printHelp() {
    if (!this->options) {
        this->createOptions();
    }
    std::cout << options->help({"rerank"}) << std::endl;
}

void RerankGraphCLIParser::parse(const std::vector<std::string>& unmatchedOptions) {
    try {
        if (!this->options) {
            this->createOptions();
        }
        std::vector<const char*> cStrArray;
        cStrArray.reserve(unmatchedOptions.size() + 1);
        cStrArray.push_back("ovms graph");
        std::transform(unmatchedOptions.begin(), unmatchedOptions.end(), std::back_inserter(cStrArray), [](const std::string& str) { return str.c_str(); });
        const char* const* args = cStrArray.data();
        result = std::make_unique<cxxopts::ParseResult>(options->parse(cStrArray.size(), args));

        if (result->unmatched().size()) {
            std::cerr << "error parsing options - unmatched arguments: ";
            for (auto& argument : result->unmatched()) {
                std::cerr << argument << ", ";
            }
            std::cerr << std::endl;
            exit(OVMS_EX_USAGE);
        }
    } catch (const std::exception& e) {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        exit(OVMS_EX_USAGE);
    }
}

void RerankGraphCLIParser::prepare(ServerSettingsImpl* serverSettings, ModelsSettingsImpl* modelsSettings) {
    if (nullptr == result) {
        // Pull with default arguments - no arguments from user
        if (serverSettings->hfSettings.pullHfModelMode) {
            serverSettings->hfSettings.rerankGraphSettings = RerankGraphCLIParser::defaultGraphSettings();
            // Deduct model name
            if (modelsSettings->modelName != "") {
                serverSettings->hfSettings.graphSettings.modelName = modelsSettings->modelName;
            } else {
                serverSettings->hfSettings.graphSettings.modelName = serverSettings->hfSettings.sourceModel;
            }
            return;
        } else {
            throw std::logic_error("Tried to prepare server and model settings without graph parse result");
        }
    }

    // Deduct model name
    if (modelsSettings->modelName != "") {
        serverSettings->hfSettings.graphSettings.modelName = modelsSettings->modelName;
    } else {
        serverSettings->hfSettings.graphSettings.modelName = serverSettings->hfSettings.sourceModel;
    }

    serverSettings->hfSettings.rerankGraphSettings.numStreams = result->operator[]("num_streams").as<uint32_t>();
    serverSettings->hfSettings.rerankGraphSettings.targetDevice = result->operator[]("graph_target_device").as<std::string>();
    serverSettings->hfSettings.rerankGraphSettings.maxDocLength = result->operator[]("max_doc_length").as<uint32_t>();
    serverSettings->hfSettings.rerankGraphSettings.version = result->operator[]("version").as<std::uint32_t>();

    if (!this->validate(serverSettings)) {
        throw std::logic_error("Error parsing graph options.");
    }
}

bool RerankGraphCLIParser::validate(ServerSettingsImpl* serverSettings) {
    // TODO: CVS-166727 add validation of rerankGraphSettings and plugin config
    if (serverSettings->hfSettings.task == "") {
        std::cerr << "Error: --task parameter not set." << std::endl;
        return false;
    }

    return true;
}

}  // namespace ovms

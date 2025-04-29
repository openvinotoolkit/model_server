//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include "graph_cli_parser.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../capi_frontend/server_settings.hpp"
#include "../ovms_exit_codes.hpp"
#include "../version.hpp"
#include "../status.hpp"

namespace ovms {

GraphSettingsImpl& GraphCLIParser::defaultGraphSettings() {
    static GraphSettingsImpl instance;
    return instance;
}

void GraphCLIParser::parse(const std::vector<std::string>& unmatchedOptions) {
    try {
        options = std::make_unique<cxxopts::Options>("ovms graph", "OpenVINO Model Server graph creation options");

        // clang-format off
        options->add_options()
            ("h, help",
                "Show this help message and exit");

        options->add_options("general options")
            ("max_num_seqs",
                "The maximum number of sequences that can be processed together. Default 256.",
                cxxopts::value<std::string>()->default_value("256"),
                "MAX_NUM_SEQS")
            ("pipeline_type",
                "Type of the pipeline to be used: Choices LM, LM_CB, VLM, VLM_CB, AUTO. AUTO is used by default.",
                cxxopts::value<std::string>()->default_value("AUTO"),
                "PIPELINE_TYPE")
            ("graph_target_device",
                "CPU, GPU, NPU or HETERO, default is CPU.",
                cxxopts::value<std::string>()->default_value("CPU"),
                "GRAPH_TARGET_DEVICE")
            ("enable_prefix_caching",
                "This algorithm is used to cache the prompt tokens.",
                cxxopts::value<std::string>()->default_value("true"),
                "ENABLE_PREFIX_CACHING")
            ("max_num_batched_tokens",
                "empty or integer. The maximum number of tokens that can be batched together.",
                cxxopts::value<std::string>()->default_value(""),
                "MAX_NUM_BATCHED_TOKENS")
            ("cache_size",
                "cache size in GB, default is 10.",
                cxxopts::value<std::string>()->default_value("10"),
                "CACHE_SIZE")
            ("draft_source_model",
                "HF model name or path to the local folder with PyTorch or OpenVINO draft model.",
                cxxopts::value<std::string>()->default_value(""),
                "DRAFT_SOURCE_MODEL")
            ("dynamic_split_fuse",
                "Dynamic split fuse algorithm enabled. Default true.",
                cxxopts::value<std::string>()->default_value("false"),
                "DYNAMIC_SPLIT_FUSE");

        options->add_options("plugin config")
            ("max_prompt_len",
                "Sets NPU specific property for maximum number of tokens in the prompt.",
                cxxopts::value<std::string>()->default_value(""),
                "MAX_PROMPT_LEN")
            ("kv_cache_precision",
                "u8 or empty (model default). Reduced kv cache precision to u8 lowers the cache size consumption.",
                cxxopts::value<std::string>()->default_value(""),
                "KV_CACHE_PRECISION");

        std::vector<const char*> cStrArray;
        cStrArray.reserve(unmatchedOptions.size()+1);
        cStrArray.push_back("ovms graph");
        for (const auto& str : unmatchedOptions) {
            cStrArray.push_back(str.c_str());
        }
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

        if (result->count("help") || result->arguments().size() == 0) {
            std::cout << options->help({"", "general options", "plugin config"}) << std::endl;
            exit(OVMS_EX_OK);
        }
    } catch (const std::exception& e) {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        exit(OVMS_EX_USAGE);
    }
}

void GraphCLIParser::prepare(ServerSettingsImpl* serverSettings, ModelsSettingsImpl* modelsSettings) {
    if (nullptr == result) {
        // Pull with default arguments - no arguments from user
        if (serverSettings->hfSettings.pullHfModelMode) {
            serverSettings->hfSettings.graphSettings = GraphCLIParser::defaultGraphSettings();
            return;
        } else {
            throw std::logic_error("Tried to prepare server and model settings without graph parse result");
        }
    }

    serverSettings->hfSettings.graphSettings.maxNumSeqs = result->operator[]("max_num_seqs").as<std::string>();
    serverSettings->hfSettings.graphSettings.pipelineType = result->operator[]("pipeline_type").as<std::string>();
    serverSettings->hfSettings.graphSettings.targetDevice = result->operator[]("graph_target_device").as<std::string>();
    serverSettings->hfSettings.graphSettings.enablePrefixCaching = result->operator[]("enable_prefix_caching").as<std::string>();
    serverSettings->hfSettings.graphSettings.maxNumBatchedTokens = result->operator[]("max_num_batched_tokens").as<std::string>();
    serverSettings->hfSettings.graphSettings.cacheSize = result->operator[]("cache_size").as<std::string>();
    serverSettings->hfSettings.graphSettings.dynamicSplitFuse = result->operator[]("dynamic_split_fuse").as<std::string>();
    serverSettings->hfSettings.graphSettings.draftModelDirName = result->operator[]("draft_source_model").as<std::string>();
    // TODO: modelPath
    // Plugin configuration
    if (result->count("max_prompt_len")) {
        serverSettings->hfSettings.graphSettings.pluginConfig.maxPromptLength = result->operator[]("max_prompt_len").as<std::string>();
    }

    if (result->count("kv_cache_precision")) {
        serverSettings->hfSettings.graphSettings.pluginConfig.kvCachePrecision = result->operator[]("kv_cache_precision").as<std::string>();
    }
}

Status GraphCLIParser::validate(ServerSettingsImpl* serverSettings) {
    // TODO: add validation of graphSettings and plugin config
    return StatusCode::OK;
}

}  // namespace ovms

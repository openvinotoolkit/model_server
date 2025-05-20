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
#include "graph_cli_parser.hpp"

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

TextGenGraphSettingsImpl& GraphCLIParser::defaultGraphSettings() {
    static TextGenGraphSettingsImpl instance;
    return instance;
}

void GraphCLIParser::createOptions() {
    this->options = std::make_unique<cxxopts::Options>("ovms --pull [PULL OPTIONS ... ]", "--pull --task text_generation graph options");
    options->allow_unrecognised_options();

    // clang-format off
    options->add_options("text generation")
        ("max_num_seqs",
            "The maximum number of sequences that can be processed together. Default 256.",
            cxxopts::value<uint32_t>()->default_value("256"),
            "MAX_NUM_SEQS")
        ("pipeline_type",
            "Type of the pipeline to be used: Choices LM, LM_CB, VLM, VLM_CB, AUTO. AUTO is used by default.",
            cxxopts::value<std::string>(),
            "PIPELINE_TYPE")
        ("enable_prefix_caching",
            "This algorithm is used to cache the prompt tokens.",
            cxxopts::value<std::string>()->default_value("true"),
            "ENABLE_PREFIX_CACHING")
        ("max_num_batched_tokens",
            "empty or integer. The maximum number of tokens that can be batched together.",
            cxxopts::value<uint32_t>(),
            "MAX_NUM_BATCHED_TOKENS")
        ("cache_size",
            "cache size in GB, default is 10.",
            cxxopts::value<uint32_t>()->default_value("10"),
            "CACHE_SIZE")
        ("draft_source_model",
            "HF model name or path to the local folder with PyTorch or OpenVINO draft model.",
            cxxopts::value<std::string>(),
            "DRAFT_SOURCE_MODEL")
        ("dynamic_split_fuse",
            "Dynamic split fuse algorithm enabled. Default true.",
            cxxopts::value<std::string>()->default_value("true"),
            "DYNAMIC_SPLIT_FUSE");

    options->add_options("plugin config")
        ("max_prompt_len",
            "Sets NPU specific property for maximum number of tokens in the prompt.",
            cxxopts::value<uint32_t>(),
            "MAX_PROMPT_LEN")
        ("kv_cache_precision",
            "u8 or empty (model default). Reduced kv cache precision to u8 lowers the cache size consumption.",
            cxxopts::value<std::string>()->default_value(""),
            "KV_CACHE_PRECISION");
}

void GraphCLIParser::printHelp() {
    if (!this->options) {
        this->createOptions();
    }
    std::cout << options->help({"text generation", "plugin config"}) << std::endl;
}

std::vector<std::string> GraphCLIParser::parse(const std::vector<std::string>& unmatchedOptions) {
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

void GraphCLIParser::prepare(HFSettingsImpl& hfSettings, const std::string& modelName, const std::string& modelPath) {
    TextGenGraphSettingsImpl graphSettings = GraphCLIParser::defaultGraphSettings();
    graphSettings.targetDevice = hfSettings.targetDevice;
    // Deduct model name
    if (modelName != "") {
        graphSettings.modelName = modelName;
    } else {
        graphSettings.modelName = hfSettings.sourceModel;
    }
    // Set model path
    if (modelPath != "") {
        graphSettings.modelPath = modelPath;
    }

    if (nullptr == result) {
        // Pull with default arguments - no arguments from user
        if (!hfSettings.pullHfModelMode || !hfSettings.pullHfAndStartModelMode) {
            throw std::logic_error("Tried to prepare server and model settings without graph parse result");
        }
    } else {
        graphSettings.maxNumSeqs = result->operator[]("max_num_seqs").as<uint32_t>();
        graphSettings.enablePrefixCaching = result->operator[]("enable_prefix_caching").as<std::string>();
        graphSettings.cacheSize = result->operator[]("cache_size").as<uint32_t>();
        graphSettings.dynamicSplitFuse = result->operator[]("dynamic_split_fuse").as<std::string>();
        if (result->count("draft_source_model")) {
            graphSettings.draftModelDirName = result->operator[]("draft_source_model").as<std::string>();
        }
        if (result->count("pipeline_type")) {
            graphSettings.pipelineType = result->operator[]("pipeline_type").as<std::string>();
        }
        if (result->count("max_num_batched_tokens")) {
            graphSettings.maxNumBatchedTokens = result->operator[]("max_num_batched_tokens").as<uint32_t>();
        }

        // Plugin configuration
        if (result->count("max_prompt_len")) {
            graphSettings.pluginConfig.maxPromptLength = result->operator[]("max_prompt_len").as<uint32_t>();
        }

        if (result->count("kv_cache_precision")) {
            graphSettings.pluginConfig.kvCachePrecision = result->operator[]("kv_cache_precision").as<std::string>();
        }
    }

    hfSettings.graphSettings = std::move(graphSettings);
}

}  // namespace ovms

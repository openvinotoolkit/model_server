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
#include "image_generation_graph_cli_parser.hpp"

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

ImageGenerationGraphSettingsImpl& ImageGenerationGraphCLIParser::defaultGraphSettings() {
    static ImageGenerationGraphSettingsImpl instance;
    return instance;
}

void ImageGenerationGraphCLIParser::createOptions() {
    this->options = std::make_unique<cxxopts::Options>("ovms --pull [PULL OPTIONS ... ]", "-pull --task image generation/edit/inpainting graph options");
    options->allow_unrecognised_options();

    // clang-format off
    options->add_options("image_generation")
        ("graph_target_device",
            "CPU, GPU, NPU or HETERO, default is CPU.",
            cxxopts::value<std::string>()->default_value("CPU"),
            "GRAPH_TARGET_DEVICE")
        ("default_resolution",
            "Default width and height of requested image in case user does not provide it.",
            cxxopts::value<std::string>()->default_value("512x512"),
            "DEFAULT_RESOLUTION");
}

void ImageGenerationGraphCLIParser::printHelp() {
    if (!this->options) {
        this->createOptions();
    }
    std::cout << options->help({"image_generation"}) << std::endl;
}

std::vector<std::string> ImageGenerationGraphCLIParser::parse(const std::vector<std::string>& unmatchedOptions) {
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

void ImageGenerationGraphCLIParser::prepare(HFSettingsImpl& hfSettings, const std::string& modelName) {
    ImageGenerationGraphSettingsImpl imageGenerationGraphSettings = ImageGenerationGraphCLIParser::defaultGraphSettings();
    // Deduct model name
    if (modelName != "") {
        imageGenerationGraphSettings.modelName = modelName;
    } else {
        imageGenerationGraphSettings.modelName = hfSettings.sourceModel;
    }
    if (nullptr == result) {
        // Pull with default arguments - no arguments from user
        if (!hfSettings.pullHfModelMode || !hfSettings.pullHfAndStartModelMode) {
            throw std::logic_error("Tried to prepare server and model settings without graph parse result");
        }
    } else {
        imageGenerationGraphSettings.targetDevice = result->operator[]("graph_target_device").as<std::string>();
        imageGenerationGraphSettings.defaultResolution = result->operator[]("default_resolution").as<std::string>();
    }

    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
}

}  // namespace ovms

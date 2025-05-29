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
    this->options = std::make_unique<cxxopts::Options>("ovms --pull [PULL OPTIONS ... ]", "--pull --task image generation/edit/inpainting graph options");
    options->allow_unrecognised_options();

    // clang-format off
    options->add_options("image_generation")
        ("graph_target_device",  // TODO: Remove
            "CPU, GPU, NPU or HETERO, default is CPU.",
            cxxopts::value<std::string>()->default_value("CPU"),
            "GRAPH_TARGET_DEVICE")
        ("max_resolution",
            "Max allowed resolution in a format of WxH; W=width H=height. If not specified, inherited from model.",
            cxxopts::value<std::string>(),
            "MAX_RESOLUTION")
        ("default_resolution",
            "Default resolution when not specified by client. If not specified, inherited from model.",
            cxxopts::value<std::string>(),
            "DEFAULT_RESOLUTION")
        ("max_number_images_per_prompt",
            "Max allowed number of images client is allowed to request for a given prompt.",
            cxxopts::value<uint32_t>(),
            "MAX_NUMBER_IMAGES_PER_PROMPT")
        ("default_num_inference_steps",
            "Default number of inference steps when not specified by client.",
            cxxopts::value<uint32_t>(),
            "DEFAULT_NUM_INFERENCE_STEPS")
        ("max_num_inference_steps",
            "Max allowed number of inference steps client is allowed to request for a given prompt.",
            cxxopts::value<uint32_t>(),
            "MAX_NUM_INFERENCE_STEPS");
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

void ImageGenerationGraphCLIParser::prepare(OvmsServerMode serverMode, HFSettingsImpl& hfSettings, const std::string& modelName) {
    ImageGenerationGraphSettingsImpl imageGenerationGraphSettings = ImageGenerationGraphCLIParser::defaultGraphSettings();
    // Deduct model name
    if (modelName != "") {
        imageGenerationGraphSettings.modelName = modelName;
    } else {
        imageGenerationGraphSettings.modelName = hfSettings.sourceModel;
    }
    if (nullptr == result) {
        // Pull with default arguments - no arguments from user
        if (serverMode != HF_PULL_MODE && serverMode != HF_PULL_AND_START_MODE) {
            throw std::logic_error("Tried to prepare server and model settings without graph parse result");
        }
    } else {
        imageGenerationGraphSettings.targetDevice = result->operator[]("graph_target_device").as<std::string>();
        imageGenerationGraphSettings.maxResolution = result->count("max_resolution") ? result->operator[]("max_resolution").as<std::string>() : "";
        imageGenerationGraphSettings.defaultResolution = result->count("default_resolution") ? result->operator[]("default_resolution").as<std::string>() : "";
        if (result->count("max_number_images_per_prompt"))  // TODO: Validate zeros?
            imageGenerationGraphSettings.maxNumberImagesPerPrompt = result->operator[]("max_number_images_per_prompt").as<uint32_t>();
        if (result->count("default_num_inference_steps"))
            imageGenerationGraphSettings.defaultNumInferenceSteps = result->operator[]("default_num_inference_steps").as<uint32_t>();
        if (result->count("max_num_inference_steps"))
            imageGenerationGraphSettings.maxNumInferenceSteps = result->operator[]("max_num_inference_steps").as<uint32_t>();
    }

    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
}

}  // namespace ovms

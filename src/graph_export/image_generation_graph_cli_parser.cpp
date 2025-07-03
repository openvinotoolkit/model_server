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
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../capi_frontend/server_settings.hpp"
#include "../ovms_exit_codes.hpp"
#include "../status.hpp"

namespace ovms {

static bool isValidResolution(const std::string& resolution) {
    static const std::regex pattern(R"(\d+x\d+)");
    return std::regex_match(resolution, pattern);
}

ImageGenerationGraphSettingsImpl& ImageGenerationGraphCLIParser::defaultGraphSettings() {
    static ImageGenerationGraphSettingsImpl instance;
    return instance;
}

void ImageGenerationGraphCLIParser::createOptions() {
    this->options = std::make_unique<cxxopts::Options>("ovms --pull [PULL OPTIONS ... ]", "--pull --task image generation/edit/inpainting graph options");
    options->allow_unrecognised_options();

    // clang-format off
    options->add_options("image_generation")
        ("resolution",
            "Allowed resolutions in a format list of WxH; W=width H=height - space separated. If not specified, inherited from model. If one is specified, the pipeline will be reshaped to static.",
            cxxopts::value<std::string>(),
            "RESOLUTION")
        ("max_resolution",
            "Max allowed resolution in a format of WxH; W=width H=height. If not specified, inherited from model.",
            cxxopts::value<std::string>(),
            "MAX_RESOLUTION")
        ("default_resolution",
            "Default resolution when not specified by client in a format of WxH; W=width H=height. If not specified, inherited from model.",
            cxxopts::value<std::string>(),
            "DEFAULT_RESOLUTION")
        ("num_images_per_prompt",
            "Number of images client is allowed to request. Can only be used when resolution parameter is specified and static. By default, inherited from GenAI (1).",
            cxxopts::value<uint32_t>(),
            "NUM_IMAGES_PER_PROMPT")
        ("guidance_scale",
            "Number of images client is allowed to request. Can only be used when resolution parameter is specified and static. By default, inherited from GenAI (7.5).",
            cxxopts::value<float>(),
            "GUIDANCE_SCALE")
        ("max_num_images_per_prompt",
            "Max allowed number of images client is allowed to request for a given prompt.",
            cxxopts::value<uint32_t>(),
            "MAX_NUM_IMAGES_PER_PROMPT")
        ("default_num_inference_steps",
            "Default number of inference steps when not specified by client.",
            cxxopts::value<uint32_t>(),
            "DEFAULT_NUM_INFERENCE_STEPS")
        ("max_num_inference_steps",
            "Max allowed number of inference steps client is allowed to request for a given prompt.",
            cxxopts::value<uint32_t>(),
            "MAX_NUM_INFERENCE_STEPS")
        ("num_streams",
            "The number of parallel execution streams to use for the image generation models. Use at least 2 on 2 socket CPU systems.",
            cxxopts::value<uint32_t>(),
            "NUM_STREAMS");
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

void ImageGenerationGraphCLIParser::prepare(ServerSettingsImpl& serverSettings, HFSettingsImpl& hfSettings, const std::string& modelName) {
    ImageGenerationGraphSettingsImpl imageGenerationGraphSettings = ImageGenerationGraphCLIParser::defaultGraphSettings();
    imageGenerationGraphSettings.targetDevice = hfSettings.targetDevice;
    // Deduct model name
    if (modelName != "") {
        imageGenerationGraphSettings.modelName = modelName;
    } else {
        imageGenerationGraphSettings.modelName = hfSettings.sourceModel;
    }
    if (nullptr == result) {
        // Pull with default arguments - no arguments from user
        if (serverSettings.serverMode != HF_PULL_MODE && serverSettings.serverMode != HF_PULL_AND_START_MODE) {
            throw std::logic_error("Tried to prepare server and model settings without graph parse result");
        }
    } else {
        imageGenerationGraphSettings.resolution = result->count("resolution") ? result->operator[]("resolution").as<std::string>() : "";
        imageGenerationGraphSettings.numImagesPerPrompt = result->count("num_images_per_prompt") ? std::optional<uint32_t>(result->operator[]("num_images_per_prompt").as<uint32_t>()) : std::nullopt;
        imageGenerationGraphSettings.guidanceScale = result->count("guidance_scale") ? std::optional<float>(result->operator[]("guidance_scale").as<float>()) : std::nullopt;
        imageGenerationGraphSettings.maxResolution = result->count("max_resolution") ? result->operator[]("max_resolution").as<std::string>() : "";
        if (!imageGenerationGraphSettings.maxResolution.empty() && !isValidResolution(imageGenerationGraphSettings.maxResolution)) {
            throw std::invalid_argument("Invalid max_resolution format. Expected WxH, e.g., 1024x1024");
        }
        imageGenerationGraphSettings.defaultResolution = result->count("default_resolution") ? result->operator[]("default_resolution").as<std::string>() : "";
        if (!imageGenerationGraphSettings.defaultResolution.empty() && !isValidResolution(imageGenerationGraphSettings.defaultResolution)) {
            throw std::invalid_argument("Invalid default_resolution format. Expected WxH, e.g., 1024x1024");
        }
        if (result->count("max_num_images_per_prompt")) {
            imageGenerationGraphSettings.maxNumberImagesPerPrompt = result->operator[]("max_num_images_per_prompt").as<uint32_t>();
            if (imageGenerationGraphSettings.maxNumberImagesPerPrompt == 0) {
                throw std::invalid_argument("max_num_images_per_prompt must be greater than 0");
            }
        }
        if (result->count("default_num_inference_steps")) {
            imageGenerationGraphSettings.defaultNumInferenceSteps = result->operator[]("default_num_inference_steps").as<uint32_t>();
            if (imageGenerationGraphSettings.defaultNumInferenceSteps == 0) {
                throw std::invalid_argument("default_num_inference_steps must be greater than 0");
            }
        }
        if (result->count("max_num_inference_steps")) {
            imageGenerationGraphSettings.maxNumInferenceSteps = result->operator[]("max_num_inference_steps").as<uint32_t>();
            if (imageGenerationGraphSettings.maxNumInferenceSteps == 0) {
                throw std::invalid_argument("max_num_inference_steps must be greater than 0");
            }
        }

        if (result->count("num_streams") || serverSettings.cacheDir != "") {
            rapidjson::Document pluginConfigDoc;
            pluginConfigDoc.SetObject();
            rapidjson::Document::AllocatorType& allocator = pluginConfigDoc.GetAllocator();
            if (result->count("num_streams")) {
                uint32_t numStreams = result->operator[]("num_streams").as<uint32_t>();
                if (numStreams == 0) {
                    throw std::invalid_argument("num_streams must be greater than 0");
                }
                pluginConfigDoc.AddMember("NUM_STREAMS", numStreams, allocator);
            }

            if (!serverSettings.cacheDir.empty()) {
                pluginConfigDoc.AddMember("CACHE_DIR", rapidjson::Value(serverSettings.cacheDir.c_str(), allocator), allocator);
            }

            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            pluginConfigDoc.Accept(writer);
            imageGenerationGraphSettings.pluginConfig = buffer.GetString();
        }
    }

    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
}

}  // namespace ovms

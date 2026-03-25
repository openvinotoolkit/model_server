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
#include <filesystem>
#include <iostream>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../capi_frontend/server_settings.hpp"
#include "../ovms_exit_codes.hpp"
#include "../status.hpp"
#include "src/stringutils.hpp"

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
            "Guidance scale used for static pipeline reshape. Can only be used when resolution parameter is specified and static. By default, inherited from GenAI (7.5).",
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
    // Deduct model name
    if (modelName != "") {
        hfSettings.exportSettings.modelName = modelName;
    } else {
        hfSettings.exportSettings.modelName = hfSettings.sourceModel;
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
            if (result->count("num_streams")) {
                uint32_t numStreams = result->operator[]("num_streams").as<uint32_t>();
                if (numStreams == 0) {
                    throw std::invalid_argument("num_streams must be greater than 0");
                }
                hfSettings.exportSettings.pluginConfig.numStreams = result->operator[]("num_streams").as<uint32_t>();
            }

            if (!serverSettings.cacheDir.empty()) {
                hfSettings.exportSettings.pluginConfig.cacheDir = serverSettings.cacheDir;
            }
        }
    }

    // Parse --source_loras
    // Supports three source types plus composite aliases:
    //   alias=org/repo              (HF_REPO)
    //   alias=org/repo@file.safetensors  (HF_REPO with explicit file)
    //   alias=https://url/file.safetensors  (DIRECT_URL)
    //   alias=/path/to/file.safetensors     (LOCAL_FILE)
    //   alias=@ref1:0.7+@ref2:0.5          (COMPOSITE - references other aliases)
    if (!hfSettings.sourceLoras.empty()) {
        auto entries = ovms::tokenize(hfSettings.sourceLoras, ',');
        // First pass: collect all real adapters
        for (const auto& entry : entries) {
            auto eqPos = entry.find('=');
            if (eqPos == std::string::npos) {
                throw std::invalid_argument("Missing alias in --source_loras entry: '" + entry + "'. Expected format: alias=source");
            }
            std::string alias = entry.substr(0, eqPos);
            std::string source = entry.substr(eqPos + 1);
            if (alias.empty() || source.empty()) {
                throw std::invalid_argument("Invalid --source_loras entry: '" + entry + "'. Alias and source must not be empty.");
            }
            // Skip composite entries in first pass
            if (source[0] == '@') {
                continue;
            }

            LoraAdapterSettings adapter;
            adapter.alias = alias;
            // Detect source type
            if (source.substr(0, 8) == "https://" || source.substr(0, 7) == "http://") {
                adapter.sourceType = LoraSourceType::DIRECT_URL;
                adapter.sourceLora = source;
                auto lastSlash = source.rfind('/');
                if (lastSlash == std::string::npos || lastSlash == source.size() - 1) {
                    throw std::invalid_argument("Cannot extract filename from URL in --source_loras entry: '" + entry + "'");
                }
                adapter.safetensorsFile = source.substr(lastSlash + 1);
                if (!endsWith(adapter.safetensorsFile, ".safetensors")) {
                    throw std::invalid_argument("URL must point to a .safetensors file in --source_loras entry: '" + entry + "'");
                }
            } else if (source[0] == '/' || source.substr(0, 2) == "./") {
                adapter.sourceType = LoraSourceType::LOCAL_FILE;
                adapter.sourceLora = source;
                if (!endsWith(source, ".safetensors")) {
                    throw std::invalid_argument("Local path must point to a .safetensors file in --source_loras entry: '" + entry + "'");
                }
                if (!std::filesystem::exists(source)) {
                    throw std::invalid_argument("Local LoRA file does not exist: '" + source + "' in --source_loras entry: '" + entry + "'");
                }
                auto lastSlash = source.rfind('/');
                adapter.safetensorsFile = (lastSlash != std::string::npos) ? source.substr(lastSlash + 1) : source;
            } else {
                adapter.sourceType = LoraSourceType::HF_REPO;
                auto atPos = source.find('@');
                if (atPos != std::string::npos) {
                    adapter.sourceLora = source.substr(0, atPos);
                    adapter.safetensorsFile = source.substr(atPos + 1);
                    if (adapter.safetensorsFile.empty()) {
                        throw std::invalid_argument("Empty filename after @ in --source_loras entry: '" + entry + "'");
                    }
                } else {
                    adapter.sourceLora = source;
                }
                if (adapter.sourceLora.empty()) {
                    throw std::invalid_argument("Invalid --source_loras entry: '" + entry + "'. HF repo source must not be empty.");
                }
            }
            imageGenerationGraphSettings.loraAdapters.push_back(std::move(adapter));
        }

        // Collect known adapter aliases for validation
        std::set<std::string> knownAliases;
        for (const auto& adapter : imageGenerationGraphSettings.loraAdapters) {
            knownAliases.insert(adapter.alias);
        }

        // Second pass: parse composite entries (source starts with @)
        for (const auto& entry : entries) {
            auto eqPos = entry.find('=');
            std::string alias = entry.substr(0, eqPos);
            std::string source = entry.substr(eqPos + 1);
            if (source[0] != '@') {
                continue;
            }
            CompositeLoraSettings composite;
            composite.alias = alias;
            // Parse @ref1:0.7+@ref2:0.5
            auto componentTokens = ovms::tokenize(source, '+');
            for (const auto& compToken : componentTokens) {
                if (compToken.empty() || compToken[0] != '@') {
                    throw std::invalid_argument("Invalid composite LoRA component '" + compToken + "' in entry: '" + entry + "'. Each component must start with @");
                }
                CompositeLoraComponent component;
                std::string ref = compToken.substr(1);  // strip @
                auto colonPos = ref.find(':');
                if (colonPos != std::string::npos) {
                    component.adapterAlias = ref.substr(0, colonPos);
                    std::string weightStr = ref.substr(colonPos + 1);
                    try {
                        component.weight = std::stof(weightStr);
                    } catch (...) {
                        throw std::invalid_argument("Invalid weight '" + weightStr + "' in composite LoRA component: '" + compToken + "'");
                    }
                } else {
                    component.adapterAlias = ref;
                }
                if (component.adapterAlias.empty()) {
                    throw std::invalid_argument("Empty adapter reference in composite LoRA component: '" + compToken + "'");
                }
                if (knownAliases.find(component.adapterAlias) == knownAliases.end()) {
                    throw std::invalid_argument("Composite LoRA references unknown adapter '" + component.adapterAlias + "' in entry: '" + entry + "'");
                }
                composite.components.push_back(std::move(component));
            }
            if (composite.components.empty()) {
                throw std::invalid_argument("Composite LoRA entry has no components: '" + entry + "'");
            }
            imageGenerationGraphSettings.compositeLoraAdapters.push_back(std::move(composite));
        }
    }

    hfSettings.graphSettings = std::move(imageGenerationGraphSettings);
}

}  // namespace ovms

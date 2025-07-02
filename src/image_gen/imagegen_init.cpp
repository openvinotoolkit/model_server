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
#include "imagegen_init.hpp"

#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/ascii.h"

#include "src/filesystem.hpp"
#include "src/image_gen/image_gen_calculator.pb.h"
#include "src/json_parser.hpp"
#include "src/logging.hpp"
#include "src/status.hpp"
#include "src/stringutils.hpp"

#include "imagegenutils.hpp"

namespace ovms {
static std::variant<Status, std::optional<resolution_t>> getDimensionsConfig(const std::string& resolutionString) {
    auto dimsOrStatus = getDimensions(resolutionString);
    if (std::holds_alternative<absl::Status>(dimsOrStatus)) {
        auto statusString = std::get<absl::Status>(dimsOrStatus).ToString();
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to parse resolution: {}", statusString);
        return Status(StatusCode::SHAPE_WRONG_FORMAT, statusString);
    }
    return std::get<std::optional<resolution_t>>(dimsOrStatus);
}

static std::variant<Status, std::vector<std::string>> getListOfDevices(const std::string& devicesString) {
    if (devicesString.empty()) {
        return std::vector<std::string>{};
    }

    std::string trimmedDevicesString = devicesString;
    absl::StripAsciiWhitespace(&trimmedDevicesString);

    // iterate over and leave only 1 space in betweens, even if there are 3 or more spaces
    while (trimmedDevicesString.find("  ") != std::string::npos) {
        trimmedDevicesString = absl::StrReplaceAll(trimmedDevicesString, {{"  ", " "}});
    }

    // split by space
    std::vector<std::string> devices = ovms::tokenize(trimmedDevicesString, ' ');

    if (devices.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "No valid devices found in: {}", devicesString);
        return Status(StatusCode::SHAPE_WRONG_FORMAT, "No valid devices found");
    }

    return devices;
}

static std::variant<Status, std::vector<resolution_t>> getListOfResolutions(const std::string& resolutionString) {
    // strip all whitespaces, leave 1 each, use abseil
    if (resolutionString.empty()) {
        return std::vector<resolution_t>{};
    }

    std::string trimmedResolutionString = resolutionString;
    absl::StripAsciiWhitespace(&trimmedResolutionString);

    // iterate over and leave only 1 space in betweens, even if there are 3 or more spaces
    while (trimmedResolutionString.find("  ") != std::string::npos) {
        trimmedResolutionString = absl::StrReplaceAll(trimmedResolutionString, {{"  ", " "}});
    }

    // split by space
    std::vector<std::string> resolutions = ovms::tokenize(trimmedResolutionString, ' ');

    // for each string, run getDimensionsConfig
    std::vector<resolution_t> result;
    for (const auto& resolution : resolutions) {
        auto resOptOrStatus = getDimensionsConfig(resolution);
        if (std::holds_alternative<Status>(resOptOrStatus)) {
            auto statusString = std::get<Status>(resOptOrStatus).string();
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to parse resolution: {}", statusString);
            return std::get<Status>(resOptOrStatus);
        }
        if (!std::get<std::optional<resolution_t>>(resOptOrStatus).has_value()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Resolution is not specified or is invalid: {}", resolution);
            return Status(StatusCode::SHAPE_WRONG_FORMAT, "Resolution is not specified or is invalid: " + resolution);
        }
        result.push_back(std::get<std::optional<resolution_t>>(resOptOrStatus).value());
    }

    if (result.empty()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "No valid resolutions found in: {}", resolutionString);
        return Status(StatusCode::SHAPE_WRONG_FORMAT, "No valid resolutions found");
    }

    // validate if there aren't duplicates
    std::set<resolution_t> uniqueResolutions(result.begin(), result.end());
    if (uniqueResolutions.size() != result.size()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Duplicate resolutions found in: {}", resolutionString);
        return Status(StatusCode::SHAPE_WRONG_FORMAT, "Duplicate resolutions found");
    }
    return result;
}

std::variant<Status, ImageGenPipelineArgs> prepareImageGenPipelineArgs(const google::protobuf::Any& calculatorOptions, const std::string& graphPath) {
    mediapipe::ImageGenCalculatorOptions nodeOptions;
    if (!calculatorOptions.UnpackTo(&nodeOptions)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to unpack calculator options");
        return StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID;
    }
    auto fsModelsPath = std::filesystem::path(nodeOptions.models_path());
    std::string pipelinePath;
    if (fsModelsPath.is_relative()) {
        pipelinePath = (std::filesystem::path(graphPath) / fsModelsPath).string();
    } else {
        pipelinePath = fsModelsPath.string();
    }
    ImageGenPipelineArgs args;
    args.modelsPath = pipelinePath;
    if (!FileSystem::dirExists(args.modelsPath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Models path does not exist: {}", args.modelsPath);
        return StatusCode::PATH_INVALID;
    }
    bool isNPU = false;
    if (nodeOptions.has_device()) {
        isNPU = nodeOptions.device().find("NPU") != std::string::npos;
        // use getListOfDevices
        auto devicesOrStatus = getListOfDevices(nodeOptions.device());
        if (std::holds_alternative<Status>(devicesOrStatus)) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to parse devices: {}", nodeOptions.device());
            return std::get<Status>(devicesOrStatus);
        }
        auto devices = std::get<std::vector<std::string>>(devicesOrStatus);
        if (devices.empty()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "No valid devices found in: {}", nodeOptions.device());
            return StatusCode::DEVICE_WRONG_FORMAT;
        }

        // allow only 1 or 3 devices
        if (devices.size() != 1 && devices.size() != 3) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Invalid number of devices specified: {}. Expected 1 or 3.", devices.size());
            return StatusCode::DEVICE_WRONG_FORMAT;
        }

        args.device = std::move(devices);
    }
    if (nodeOptions.has_resolution()) {
        auto res = getListOfResolutions(nodeOptions.resolution());
        // list all the res
        if (std::holds_alternative<Status>(res)) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to parse resolution: {}", nodeOptions.resolution());
            return std::get<Status>(res);
        }
        args.staticReshapeSettings = StaticReshapeSettingsArgs(std::get<std::vector<resolution_t>>(res));

        if (isNPU && args.staticReshapeSettings.value().resolution.size() > 1) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "NPU cannot have multiple resolutions in static settings");
            return StatusCode::SHAPE_DYNAMIC_BUT_NPU_USED;
        }
    } else if (isNPU) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot use NPU without setting static resolution");
        return StatusCode::SHAPE_DYNAMIC_BUT_NPU_USED;
    }
    if (args.staticReshapeSettings.has_value()) {
        if (nodeOptions.has_max_resolution()) {  // non default
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot explicitly use max resolution when using static settings");
            return StatusCode::STATIC_RESOLUTION_MISUSE;
        }
        if (nodeOptions.has_num_images_per_prompt()) {
            if (args.staticReshapeSettings.value().resolution.size() > 1) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot use static num images per prompt with multiple resolutions");
                return StatusCode::STATIC_RESOLUTION_MISUSE;
            }
            args.staticReshapeSettings->numImagesPerPrompt = nodeOptions.num_images_per_prompt();
        }
        if (nodeOptions.has_guidance_scale()) {
            if (args.staticReshapeSettings.value().resolution.size() > 1) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot use static guidance scale with multiple resolutions");
                return StatusCode::STATIC_RESOLUTION_MISUSE;
            }
            args.staticReshapeSettings->guidanceScale = nodeOptions.guidance_scale();
        }
        if (args.staticReshapeSettings.value().resolution.size() == 1 && nodeOptions.has_max_num_images_per_prompt()) {  // non default
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot explicitly use max num images per prompt when using static settings");
            return StatusCode::STATIC_RESOLUTION_MISUSE;
        }
        if (args.staticReshapeSettings.value().resolution.size() == 1 && nodeOptions.has_max_resolution()) {  // non default
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot explicitly use max resolution when using static settings");
            return StatusCode::STATIC_RESOLUTION_MISUSE;
        }
    } else {
        if (nodeOptions.has_guidance_scale()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot explicitly use static guidance scale when not using static resolution");
            return StatusCode::STATIC_RESOLUTION_MISUSE;
        }
        if (nodeOptions.has_num_images_per_prompt()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot explicitly use static num images per prompt when not using static resolution");
            return StatusCode::STATIC_RESOLUTION_MISUSE;
        }
    }
    if (nodeOptions.has_plugin_config()) {
        std::string pluginConfig = nodeOptions.plugin_config();
        auto status = JsonParser::parsePluginConfig(pluginConfig, args.pluginConfig);
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to parse plugin config: {}", status.string());
            return status;
        }
    }
    auto maxResOptOrStatus = getDimensionsConfig(nodeOptions.max_resolution());
    if (std::holds_alternative<Status>(maxResOptOrStatus)) {
        return std::get<Status>(maxResOptOrStatus);
    }
    if (!std::get<std::optional<resolution_t>>(maxResOptOrStatus).has_value()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Max resolution is not specified or is invalid: {}", nodeOptions.max_resolution());
        return StatusCode::SHAPE_WRONG_FORMAT;
    }
    args.maxResolution = std::get<std::optional<resolution_t>>(maxResOptOrStatus).value();
    if (nodeOptions.has_default_resolution()) {
        auto defaultResOptOrStatus = getDimensionsConfig(nodeOptions.default_resolution());
        if (std::holds_alternative<Status>(defaultResOptOrStatus)) {
            return std::get<Status>(defaultResOptOrStatus);
        }
        args.defaultResolution = std::get<std::optional<resolution_t>>(defaultResOptOrStatus);
        if (args.defaultResolution.value().first > args.maxResolution.first ||
            args.defaultResolution.value().second > args.maxResolution.second) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Default resolution exceeds maximum allowed resolution: {} > {}", args.defaultResolution.value(), args.maxResolution);
            return StatusCode::DEFAULT_EXCEEDS_MAXIMUM_ALLOWED_RESOLUTION;
        }
        // default resolution is not among the ones allowed
        if (args.staticReshapeSettings.has_value()) {
            auto& resolutions = args.staticReshapeSettings.value().resolution;
            auto it = std::find(resolutions.begin(), resolutions.end(), args.defaultResolution.value());
            if (it == resolutions.end()) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Default resolution {} is not among the static resolutions: {}", args.defaultResolution.value(), resolutions);
                return StatusCode::SHAPE_WRONG_FORMAT;
            }
        }
    }

    args.maxNumImagesPerPrompt = nodeOptions.max_num_images_per_prompt();
    args.defaultNumInferenceSteps = nodeOptions.default_num_inference_steps();
    args.maxNumInferenceSteps = nodeOptions.max_num_inference_steps();
    return std::move(args);
}
}  // namespace ovms

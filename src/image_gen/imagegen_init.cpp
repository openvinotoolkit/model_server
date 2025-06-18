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
#include <utility>

#include "src/filesystem.hpp"
#include "src/image_gen/image_gen_calculator.pb.h"
#include "src/json_parser.hpp"
#include "src/logging.hpp"
#include "src/status.hpp"

#include "imagegenutils.hpp"

namespace ovms {
std::variant<Status, std::optional<resolution_t>> getDimensionsConfig(const std::string& resolutionString) {
    auto dimsOrStatus = getDimensions(resolutionString);
    if (std::holds_alternative<absl::Status>(dimsOrStatus)) {
        auto statusString = std::get<absl::Status>(dimsOrStatus).ToString();
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to parse resolution: {}", statusString);
        return Status(StatusCode::SHAPE_WRONG_FORMAT, statusString);
    }
    return std::get<std::optional<resolution_t>>(dimsOrStatus);
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
    if (nodeOptions.has_device()) {
        args.device = nodeOptions.device();
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
    }
    args.maxNumImagesPerPrompt = nodeOptions.max_num_images_per_prompt();
    args.defaultNumInferenceSteps = nodeOptions.default_num_inference_steps();
    args.maxNumInferenceSteps = nodeOptions.max_num_inference_steps();
    return std::move(args);
}
}  // namespace ovms

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

#include "src/audio/speech_to_text/s2t_servable.hpp"

#include <stdexcept>

#include "openvino/genai/whisper_pipeline.hpp"
#include "src/audio/speech_to_text/s2t_calculator.pb.h"
#include "src/json_parser.hpp"
#include "src/logging.hpp"
#include "src/status.hpp"

namespace ovms {

SttServable::SttServable(const ::mediapipe::S2tCalculatorOptions& nodeOptions, const std::string& graphPath) {
    auto fsModelsPath = std::filesystem::path(nodeOptions.models_path());
    if (fsModelsPath.is_relative()) {
        parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
    } else {
        parsedModelsPath = fsModelsPath;
    }

    ov::AnyMap config;
    auto status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), config);
    if (!status.ok()) {
        SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
        throw std::runtime_error("Error during plugin_config option parsing");
    }

    enableWordTimestamps = nodeOptions.enable_word_timestamps();
    if (enableWordTimestamps && nodeOptions.target_device() == "NPU") {
        config["STATIC_PIPELINE"] = true;
    }
    config["word_timestamps"] = enableWordTimestamps;

    sttPipeline = std::make_shared<ov::genai::WhisperPipeline>(parsedModelsPath.string(), nodeOptions.target_device(), config);
}

}  // namespace ovms

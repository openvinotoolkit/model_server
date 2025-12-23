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
#pragma once
// cSpell:ignore genai

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "src/audio/text_to_speech/t2s_calculator.pb.h"
#include "src/status.hpp"
#include "src/logging.hpp"
#include "src/json_parser.hpp"

namespace ovms {

struct TtsServable {
    std::filesystem::path parsedModelsPath;
    std::shared_ptr<ov::genai::Text2SpeechPipeline> ttsPipeline;
    std::mutex ttsPipelineMutex;

    TtsServable(const mediapipe::T2sCalculatorOptions& nodeOptions, const std::string& graphPath) {
        auto fsModelsPath = std::filesystem::path(nodeOptions.models_path());
        if (fsModelsPath.is_relative()) {
            parsedModelsPath = (std::filesystem::path(graphPath) / fsModelsPath);
        } else {
            parsedModelsPath = fsModelsPath;
        }
        ov::AnyMap config;
        Status status = JsonParser::parsePluginConfig(nodeOptions.plugin_config(), config);
        if (!status.ok()) {
            SPDLOG_ERROR("Error during llm node plugin_config option parsing to JSON: {}", nodeOptions.plugin_config());
            throw std::runtime_error("Error during plugin_config option parsing");
        }
        ttsPipeline = std::make_shared<ov::genai::Text2SpeechPipeline>(parsedModelsPath.string(), nodeOptions.target_device(), config);
    }
};

using TtsServableMap = std::unordered_map<std::string, std::shared_ptr<TtsServable>>;
}  // namespace ovms

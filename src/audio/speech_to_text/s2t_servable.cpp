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

#include "s2t_servable.hpp"

#include <stdexcept>

#pragma warning(push)
#pragma warning(disable : 6386)
#include "absl/status/status.h"
#pragma warning(pop)
#include "openvino/genai/whisper_pipeline.hpp"

#include "src/audio/speech_to_text/s2t_calculator.pb.h"
#include "src/http_payload.hpp"
#include "src/json_parser.hpp"
#include "src/logging.hpp"
#include "src/stringutils.hpp"

namespace ovms {

namespace {
constexpr size_t ISO_LANG_CODE_MAX = 3;
}

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

    streamingExecutor = std::make_unique<SttExecutorWrapper>(sttPipeline, sttPipelineMutex);
}

void SttServable::addRequest(std::shared_ptr<SttServableExecutionContext> executionContext) {
    if (!streamingExecutor) {
        throw std::runtime_error("Cannot schedule STT streaming job - executor not initialized");
    }
    streamingExecutor->addRequest(std::move(executionContext));
}

absl::Status SttServable::parseTemperature(const HttpPayload& payload, ov::genai::WhisperGenerationConfig& config) {
    std::string temperatureStr = payload.multipartParser->getFieldByName("temperature");
    if (temperatureStr.size() > 0) {
        SPDLOG_LOGGER_TRACE(s2t_calculator_logger, "Received temperature: {}", temperatureStr);
        auto temp = ovms::stof(temperatureStr);
        if (!temp.has_value()) {
            temp = ovms::stou32(temperatureStr);
            if (!temp.has_value())
                return absl::InvalidArgumentError("Invalid temperature type.");
        }
        config.temperature = temp.value();
        if (config.temperature > 0) {
            config.do_sample = true;
        }
    }
    return absl::OkStatus();
}

absl::Status SttServable::updateTranscriptionConfig(ov::genai::WhisperGenerationConfig& config,
    const std::shared_ptr<SttServable>& servable, const HttpPayload& payload) {
    std::string language = payload.multipartParser->getFieldByName("language");
    if (language.size() > 0) {
        if (language.size() > ISO_LANG_CODE_MAX) {
            return absl::InvalidArgumentError("Invalid language code.");
        }
        SPDLOG_LOGGER_TRACE(s2t_calculator_logger, "Received language: {}", language);
        config.language = "<|" + language + "|>";
    }
    std::vector<std::string> timestampsTypes = payload.multipartParser->getArrayFieldByName("timestamp_granularities[]");
    config.word_timestamps = false;
    for (const auto& timestampsType : timestampsTypes) {
        SPDLOG_LOGGER_TRACE(s2t_calculator_logger, "Received timestamp type: {}", timestampsType);
        if (timestampsType == "segment") {
            config.return_timestamps = true;
        } else if (timestampsType == "word") {
            if (!servable->enableWordTimestamps)
                return absl::InvalidArgumentError("Word timestamps not supported for this model");
            config.word_timestamps = true;
        } else {
            return absl::InvalidArgumentError("Invalid timestamp_granularities type. Allowed types: \"segment\", \"word\"");
        }
    }
    auto status = parseTemperature(payload, config);
    if (status != absl::OkStatus())
        return status;
    return absl::OkStatus();
}

}  // namespace ovms

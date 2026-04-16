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
#include "s2t_streaming_handler.hpp"

#include <vector>

#include "src/http_payload.hpp"
#include "src/logging.hpp"
#include "src/stringutils.hpp"

namespace mediapipe {

static constexpr size_t ISO_LANG_CODE_MAX = 3;

absl::Status S2tStreamingHandler::parseTemperature(const ovms::HttpPayload& payload, float& temperature) {
    std::string temperatureStr = payload.multipartParser->getFieldByName("temperature");
    if (temperatureStr.size() > 0) {
        SPDLOG_LOGGER_TRACE(ovms::s2t_calculator_logger, "Received temperature: {}", temperatureStr);
        auto temp = ovms::stof(temperatureStr);
        if (!temp.has_value()) {
            temp = ovms::stou32(temperatureStr);
            if (!temp.has_value())
                return absl::InvalidArgumentError("Invalid temperature type.");
        }
        if (temp.value() < 0.0f || temp.value() > 2.0f)
            return absl::InvalidArgumentError("Temperature out of range(0.0, 2.0)");
        temperature = temp.value();
    } else {
        temperature = 1.0f;
    }
    return absl::OkStatus();
}

absl::Status S2tStreamingHandler::applyTranscriptionConfig(ov::genai::WhisperGenerationConfig& config,
    const std::shared_ptr<ovms::SttServable>& pipe, const ovms::HttpPayload& payload) {
    std::string language = payload.multipartParser->getFieldByName("language");
    if (language.size() > 0) {
        if (language.size() > ISO_LANG_CODE_MAX) {
            return absl::InvalidArgumentError("Invalid language code.");
        }
        SPDLOG_LOGGER_TRACE(ovms::s2t_calculator_logger, "Received language: {}", language);
        config.language = "<|" + language + "|>";
    }
    std::vector<std::string> timestampsTypes = payload.multipartParser->getArrayFieldByName("timestamp_granularities[]");
    config.word_timestamps = false;
    for (const auto& timestampsType : timestampsTypes) {
        SPDLOG_LOGGER_TRACE(ovms::s2t_calculator_logger, "Received timestamp type: {}", timestampsType);
        if (timestampsType == "segment") {
            config.return_timestamps = true;
        } else if (timestampsType == "word") {
            if (!pipe->enableWordTimestamps)
                return absl::InvalidArgumentError("Word timestamps not supported for this model");
            config.word_timestamps = true;
        } else {
            return absl::InvalidArgumentError("Invalid timestamp_granularities type. Allowed types: \"segment\", \"word\"");
        }
    }
    auto status = parseTemperature(payload, config.temperature);
    if (status != absl::OkStatus())
        return status;
    return absl::OkStatus();
}

}  // namespace mediapipe

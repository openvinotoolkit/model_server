//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include "audio_decoding_processor.hpp"

#include <cstring>
#include <string>
#include <variant>
#include <vector>
#include <utility>

#pragma warning(push)
#pragma warning(disable : 6001)
#include "absl/strings/escaping.h"
#pragma warning(pop)
#include <openvino/runtime/tensor.hpp>

#include "../../../logging.hpp"
#include "src/audio/audio_utils.hpp"

namespace ovms {

absl::Status AudioDecodingProcessor::process(InputRequest& req) {
    if (!std::holds_alternative<ov::genai::ChatHistory>(req.input)) {
        return absl::Status(absl::StatusCode::kInternal,
            "AudioDecodingProcessor received input that is not a ChatHistory");
    }
    auto& chatHistory = std::get<ov::genai::ChatHistory>(req.input);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "AudioDecodingProcessor: processing {} messages", chatHistory.size());

    for (size_t i = 0; i < chatHistory.size(); i++) {
        const auto content = chatHistory[i]["content"];
        if (content.as_string().value_or("").find("<ov_genai_audio_") != std::string::npos) {
            return absl::InvalidArgumentError("Message contains restricted <ov_genai_audio> tag");
        }
        if (!content.is_array()) {
            continue;
        }

        for (size_t j = 0; j < content.size(); j++) {
            const auto part = content[j];
            const auto type = part["type"].as_string().value_or("");
            if (type == "text" && part["text"].as_string().value_or("").find("<ov_genai_audio_") != std::string::npos) {
                return absl::InvalidArgumentError("Message contains restricted <ov_genai_audio> tag");
            }

            if (type == "input_audio") {
                const auto data = part["input_audio"]["data"].as_string().value_or("");
                const auto format = part["input_audio"]["format"].as_string().value_or("wav");

                if (data.empty()) {
                    return absl::InvalidArgumentError("input_audio data field is empty");
                }

                std::string decoded;
                if (!absl::Base64Unescape(data, &decoded)) {
                    return absl::InvalidArgumentError("Invalid base64 string in input_audio data");
                }

                try {
                    std::vector<float> pcm = ovms::audio_utils::readWithoutResample(
                        std::string_view(decoded.data(), decoded.size()), format);
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "AudioDecodingProcessor: decoded audio {} samples, format='{}'",
                        pcm.size(), format);
                    ov::Tensor audioTensor(ov::element::f32, ov::Shape{pcm.size()});
                    std::memcpy(audioTensor.data<float>(), pcm.data(), pcm.size() * sizeof(float));
                    req.inputAudios.push_back(std::move(audioTensor));
                    const std::string tag = "<ov_genai_audio_" + std::to_string(req.inputAudios.size() - 1) + ">";
                    content[j] = ov::genai::JsonContainer({{"type", "text"}, {"text", tag}});
                } catch (const std::exception& e) {
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Audio decoding failed: {}", e.what());
                    return absl::InvalidArgumentError(std::string("Audio decoding failed: ") + e.what());
                }
            }
        }
    }

    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "AudioDecodingProcessor: total audios collected: {}", req.inputAudios.size());
    return absl::OkStatus();
}

}  // namespace ovms

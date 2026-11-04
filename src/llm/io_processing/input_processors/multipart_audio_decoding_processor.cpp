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

#include "multipart_audio_decoding_processor.hpp"

#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include <openvino/runtime/tensor.hpp>

#include "../../../logging.hpp"
#include "src/audio/audio_utils.hpp"

namespace ovms {

absl::Status MultipartAudioDecodingProcessor::process(InputRequest& req) {
    for (const MultipartAudioInput& input : req.multipartAudioInputs) {
        try {
            std::vector<float> pcm;
            if (audio_utils::isWavBuffer(input.data)) {
                pcm = audio_utils::readWav(input.data);
            } else {
                pcm = audio_utils::readMp3(input.data);
            }
            ov::Tensor audioTensor(ov::element::f32, ov::Shape{pcm.size()});
            std::memcpy(audioTensor.data<float>(), pcm.data(), pcm.size() * sizeof(float));
            req.inputAudios.push_back(std::move(audioTensor));
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Multipart audio decoding failed: {}", e.what());
            return absl::InvalidArgumentError(std::string("Audio decoding failed: ") + e.what());
        }
    }
    return absl::OkStatus();
}

}  // namespace ovms
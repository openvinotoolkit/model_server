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
//*****************************************************************************/
#include "mock_echo_streaming_audio_model.hpp"

#include "src/logging.hpp"

#include <cmath>

namespace ovms {

MockEchoStreamingAudioModel::MockEchoStreamingAudioModel(uint32_t sampleRate) :
    sampleRate_(sampleRate) {}

AudioChunk MockEchoStreamingAudioModel::process(const AudioChunk& input) {
    if (input.sampleRate != sampleRate_) {
        throw std::invalid_argument("Unexpected audio sample rate");
    }

    if (webrtc_logger->should_log(spdlog::level::trace)) {
        float squaredSampleSum = 0.0f;
        for (const float sample : input.samples) {
            squaredSampleSum += sample * sample;
        }
        const float rmsVolume = input.samples.empty() ? 0.0f : std::sqrt(squaredSampleSum / input.samples.size());
        SPDLOG_LOGGER_TRACE(webrtc_logger, "Mock audio model input RMS volume: {}, timestamp: {}", rmsVolume, input.timestampUs);
    }

    return input;
}

}  // namespace ovms

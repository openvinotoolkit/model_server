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

#include <algorithm>

namespace ovms {

MockEchoStreamingAudioModel::MockEchoStreamingAudioModel(uint32_t sampleRate, float directGain, float echoGain) :
    sampleRate_(sampleRate),
    directGain_(directGain),
    echoGain_(echoGain) {}

AudioChunk MockEchoStreamingAudioModel::process(const AudioChunk& input) {
    if (input.sampleRate != sampleRate_) {
        throw std::invalid_argument("Unexpected audio sample rate");
    }

    AudioChunk output{input.samples, input.sampleRate, input.timestampUs};
    for (size_t index = 0; index < input.samples.size(); ++index) {
        const float delayedSample = index < previousChunk_.size() ? previousChunk_[index] : 0.0f;
        const float value = directGain_ * input.samples[index] + echoGain_ * delayedSample;
        output.samples[index] = std::clamp(value, -1.0f, 1.0f);
    }
    previousChunk_ = input.samples;
    return output;
}

}  // namespace ovms

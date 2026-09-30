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
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "streaming_audio_model.hpp"

namespace ovms {

class AudioUtteranceBuffer {
public:
    struct Config {
        float rmsThreshold = 0.01f;
        size_t minimumSpeechFrames = 2;
        size_t silenceHangoverFrames = 15;
    };

    AudioUtteranceBuffer();
    explicit AudioUtteranceBuffer(Config config);

    std::optional<AudioChunk> push(const AudioChunk& frame);
    void reset();

private:
    bool isSpeech(const AudioChunk& frame) const;

    Config config_;
    std::vector<float> samples_;
    uint32_t sampleRate_ = 0;
    uint32_t channels_ = 0;
    uint64_t startTimestampUs_ = 0;
    size_t speechFrames_ = 0;
    size_t silentFrames_ = 0;
};

}  // namespace ovms

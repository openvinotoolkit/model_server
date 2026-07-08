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

#include <string>
#include <vector>
#include <stdint.h>
#include <string_view>

namespace ovms::audio_utils {

bool isWavBuffer(const std::string buf);

std::vector<float> readWav(const std::string_view& wavData);
std::vector<float> readMp3(const std::string_view& mp3Data);

// Decode WAV or MP3 audio data into mono float32 PCM samples.
// No resampling is applied — returns the raw PCM at the file's native sample rate.
std::vector<float> readWithoutResample(const std::string_view& audioData, const std::string& format);

void prepareAudioOutput(void** ppData, size_t& pDataSize, uint32_t sampleRate, uint16_t bitsPerSample, size_t speechSize, const float* waveformPtr);

// Throws if the estimated resampled audio buffer size would exceed the maximum allowed size
void validateAudioFileSize(
    size_t inputSamples,
    uint32_t inputRate,
    uint32_t targetRate,
    uint32_t channels,
    size_t bytesPerSample);

}  // namespace ovms::audio_utils

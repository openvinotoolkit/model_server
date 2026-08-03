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
#pragma once

#include <cstdint>
#include <string>

namespace ovms {
namespace test_utils {

// Builds a minimal valid PCM16 mono WAV buffer with the given sample rate and
// number of samples. Useful for unit tests that need syntactically correct WAV
// data without depending on external files.
inline std::string buildWavBuffer(uint32_t numSamples, uint32_t sampleRate = 16000) {
    constexpr uint16_t channels = 1;
    constexpr uint16_t bitsPerSample = 16;
    const uint32_t byteRate = sampleRate * channels * bitsPerSample / 8;
    const uint16_t blockAlign = channels * bitsPerSample / 8;
    const uint32_t dataSize = numSamples * blockAlign;
    const uint32_t riffSize = 36 + dataSize;

    std::string out;
    out.reserve(44 + dataSize);
    auto append = [&out](const void* p, size_t n) {
        out.append(reinterpret_cast<const char*>(p), n);
    };
    out.append("RIFF", 4);
    append(&riffSize, 4);
    out.append("WAVE", 4);
    out.append("fmt ", 4);
    uint32_t fmtChunkSize = 16;
    uint16_t audioFormat = 1;  // PCM
    append(&fmtChunkSize, 4);
    append(&audioFormat, 2);
    append(&channels, 2);
    append(&sampleRate, 4);
    append(&byteRate, 4);
    append(&blockAlign, 2);
    append(&bitsPerSample, 2);
    out.append("data", 4);
    append(&dataSize, 4);
    out.append(dataSize, '\0');
    return out;
}

}  // namespace test_utils
}  // namespace ovms

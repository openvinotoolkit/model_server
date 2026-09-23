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

#include <cstdint>
#include <memory>
#include <vector>

struct OpusDecoder;
struct OpusEncoder;

namespace ovms {

class OpusAudioCodec {
public:
    static constexpr uint32_t SampleRate = 48000;
    static constexpr size_t Channels = 1;
    static constexpr size_t FrameSamples = 960;

    OpusAudioCodec(uint32_t sampleRate, size_t channels);
    ~OpusAudioCodec();

    OpusAudioCodec(const OpusAudioCodec&) = delete;
    OpusAudioCodec& operator=(const OpusAudioCodec&) = delete;

    std::vector<uint8_t> encode(const std::vector<float>& pcm) const;
    std::vector<float> decode(const std::vector<uint8_t>& encoded) const;

private:
    std::unique_ptr<OpusEncoder, void (*)(OpusEncoder*)> encoder_;
    std::unique_ptr<OpusDecoder, void (*)(OpusDecoder*)> decoder_;
};

}  // namespace ovms

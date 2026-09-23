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
#include "opus_audio_codec.hpp"

#include <opus/opus.h>

#include <stdexcept>

namespace ovms {

namespace {

void destroyEncoder(OpusEncoder* encoder) {
    opus_encoder_destroy(encoder);
}

void destroyDecoder(OpusDecoder* decoder) {
    opus_decoder_destroy(decoder);
}

}  // namespace

OpusAudioCodec::OpusAudioCodec(uint32_t sampleRate, size_t channels) :
    encoder_(nullptr, destroyEncoder),
    decoder_(nullptr, destroyDecoder) {
    if (sampleRate != SampleRate || channels != Channels) {
        throw std::invalid_argument("Only mono 48 kHz Opus is supported");
    }

    int error = OPUS_OK;
    encoder_.reset(opus_encoder_create(static_cast<opus_int32>(sampleRate), static_cast<int>(channels), OPUS_APPLICATION_AUDIO, &error));
    if (error != OPUS_OK || !encoder_) {
        throw std::runtime_error("Opus encoder creation failed");
    }

    decoder_.reset(opus_decoder_create(static_cast<opus_int32>(sampleRate), static_cast<int>(channels), &error));
    if (error != OPUS_OK || !decoder_) {
        throw std::runtime_error("Opus decoder creation failed");
    }
}

OpusAudioCodec::~OpusAudioCodec() = default;

std::vector<uint8_t> OpusAudioCodec::encode(const std::vector<float>& pcm) const {
    if (pcm.size() != FrameSamples) {
        throw std::invalid_argument("Opus input must contain one 20 ms frame");
    }

    std::vector<uint8_t> encoded(4000);
    const int encodedSize = opus_encode_float(encoder_.get(), pcm.data(), static_cast<int>(FrameSamples), encoded.data(), encoded.size());
    if (encodedSize < 0) {
        throw std::runtime_error("Opus encoding failed");
    }
    encoded.resize(static_cast<size_t>(encodedSize));
    return encoded;
}

std::vector<float> OpusAudioCodec::decode(const std::vector<uint8_t>& encoded) const {
    if (encoded.empty()) {
        throw std::invalid_argument("Opus input packet is empty");
    }

    std::vector<float> decoded(FrameSamples);
    const int decodedSamples = opus_decode_float(decoder_.get(), encoded.data(), encoded.size(), decoded.data(), static_cast<int>(FrameSamples), 0);
    if (decodedSamples < 0) {
        throw std::runtime_error("Opus decoding failed");
    }
    decoded.resize(static_cast<size_t>(decodedSamples));
    return decoded;
}

}  // namespace ovms

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
#include "streaming_audio_processor.hpp"

namespace ovms {

StreamingAudioProcessor::StreamingAudioProcessor(OpusAudioCodec& codec, StreamingAudioModel& model) :
    codec_(codec),
    model_(model) {}

std::vector<uint8_t> StreamingAudioProcessor::process(const std::vector<uint8_t>& encodedInput, uint64_t timestampUs) {
    const auto decoded = codec_.decode(encodedInput);
    const auto output = model_.process(AudioChunk{decoded, OpusAudioCodec::SampleRate, timestampUs});
    lastTimestampUs_ = output.timestampUs;
    return codec_.encode(output.samples);
}

uint64_t StreamingAudioProcessor::lastTimestampUs() const {
    return lastTimestampUs_;
}

}  // namespace ovms

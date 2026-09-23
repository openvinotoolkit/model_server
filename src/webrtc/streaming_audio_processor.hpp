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
#include <vector>

#include "opus_audio_codec.hpp"
#include "streaming_audio_model.hpp"

namespace ovms {

class StreamingAudioProcessor {
public:
    StreamingAudioProcessor(OpusAudioCodec& codec, StreamingAudioModel& model);

    std::vector<uint8_t> process(const std::vector<uint8_t>& encodedInput, uint64_t timestampUs);
    uint64_t lastTimestampUs() const;

private:
    OpusAudioCodec& codec_;
    StreamingAudioModel& model_;
    uint64_t lastTimestampUs_ = 0;
};

}  // namespace ovms

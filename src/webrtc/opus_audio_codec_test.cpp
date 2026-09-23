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
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "mock_echo_streaming_audio_model.hpp"
#include "opus_audio_codec.hpp"
#include "streaming_audio_processor.hpp"

using ovms::AudioChunk;
using ovms::MockEchoStreamingAudioModel;
using ovms::OpusAudioCodec;
using ovms::StreamingAudioProcessor;

TEST(OpusAudioCodecTest, RejectsUnsupportedConfiguration) {
    EXPECT_THROW(OpusAudioCodec(16000, 1), std::invalid_argument);
    EXPECT_THROW(OpusAudioCodec(48000, 2), std::invalid_argument);
}

TEST(OpusAudioCodecTest, EncodesAndDecodesTwentyMillisecondMonoFrame) {
    OpusAudioCodec codec(48000, 1);
    std::vector<float> input(960);
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] = 0.2f * std::sin(static_cast<float>(index) * 0.05f);
    }

    const auto encoded = codec.encode(input);
    const auto decoded = codec.decode(encoded);

    ASSERT_FALSE(encoded.empty());
    ASSERT_EQ(decoded.size(), input.size());
    float maximumError = 0.0f;
    for (size_t index = 0; index < input.size(); ++index) {
        maximumError = std::max(maximumError, std::abs(input[index] - decoded[index]));
    }
    EXPECT_LT(maximumError, 0.5f);
}

TEST(StreamingAudioProcessorTest, DecodesProcessesAndReencodesWithTimestamp) {
    OpusAudioCodec codec(48000, 1);
    MockEchoStreamingAudioModel model(48000, 0.5f, 0.5f);
    StreamingAudioProcessor processor(codec, model);
    std::vector<float> input(960, 0.25f);
    const auto encodedInput = codec.encode(input);

    const auto encodedOutput = processor.process(encodedInput, 123456);
    const auto decodedOutput = codec.decode(encodedOutput);

    ASSERT_EQ(decodedOutput.size(), input.size());
    EXPECT_NE(decodedOutput, input);
    EXPECT_EQ(processor.lastTimestampUs(), 123456);
}

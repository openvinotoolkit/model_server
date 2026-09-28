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
#include <stdexcept>

#include <gtest/gtest.h>

#include "mock_echo_streaming_audio_model.hpp"

using ovms::AudioChunk;
using ovms::MockEchoStreamingAudioModel;

TEST(MockEchoStreamingAudioModelTest, ReturnsAudioSamplesUnchanged) {
    MockEchoStreamingAudioModel model(1760);
    const AudioChunk input{{0.25f, -0.5f, 0.75f}, 1760, 123456, 1};

    const auto output = model.process(input);

    EXPECT_EQ(output.samples, input.samples);
    EXPECT_EQ(output.sampleRate, input.sampleRate);
    EXPECT_EQ(output.timestampUs, input.timestampUs);
    EXPECT_EQ(output.channels, input.channels);
}

TEST(MockEchoStreamingAudioModelTest, RejectsUnexpectedSampleRate) {
    MockEchoStreamingAudioModel model(2);
    EXPECT_THROW(model.process(AudioChunk{{1.0f}, 48000, 0}), std::invalid_argument);
}

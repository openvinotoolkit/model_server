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

TEST(MockEchoStreamingAudioModelTest, PreservesChunkShapeAndAppliesEcho) {
    MockEchoStreamingAudioModel model(2, 0.5f, 0.5f);
    AudioChunk first{{1.0f, 0.0f}, 2, 0};
    AudioChunk second{{0.0f, 1.0f}, 2, 1000000};

    const auto firstOutput = model.process(first);
    const auto secondOutput = model.process(second);

    EXPECT_EQ(firstOutput.samples.size(), first.samples.size());
    EXPECT_FLOAT_EQ(firstOutput.samples[0], 0.5f);
    EXPECT_FLOAT_EQ(firstOutput.samples[1], 0.0f);
    EXPECT_FLOAT_EQ(secondOutput.samples[0], 0.5f);
    EXPECT_FLOAT_EQ(secondOutput.samples[1], 0.5f);
    EXPECT_EQ(secondOutput.timestampUs, second.timestampUs);
}

TEST(MockEchoStreamingAudioModelTest, RejectsUnexpectedSampleRate) {
    MockEchoStreamingAudioModel model(2, 0.5f, 0.5f);
    EXPECT_THROW(model.process(AudioChunk{{1.0f}, 48000, 0}), std::invalid_argument);
}

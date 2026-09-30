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
#include "audio_utterance_buffer.hpp"

#include <gtest/gtest.h>

namespace ovms {

TEST(AudioUtteranceBufferTest, EmitsAfterSpeechAndSilenceHangover) {
    AudioUtteranceBuffer buffer({0.1f, 1, 2});
    const AudioChunk speech{{0.2f, 0.2f}, 48000, 1000, 1};
    const AudioChunk silence{{0.0f, 0.0f}, 48000, 1020, 1};

    EXPECT_FALSE(buffer.push(speech).has_value());
    EXPECT_FALSE(buffer.push(silence).has_value());
    const auto utterance = buffer.push(silence);

    ASSERT_TRUE(utterance.has_value());
    EXPECT_EQ(utterance->samples, (std::vector<float>{0.2f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f}));
    EXPECT_EQ(utterance->sampleRate, 48000);
    EXPECT_EQ(utterance->timestampUs, 1000);
}

TEST(AudioUtteranceBufferTest, IgnoresSilenceBeforeSpeech) {
    AudioUtteranceBuffer buffer({0.1f, 1, 1});
    const AudioChunk silence{{0.0f, 0.0f}, 48000, 1000, 1};

    for (uint64_t frame = 0; frame < 1000; ++frame)
        EXPECT_FALSE(buffer.push(silence).has_value());
    const AudioChunk speech{{0.2f, 0.2f}, 48000, 21000, 1};
    EXPECT_FALSE(buffer.push(speech).has_value());
    const auto utterance = buffer.push(silence);
    ASSERT_TRUE(utterance.has_value());
    EXPECT_EQ(utterance->samples, (std::vector<float>{0.2f, 0.2f, 0.0f, 0.0f}));
    EXPECT_EQ(utterance->timestampUs, speech.timestampUs);
}

}  // namespace ovms

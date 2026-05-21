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
#include <cstdint>
#include <cstring>
#include <cstdlib>
#if defined(_WIN32)
#include <stdlib.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../audio/audio_utils.hpp"
#include "../../utils/env_guard.hpp"

namespace {

// Builds an in-memory PCM16 mono WAV file with an attacker-controlled sample rate
// and an arbitrary number of zeroed PCM samples. Used to reproduce the DoS report
// where a 1 Hz sample rate caused readWav() to allocate gigabytes during resampling.
std::string buildWavBuffer(uint32_t sampleRate, uint32_t numSamples) {
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

class AudioUtilsSampleRateTest : public ::testing::Test {
protected:
    void SetUp() override {
        UnSetEnvironmentVar("OVMS_AUDIO_MIN_SAMPLE_RATE");
        UnSetEnvironmentVar("OVMS_AUDIO_MAX_SAMPLE_RATE");
    }
};

TEST_F(AudioUtilsSampleRateTest, rejectsExtremelyLowSampleRate) {
    // PoC from the security report: 1 Hz sample rate + 100k samples
    // would otherwise allocate ~6.4 GB in readWav().
    const std::string wav = buildWavBuffer(/*sampleRate=*/1, /*numSamples=*/100000);
    std::string_view view(wav);
    EXPECT_THROW(readWav(view), std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, rejectsExtremelyHighSampleRate) {
    const std::string wav = buildWavBuffer(/*sampleRate=*/10000000u, /*numSamples=*/16);
    std::string_view view(wav);
    EXPECT_THROW(readWav(view), std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, acceptsSupportedSampleRateWithoutResampling) {
    const std::string wav = buildWavBuffer(/*sampleRate=*/16000, /*numSamples=*/16);
    std::string_view view(wav);
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readWav(view); });
    EXPECT_EQ(decoded.size(), 16u);
}

TEST_F(AudioUtilsSampleRateTest, acceptsCommonSampleRateRequiringResampling) {
    const std::string wav = buildWavBuffer(/*sampleRate=*/8000, /*numSamples=*/16);
    std::string_view view(wav);
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readWav(view); });
    // 8000 -> 16000 doubles the sample count.
    EXPECT_EQ(decoded.size(), 32u);
}

// Builds a minimal MPEG-1 Layer III header followed by a Xing tag that lies about
// the FRAMES field. dr_mp3 reads that field verbatim into totalPCMFrameCount, so
// the old pre-allocation path would request multi-TB of RAM in readMp3. With the
// streaming decode in readMp3, output capacity is bounded by what the decoder
// actually produces from the input bytes, regardless of the declared count.
TEST_F(AudioUtilsSampleRateTest, decodesMp3WithInflatedXingFrameCountWithoutOomAllocation) {
    // MPEG-1 Layer III, 128 kbps, 44.1 kHz, mono, no padding, no CRC.
    // 4-byte sync header: 0xFF 0xFB 0x90 0x40
    std::string mp3;
    mp3.reserve(417);
    mp3.push_back(static_cast<char>(0xFF));
    mp3.push_back(static_cast<char>(0xFB));
    mp3.push_back(static_cast<char>(0x90));
    mp3.push_back(static_cast<char>(0x40));
    // Side-info area for MPEG-1 mono is 17 bytes; pad with zeros up to the Xing tag offset.
    mp3.append(17, '\0');
    // "Xing" identifier + flags byte 0x01 (FRAMES present) + claim 0xFFFFFFFE frames.
    // 0xFFFFFFFF is dr_mp3's "unknown" sentinel, so we use one less to force
    // totalPCMFrameCount = 0xFFFFFFFE * 1152 ~= 4.9e12 (about 19.6 TB if trusted).
    mp3.append("Xing", 4);
    mp3.append(3, '\0');
    mp3.push_back(static_cast<char>(0x01));
    mp3.push_back(static_cast<char>(0xFF));
    mp3.push_back(static_cast<char>(0xFF));
    mp3.push_back(static_cast<char>(0xFF));
    mp3.push_back(static_cast<char>(0xFE));
    // Pad the rest of the 417-byte MPEG-1 Layer III frame at 128 kbps / 44.1 kHz.
    mp3.append(417 - mp3.size(), '\0');

    std::string_view view(mp3);
    std::vector<float> decoded;
    // Must not OOM, must not throw on size grounds. Whether dr_mp3 manages to extract
    // any PCM from the synthetic (all-zero) Huffman payload is irrelevant - the
    // important property is that the output stays proportional to the actual input
    // bytes rather than the attacker-declared frame count.
    try {
        decoded = readMp3(view);
    } catch (const std::runtime_error&) {
        // Acceptable: dr_mp3 may decide the synthetic frame is unparseable. The
        // attack surface we are guarding against is OOM, not validity rejection.
        return;
    }
    // Upper bound: a single MPEG-1 Layer III frame yields at most 1152 PCM frames
    // per channel; allow a couple of frames of slack but assert the result is
    // bounded by the input, not by the declared FRAMES tag.
    constexpr size_t MAX_PLAUSIBLE_PCM_FRAMES = 1152u * 2u * 2u;
    EXPECT_LE(decoded.size(), MAX_PLAUSIBLE_PCM_FRAMES);
}

}  // namespace

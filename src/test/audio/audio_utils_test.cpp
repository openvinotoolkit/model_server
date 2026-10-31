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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <array>
#include <cstdlib>
#include <limits>
#if defined(_WIN32)
#include <stdlib.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>
#include <string_view>

#include <gtest/gtest.h>

#include "../../audio/audio_utils.hpp"
#include "../../port/dr_audio.hpp"
#include "../../utils/env_guard.hpp"

using namespace ovms::audio_utils;

namespace {

uint16_t readLe16(const char* ptr) {
    return static_cast<uint16_t>(static_cast<unsigned char>(ptr[0])) |
           (static_cast<uint16_t>(static_cast<unsigned char>(ptr[1])) << 8);
}

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

class AudioUtilsSampleRateTest : public ::testing::Test {};

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
        // Acceptable: dr_mp3 may decide the synthetic frame is unparsable. The
        // attack surface we are guarding against is OOM, not validity rejection.
        return;
    }
    // Upper bound: a single MPEG-1 Layer III frame yields at most 1152 PCM frames
    // per channel; allow a couple of frames of slack but assert the result is
    // bounded by the input, not by the declared FRAMES tag.
    constexpr size_t MAX_PLAUSIBLE_PCM_FRAMES = 1152u * 2u * 2u;
    EXPECT_LE(decoded.size(), MAX_PLAUSIBLE_PCM_FRAMES);
}

TEST_F(AudioUtilsSampleRateTest, wavFileRejectedWhenExceedsMaxFileSizeEnv) {
    const std::string wav = buildWavBuffer(/*sampleRate=*/16000, /*numSamples=*/16);
    std::string_view view(wav);
    size_t expectedDecodedSize = (size_t)(16 * sizeof(float));
    // Set max file size to one less than needed
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", std::to_string(expectedDecodedSize - 1));
    std::vector<float> decoded;
    EXPECT_THROW({ decoded = readWav(view); }, std::runtime_error);
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, wavFileAcceptedWhenAtMaxFileSizeEnv) {
    const std::string wav = buildWavBuffer(/*sampleRate=*/32000, /*numSamples=*/16);
    std::string_view view(wav);
    size_t expectedDecodedSize = static_cast<size_t>(8 * sizeof(float));
    // Set max file size to exactly the needed size
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", std::to_string(expectedDecodedSize));
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readWav(view); });
    EXPECT_EQ(decoded.size(), 8u);
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, readWithoutResampleWavRejectedWhenExceedsMaxFileSizeEnv) {
    const std::string wav = buildWavBuffer(/*sampleRate=*/16000, /*numSamples=*/16);
    std::string_view view(wav);
    size_t expectedDecodedSize = static_cast<size_t>(16 * sizeof(float));
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", std::to_string(expectedDecodedSize - 1));
    EXPECT_THROW({ auto decoded = readWithoutResample(view, "wav"); }, std::runtime_error);
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, readWithoutResampleWavAcceptedWhenAtMaxFileSizeEnv) {
    const std::string wav = buildWavBuffer(/*sampleRate=*/32000, /*numSamples=*/16);
    std::string_view view(wav);
    // Native-rate WAV decode keeps the full 16 float samples, or 64 bytes total.
    size_t expectedDecodedSize = static_cast<size_t>(16 * sizeof(float));
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", std::to_string(expectedDecodedSize));
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readWithoutResample(view, "wav"); });
    EXPECT_EQ(decoded.size(), 16u);
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, mp3FileRejectedWhenExceedsMaxFileSizeEnv) {
    // Minimal valid MP3 frame (see previous test for structure)
    std::string mp3;
    mp3.reserve(417);
    mp3.push_back(static_cast<char>(0xFF));
    mp3.push_back(static_cast<char>(0xFB));
    mp3.push_back(static_cast<char>(0x90));
    mp3.push_back(static_cast<char>(0x40));
    mp3.append(413, '\0');
    std::string_view view(mp3);
    // This joint-stereo frame decodes to 1152 PCM frames, returned as 1152 mono samples.
    size_t expectedDecodedSize = 1152 * sizeof(float);
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", std::to_string(expectedDecodedSize - 1));
    std::vector<float> decoded;
    EXPECT_THROW({ decoded = readMp3(view); }, std::runtime_error);
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, mp3FileAcceptedWhenAtMaxFileSizeEnv) {
    std::string mp3;
    mp3.reserve(417);
    mp3.push_back(static_cast<char>(0xFF));
    mp3.push_back(static_cast<char>(0xFB));
    mp3.push_back(static_cast<char>(0x90));
    mp3.push_back(static_cast<char>(0x40));
    mp3.append(413, '\0');
    std::string_view view(mp3);
    // This joint-stereo frame decodes to 1152 PCM frames, returned as 1152 mono samples.
    size_t expectedDecodedSize = 1152 * sizeof(float);
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", std::to_string(expectedDecodedSize));
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readMp3(view); });
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

namespace {
// 6 frames of 44.1 kHz stereo MP3 (about 0.157 s), 440 Hz on the left channel and 1320 Hz on the
// right, so the two channels are clearly distinguishable and the stream spans several decode chunks.
// clang-format off
const unsigned char STEREO_MP3_DISTINCT_CHANNELS[] = {
    0xFF, 0xFB, 0x30, 0x04, 0x00, 0x00, 0x00, 0xBD, 0x04, 0x56, 0x55, 0x0C, 0x00, 0x0C, 0x1A, 0xE0,
    0x1B, 0x0A, 0xA0, 0x88, 0x01, 0xC3, 0xB4, 0x33, 0x44, 0x19, 0xA1, 0x00, 0x00, 0x7D, 0x86, 0x6A,
    0x03, 0x33, 0x80, 0x00, 0x00, 0x16, 0xF0, 0x0C, 0x00, 0x00, 0x07, 0x77, 0x38, 0x80, 0x02, 0x03,
    0x22, 0x70, 0x7C, 0xBB, 0xDC, 0x08, 0x1C, 0xFF, 0xE5, 0xC1, 0xF0, 0xFD, 0x54, 0xA0, 0xEE, 0x50,
    0x00, 0x00, 0x0C, 0x3C, 0x3C, 0x3C, 0xF3, 0x00, 0x0C, 0xE8, 0x60, 0x02, 0x9F, 0xAF, 0xCC, 0x3C,
    0xFF, 0xC0, 0x00, 0x47, 0xFC, 0x70, 0xF4, 0x6E, 0x11, 0xDE, 0x98, 0x0B, 0xC6, 0xB0, 0x0D, 0x10,
    0x06, 0x8B, 0x12, 0x12, 0xE1, 0xB1, 0xB2, 0x97, 0x5F, 0xA8, 0xD5, 0xFF, 0xEF, 0x85, 0x13, 0xE1,
    0x44, 0x5C, 0x2A, 0x0A, 0xD6, 0xA6, 0xFF, 0xFF, 0x86, 0xA5, 0x3F, 0xBF, 0xDB, 0xA2, 0x82, 0xB4,
    0x02, 0x28, 0x25, 0x05, 0xC6, 0x0E, 0x4C, 0x3A, 0x81, 0xF6, 0x84, 0x8D, 0x9C, 0xA4, 0x1A, 0x76,
    0x15, 0x05, 0x7E, 0xC5, 0x70, 0x5C, 0x10, 0x20, 0x02, 0x92, 0x82, 0x30, 0xFF, 0xFB, 0x32, 0x04,
    0x04, 0x0F, 0x30, 0xE9, 0x07, 0x49, 0x87, 0x74, 0x00, 0x08, 0x1F, 0x41, 0xA9, 0x78, 0xEA, 0xE8,
    0x01, 0x03, 0xC4, 0x1F, 0x1E, 0x0C, 0x7B, 0x02, 0x60, 0x78, 0x86, 0xE4, 0x80, 0x0E, 0xE8, 0xA4,
    0x18, 0xD8, 0x7C, 0x24, 0xD2, 0xC2, 0x9C, 0xC4, 0xA0, 0x9C, 0xC1, 0x40, 0x24, 0x12, 0x01, 0x16,
    0x48, 0x14, 0x01, 0x35, 0xEE, 0x80, 0x08, 0x06, 0x00, 0x7F, 0xE7, 0x0A, 0xC7, 0x07, 0x58, 0x9F,
    0x43, 0x5E, 0x06, 0x08, 0x57, 0x01, 0xFF, 0x94, 0x27, 0x45, 0x10, 0x3A, 0xD3, 0x8C, 0xD2, 0x5F,
    0x9B, 0x35, 0xAD, 0x43, 0x51, 0xB5, 0x84, 0x2C, 0xB9, 0xC3, 0x06, 0x08, 0xE1, 0x0A, 0x64, 0xCC,
    0x95, 0x66, 0x4A, 0xE1, 0x0A, 0x60, 0x90, 0x03, 0xC7, 0x56, 0x80, 0x42, 0x59, 0x40, 0x2C, 0x55,
    0x3C, 0xB3, 0x3F, 0xEE, 0x35, 0x6B, 0x50, 0xC1, 0x8D, 0x55, 0x0E, 0xC2, 0x20, 0x94, 0xC2, 0x57,
    0x20, 0xE6, 0xD6, 0x74, 0x00, 0xBC, 0xF4, 0x92, 0x07, 0x15, 0x4A, 0xE7, 0x16, 0x1E, 0x96, 0xDA,
    0xE2, 0x58, 0x64, 0xBD, 0x0A, 0x80, 0xC5, 0xB3, 0x02, 0xFF, 0xFB, 0x32, 0x04, 0x04, 0x0F, 0xF0,
    0xE4, 0x07, 0xC8, 0x03, 0x1E, 0xE0, 0x88, 0x1B, 0xC1, 0x99, 0x30, 0x07, 0xBA, 0x45, 0x44, 0x0C,
    0x1D, 0x1A, 0x0C, 0xFB, 0x22, 0x68, 0x83, 0x06, 0xE4, 0x00, 0x0E, 0xEC, 0xA5, 0xB0, 0x8A, 0x31,
    0xB6, 0x4E, 0xA3, 0xD5, 0xA1, 0x8C, 0x48, 0x13, 0x53, 0x67, 0x25, 0x22, 0x80, 0x80, 0x35, 0x73,
    0x3D, 0xCF, 0xE5, 0x05, 0x3C, 0x61, 0xCB, 0x5D, 0x85, 0xCB, 0x30, 0x38, 0x10, 0x31, 0x94, 0xD3,
    0x3E, 0xBC, 0xE0, 0x1B, 0x7A, 0x61, 0x88, 0x2D, 0xB7, 0xA2, 0x86, 0xE5, 0xDA, 0x5F, 0xC4, 0x8B,
    0x31, 0x04, 0x37, 0xE7, 0x30, 0x68, 0x0A, 0xC3, 0x33, 0x95, 0xDC, 0x33, 0x24, 0x0A, 0xE3, 0x06,
    0xB0, 0x2F, 0x38, 0x96, 0x31, 0x8A, 0x30, 0xC7, 0x30, 0x9E, 0x41, 0xC9, 0x78, 0x1F, 0xF3, 0x2A,
    0xD3, 0x30, 0x63, 0x63, 0x55, 0x02, 0x50, 0x24, 0x74, 0x3C, 0x30, 0x8A, 0x71, 0x3A, 0x52, 0x3A,
    0x2C, 0x22, 0x1C, 0xA0, 0xD8, 0x18, 0xC8, 0xBD, 0x2C, 0x19, 0xF9, 0x8B, 0x59, 0xED, 0x55, 0x58,
    0x54, 0xBE, 0x0A, 0x0C, 0xC7, 0xC3, 0xFF, 0xFB, 0x32, 0x04, 0x03, 0x8F, 0xF0, 0xE3, 0x07, 0xC8,
    0x03, 0x1E, 0xE0, 0x88, 0x1B, 0xE1, 0x99, 0x30, 0x00, 0x3D, 0x23, 0x43, 0xD4, 0x1F, 0x1C, 0x0C,
    0x7B, 0x02, 0x60, 0x85, 0x06, 0xE4, 0x00, 0x0E, 0xEC, 0xA5, 0x02, 0x80, 0x8F, 0x31, 0xA5, 0x50,
    0x73, 0xCC, 0x22, 0xCC, 0x46, 0x14, 0x54, 0xAE, 0x52, 0x3E, 0x81, 0x40, 0x4A, 0xEA, 0x77, 0xB3,
    0x27, 0x49, 0x1B, 0x77, 0x17, 0x7A, 0x00, 0x0C, 0x0F, 0x02, 0xCC, 0x66, 0x28, 0x8F, 0xCE, 0x2D,
    0xC2, 0x0B, 0xE0, 0xA8, 0x2A, 0xAA, 0x4F, 0x4C, 0x7A, 0xAD, 0xCA, 0xA1, 0xF5, 0x04, 0x30, 0x84,
    0xF1, 0xC3, 0x05, 0xB0, 0x8D, 0x32, 0xDE, 0x4E, 0xD3, 0x2C, 0xD0, 0x8F, 0x30, 0x5C, 0x02, 0x23,
    0xEC, 0x4C, 0x67, 0x00, 0x08, 0x03, 0x94, 0xC7, 0x96, 0x67, 0xFD, 0xC6, 0xAC, 0x92, 0x0B, 0x67,
    0x89, 0xBE, 0x38, 0x05, 0x0E, 0x88, 0x66, 0x03, 0xD0, 0x46, 0xF4, 0xCA, 0x26, 0x0E, 0x86, 0x72,
    0xA2, 0x40, 0xA2, 0xD2, 0xFD, 0x30, 0x57, 0xEA, 0x2B, 0x6B, 0x94, 0xAA, 0x43, 0x77, 0x2C, 0x44,
    0x08, 0x18, 0x50, 0xFF, 0xFB, 0x30, 0x04, 0x03, 0x80, 0x00, 0xDB, 0x07, 0x49, 0x05, 0x74, 0x00,
    0x0A, 0x1F, 0x01, 0xA9, 0x60, 0xAB, 0x64, 0x01, 0x03, 0x38, 0x1D, 0x4C, 0xB9, 0x91, 0x00, 0x10,
    0x77, 0x06, 0x29, 0x57, 0x29, 0x80, 0x00, 0x1A, 0x18, 0xB4, 0x5F, 0x1A, 0x5D, 0xDB, 0x07, 0x63,
    0x4A, 0x62, 0xCA, 0x21, 0xC8, 0xB1, 0x50, 0x01, 0x69, 0xB1, 0x71, 0xFF, 0xAE, 0xA4, 0xCE, 0x0E,
    0x61, 0xD1, 0x6B, 0x00, 0x10, 0x06, 0x01, 0x20, 0x24, 0x02, 0x83, 0xD8, 0x1A, 0x6D, 0x0F, 0xC0,
    0xF6, 0x01, 0x8B, 0x02, 0x90, 0x4C, 0x09, 0x6C, 0xBE, 0x9F, 0xA0, 0x62, 0x41, 0x00, 0x01, 0x70,
    0x76, 0xCC, 0x05, 0x54, 0xBC, 0xDB, 0x0D, 0x19, 0xE3, 0xDA, 0x6C, 0x8C, 0x4B, 0xD7, 0xFD, 0xFC,
    0xD7, 0x0F, 0xFF, 0xF4, 0x0E, 0x07, 0x00, 0x00, 0x07, 0xF4, 0x27, 0x50, 0x90, 0x30, 0x06, 0x28,
    0x35, 0x01, 0xAC, 0x02, 0x28, 0x06, 0x34, 0x18, 0x28, 0x35, 0x86, 0xE7, 0x22, 0x05, 0x0A, 0xCB,
    0x51, 0x21, 0x26, 0x03, 0x45, 0x49, 0x78, 0x22, 0x46, 0x44, 0xE2, 0x59, 0x26, 0x4E, 0x3F, 0xFF,
    0xFB, 0x32, 0x04, 0x06, 0x80, 0x00, 0xF8, 0x0D, 0x55, 0x06, 0x40, 0x40, 0x00, 0x1E, 0x62, 0x4A,
    0x90, 0xC9, 0xA0, 0x00, 0x00, 0x00, 0x01, 0xA4, 0x1C, 0x00, 0x00, 0x20, 0x00, 0x00, 0x34, 0x83,
    0x80, 0x00, 0x04, 0x08, 0x1F, 0xD0, 0x51, 0xDC, 0xB8, 0x10, 0xFF, 0x2A, 0x40, 0xCF, 0xC2, 0x02,
    0xC1, 0xCF, 0xCD, 0x8D, 0x16, 0x37, 0x2E, 0x2F, 0x97, 0xC9, 0x9A, 0xF2, 0x0C, 0x00, 0x91, 0xF0,
    0x3C, 0x02, 0x2D, 0x78, 0xB8, 0xAB, 0x7F, 0xFD, 0x0E, 0x70, 0x87, 0x53, 0x43, 0x9F, 0x3E, 0x2A,
    0xDF, 0xCC, 0x3D, 0x6A, 0x4C, 0x41, 0x4D, 0x45, 0x33, 0x2E, 0x31, 0x30, 0x30, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
};
// clang-format on
}  // namespace

// Reads the same stream with dr_mp3 directly and checks readMp3 returns (left + right) / 2 for
// every frame. A length-only check would also pass if one channel were dropped.
TEST_F(AudioUtilsSampleRateTest, mp3StereoDownmixAveragesChannels) {
    std::string_view view(reinterpret_cast<const char*>(STEREO_MP3_DISTINCT_CHANNELS), sizeof(STEREO_MP3_DISTINCT_CHANNELS));

    drmp3 reference;
    ASSERT_TRUE(drmp3_init_memory(&reference, view.data(), view.size(), nullptr));
    ASSERT_EQ(reference.channels, 2u);
    std::vector<float> interleaved;
    float chunk[2 * 1152];
    for (;;) {
        const drmp3_uint64 framesRead = drmp3_read_pcm_frames_f32(&reference, 1152, chunk);
        if (framesRead == 0) {
            break;
        }
        interleaved.insert(interleaved.end(), chunk, chunk + framesRead * 2);
    }
    drmp3_uninit(&reference);
    const size_t frames = interleaved.size() / 2;
    ASSERT_GT(frames, 2u * 1152u);

    std::vector<float> decoded;
    ASSERT_NO_THROW({ decoded = readMp3(view, DISABLED_RESAMPLING_SAMPLE_RATE); });
    ASSERT_EQ(decoded.size(), frames);

    float maxChannelDifference = 0.0f;
    for (size_t i = 0; i < frames; i++) {
        const float left = interleaved[2 * i];
        const float right = interleaved[2 * i + 1];
        maxChannelDifference = std::max(maxChannelDifference, std::abs(left - right));
        EXPECT_NEAR(decoded[i], (left + right) * 0.5f, 1e-6f) << "frame " << i;
    }
    // The channels must really differ, otherwise averaging would be indistinguishable from
    // keeping a single channel.
    EXPECT_GT(maxChannelDifference, 0.1f);
}

// Validates that validateAudioFileSize correctly rejects when inputSamples * targetRate
// would overflow size_t. This guards against the case where dr_mp3 provides an inflated
// totalPCMFrameCount (e.g. from a malicious Xing tag or UINT64_MAX sentinel).
TEST_F(AudioUtilsSampleRateTest, validateAudioFileSizeRejectsOnOverflow) {
    // Use a value large enough to cause overflow when multiplied by targetRate (16000)
    size_t hugeInputSamples = std::numeric_limits<size_t>::max() / 16000 + 1;
    EXPECT_THROW(
        validateAudioFileSize(hugeInputSamples, /*inputRate=*/44100, /*targetRate=*/16000, /*channels=*/1, /*bytesPerSample=*/sizeof(float)),
        std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, validateAudioFileSizeAcceptsBelowOverflowThreshold) {
    // Just below the overflow threshold — should not throw due to overflow detection,
    // but may still throw if the estimated size exceeds the 1GB default limit.
    size_t belowOverflow = std::numeric_limits<size_t>::max() / 16000;
    // This produces an enormous output (still ~1.15e15 bytes), so it should be rejected
    // by the max-file-size check, not the overflow check.
    EXPECT_THROW(
        validateAudioFileSize(belowOverflow, /*inputRate=*/44100, /*targetRate=*/16000, /*channels=*/1, /*bytesPerSample=*/sizeof(float)),
        std::runtime_error);
}

// Helper: builds an in-memory PCM16 WAV file with configurable channels and bitsPerSample.
std::string buildWavBufferEx(uint32_t sampleRate, uint32_t numSamples, uint16_t channels, uint16_t bitsPerSample) {
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

TEST_F(AudioUtilsSampleRateTest, wavStereoDownmixProducesCorrectSampleCount) {
    const uint32_t numSamples = 32;
    const std::string wav = buildWavBufferEx(/*sampleRate=*/16000, numSamples, /*channels=*/2, /*bitsPerSample=*/16);
    std::string_view view(wav);
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readWav(view); });
    // Stereo is downmixed to mono: output has numSamples frames
    EXPECT_EQ(decoded.size(), numSamples);
}

TEST_F(AudioUtilsSampleRateTest, wavWithZeroChannelsThrows) {
    const std::string wav = buildWavBufferEx(/*sampleRate=*/16000, /*numSamples=*/16, /*channels=*/0, /*bitsPerSample=*/16);
    std::string_view view(wav);
    EXPECT_THROW({ auto decoded = readWav(view); }, std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, wavWithThreeChannelsThrows) {
    const std::string wav = buildWavBufferEx(/*sampleRate=*/16000, /*numSamples=*/16, /*channels=*/3, /*bitsPerSample=*/16);
    std::string_view view(wav);
    EXPECT_THROW({ auto decoded = readWav(view); }, std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, wavWithNonByteAlignedBitsPerSampleThrows) {
    const std::string wav = buildWavBufferEx(/*sampleRate=*/16000, /*numSamples=*/16, /*channels=*/1, /*bitsPerSample=*/12);
    std::string_view view(wav);
    EXPECT_THROW({ auto decoded = readWav(view); }, std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, wavWith3BitSamplesThrows) {
    const std::string wav = buildWavBufferEx(/*sampleRate=*/16000, /*numSamples=*/16, /*channels=*/1, /*bitsPerSample=*/3);
    std::string_view view(wav);
    EXPECT_THROW({ auto decoded = readWav(view); }, std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, wavTruncatedDataReturnsFewerSamples) {
    // Build a valid WAV with 100 samples then truncate the data section to 10 bytes (5 frames)
    std::string wav = buildWavBuffer(/*sampleRate=*/16000, /*numSamples=*/100);
    wav.resize(44 + 10);
    uint32_t riffSize = static_cast<uint32_t>(wav.size() - 8);
    std::memcpy(&wav[4], &riffSize, 4);
    std::string_view view(wav);
    std::vector<float> decoded;
    // dr_wav clamps frame count to available data; readWav should not crash or throw
    EXPECT_NO_THROW({ decoded = readWav(view); });
    EXPECT_LE(decoded.size(), 5u);
}

TEST_F(AudioUtilsSampleRateTest, wavWithVeryLowSampleRateRejected) {
    // 100 samples at 1 Hz resampled to 16000 Hz = 1,600,000 output samples = 6.4MB
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", "1000");
    const std::string wav = buildWavBuffer(/*sampleRate=*/1, /*numSamples=*/100);
    std::string_view view(wav);
    EXPECT_THROW({ auto decoded = readWav(view); }, std::runtime_error);
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, wavWithHighSampleRateDownsamples) {
    const uint32_t numSamples = 96;
    const std::string wav = buildWavBuffer(/*sampleRate=*/96000, numSamples);
    std::string_view view(wav);
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readWav(view); });
    // 96000 -> 16000 is 1/6 ratio
    EXPECT_EQ(decoded.size(), static_cast<size_t>(numSamples * 16000 / 96000));
}

TEST_F(AudioUtilsSampleRateTest, wavCorruptHeaderThrows) {
    std::string corrupt = "RIFF\x00\x00\x00\x00WAVE";
    std::string_view view(corrupt);
    EXPECT_THROW({ auto decoded = readWav(view); }, std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, wavEmptyInputThrows) {
    std::string empty;
    std::string_view view(empty);
    EXPECT_THROW({ auto decoded = readWav(view); }, std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, mp3CorruptHeaderThrows) {
    std::string corrupt = "not an mp3 file at all";
    std::string_view view(corrupt);
    EXPECT_THROW({ auto decoded = readMp3(view); }, std::runtime_error);
}

// 8b. MP3 empty input
TEST_F(AudioUtilsSampleRateTest, mp3EmptyInputThrows) {
    std::string empty;
    std::string_view view(empty);
    EXPECT_THROW({ auto decoded = readMp3(view); }, std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, wavZeroSamplesReturnsEmpty) {
    const std::string wav = buildWavBuffer(/*sampleRate=*/16000, /*numSamples=*/0);
    std::string_view view(wav);
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readWav(view); });
    EXPECT_EQ(decoded.size(), 0u);
}

TEST_F(AudioUtilsSampleRateTest, isWavBufferReturnsTrueForValidWav) {
    const std::string wav = buildWavBuffer(/*sampleRate=*/16000, /*numSamples=*/16);
    EXPECT_TRUE(isWavBuffer(wav));
}

TEST_F(AudioUtilsSampleRateTest, isWavBufferReturnsFalseForNonWav) {
    EXPECT_FALSE(isWavBuffer("not a wav file"));
}

TEST_F(AudioUtilsSampleRateTest, isWavBufferReturnsFalseForShortInput) {
    EXPECT_FALSE(isWavBuffer("RIFF"));
    EXPECT_FALSE(isWavBuffer(""));
}

TEST_F(AudioUtilsSampleRateTest, isWavBufferReturnsFalseForWrongChunkSize) {
    std::string wav = buildWavBuffer(/*sampleRate=*/16000, /*numSamples=*/16);
    // Corrupt the chunk_size field at offset 4
    uint32_t badSize = 9999;
    std::memcpy(&wav[4], &badSize, 4);
    EXPECT_FALSE(isWavBuffer(wav));
}

TEST_F(AudioUtilsSampleRateTest, validateAudioFileSizeRejectsWithMultipleChannels) {
    // 100 samples at 16000->16000 (no ratio change) = 100 output samples
    // With 2 channels and 4 bytes/sample: 100 * 2 * 4 = 800 bytes — under default 1GB
    EXPECT_NO_THROW(
        validateAudioFileSize(/*inputSamples=*/100, /*inputRate=*/16000, /*targetRate=*/16000, /*channels=*/2, /*bytesPerSample=*/sizeof(float)));
    // With a tight limit, 800 bytes should exceed 500
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", "500");
    EXPECT_THROW(
        validateAudioFileSize(/*inputSamples=*/100, /*inputRate=*/16000, /*targetRate=*/16000, /*channels=*/2, /*bytesPerSample=*/sizeof(float)),
        std::runtime_error);
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, envVarNonNumericUsesDefault) {
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", "not_a_number");
    const std::string wav = buildWavBuffer(/*sampleRate=*/16000, /*numSamples=*/16);
    std::string_view view(wav);
    std::vector<float> decoded;
    // Default is 1GB, 16 samples * 4 bytes = 64 bytes — well under limit
    EXPECT_NO_THROW({ decoded = readWav(view); });
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, envVarZeroUsesDefault) {
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", "0");
    const std::string wav = buildWavBuffer(/*sampleRate=*/16000, /*numSamples=*/16);
    std::string_view view(wav);
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readWav(view); });
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, envVarEmptyStringUsesDefault) {
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", "");
    const std::string wav = buildWavBuffer(/*sampleRate=*/16000, /*numSamples=*/16);
    std::string_view view(wav);
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readWav(view); });
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, validateAudioFileSizeRejectsOnCeilingDivisionOverflow) {
    size_t maxVal = std::numeric_limits<size_t>::max();
    uint32_t targetRate = 16000;
    uint32_t inputRate = 44100;
    // inputSamples = max / targetRate passes first check but
    // product = inputSamples * targetRate ≈ max - (max % 16000) which is > max - 44099
    size_t inputSamples = maxVal / targetRate;
    EXPECT_THROW(
        validateAudioFileSize(inputSamples, inputRate, targetRate, /*channels=*/1, /*bytesPerSample=*/sizeof(float)),
        std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, wavAtExactSizeLimitPasses) {
    const uint32_t numSamples = 32;
    // At 16kHz no resampling needed. Output = 32 floats = 128 bytes
    size_t expectedSize = numSamples * sizeof(float);
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", std::to_string(expectedSize));
    const std::string wav = buildWavBuffer(/*sampleRate=*/16000, numSamples);
    std::string_view view(wav);
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readWav(view); });
    EXPECT_EQ(decoded.size(), static_cast<size_t>(numSamples));
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, wavOneByteOverLimitThrows) {
    const uint32_t numSamples = 32;
    size_t expectedSize = numSamples * sizeof(float);
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", std::to_string(expectedSize - 1));
    const std::string wav = buildWavBuffer(/*sampleRate=*/16000, numSamples);
    std::string_view view(wav);
    EXPECT_THROW({ auto decoded = readWav(view); }, std::runtime_error);
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

// ---- prepareAudioOutput size-cap tests ----------------------------------------

TEST_F(AudioUtilsSampleRateTest, prepareAudioOutputRejectsOversizedSpeech) {
    // Simulate what a tiny speed value (e.g. 1e-9) would produce: a speechSize
    // that exceeds the 1 GB default cap.  The function must throw before any
    // allocation attempt.
    constexpr uint32_t sampleRate = 24000;
    // 512 Mi samples × 2 bytes is exactly the 1 GiB PCM16 cap, so add one
    // sample to keep the early oversized-buffer rejection path valid.
    constexpr size_t oversizedSpeech = (512ull * 1024 * 1024) + 1;
    // A non-null dummy pointer is enough; prepareAudioOutput throws before
    // it dereferences waveformPtr.
    const float dummyWaveform = 0.0f;
    void* ppData = nullptr;
    size_t pDataSize = 0;
    EXPECT_THROW(
        prepareAudioOutput(&ppData, pDataSize, sampleRate, oversizedSpeech, &dummyWaveform),
        std::runtime_error);
}

TEST_F(AudioUtilsSampleRateTest, prepareAudioOutputRejectsOversizedSpeechWithCustomEnvVar) {
    // Honour OVMS_AUDIO_MAX_FILE_SIZE_BYTES for the synthesis path too.
    constexpr uint32_t sampleRate = 24000;
    // 100 samples × 4 bytes = 400 bytes — normally fine, but tiny cap rejects it.
    constexpr size_t speechSize = 100;
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", "100");
    const float dummyWaveform = 0.0f;
    void* ppData = nullptr;
    size_t pDataSize = 0;
    EXPECT_THROW(
        prepareAudioOutput(&ppData, pDataSize, sampleRate, speechSize, &dummyWaveform),
        std::runtime_error);
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, prepareAudioOutputAcceptsSmallSpeech) {
    // A small, realistic synthesis output must pass the cap check.
    constexpr uint32_t sampleRate = 24000;
    // 1000 samples × 4 bytes = 4000 bytes — well under the 1 GB default.
    constexpr size_t speechSize = 1000;
    std::vector<float> waveform(speechSize, 0.0f);
    void* ppData = nullptr;
    size_t pDataSize = 0;
    EXPECT_NO_THROW(
        prepareAudioOutput(&ppData, pDataSize, sampleRate, speechSize, waveform.data()));
    if (ppData) {
        free(ppData);  // drwav allocates via DRWAV_MALLOC
    }
}

TEST_F(AudioUtilsSampleRateTest, prepareAudioOutputRejectsWhenHeaderPushesOverLimit) {
    // The cap is set to exactly the raw PCM byte count.  The WAV container adds
    // RIFF/fmt/fact/data header overhead on top of that, so the final pDataSize
    // returned by drwav must exceed the limit and be rejected — even though the
    // raw PCM payload alone would have been accepted.
    constexpr uint32_t sampleRate = 24000;
    constexpr size_t speechSize = 100;
    constexpr size_t rawPcmBytes = speechSize * sizeof(int16_t);
    // Cap == raw PCM size; the WAV container will be larger, so it must be rejected.
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", std::to_string(rawPcmBytes));
    std::vector<float> waveform(speechSize, 0.0f);
    void* ppData = nullptr;
    size_t pDataSize = 0;
    EXPECT_THROW(
        prepareAudioOutput(&ppData, pDataSize, sampleRate, speechSize, waveform.data()),
        std::runtime_error);
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
}

TEST_F(AudioUtilsSampleRateTest, prepareAudioOutputProducesPcm16Wav) {
    constexpr uint32_t sampleRate = 24000;
    std::array<float, 3> waveform{0.0f, 0.5f, -0.5f};
    void* ppData = nullptr;
    size_t pDataSize = 0;
    ASSERT_NO_THROW(prepareAudioOutput(&ppData, pDataSize, sampleRate, waveform.size(), waveform.data()));
    ASSERT_NE(ppData, nullptr);
    std::string wav(reinterpret_cast<const char*>(ppData), pDataSize);
    EXPECT_TRUE(isWavBuffer(wav));
    EXPECT_EQ(readLe16(wav.data() + 20), 1);   // PCM
    EXPECT_EQ(readLe16(wav.data() + 22), 1);   // mono
    EXPECT_EQ(readLe16(wav.data() + 34), 16);  // PCM16
    EXPECT_EQ(static_cast<unsigned char>(wav[44]), 0x00);
    EXPECT_EQ(static_cast<unsigned char>(wav[45]), 0x00);
    EXPECT_EQ(static_cast<unsigned char>(wav[46]), 0xFF);
    EXPECT_EQ(static_cast<unsigned char>(wav[47]), 0x3F);
    free(ppData);
}

TEST_F(AudioUtilsSampleRateTest, prepareRawPcm16LEOutputProducesLittleEndianSamples) {
    std::array<float, 2> waveform{0.0f, 0.5f};
    void* ppData = nullptr;
    size_t pDataSize = 0;
    ASSERT_NO_THROW(prepareRawPcm16LEOutput(&ppData, pDataSize, waveform.size(), waveform.data()));
    ASSERT_NE(ppData, nullptr);
    ASSERT_EQ(pDataSize, waveform.size() * sizeof(int16_t));
    const auto* bytes = reinterpret_cast<const unsigned char*>(ppData);
    EXPECT_EQ(bytes[0], 0x00);
    EXPECT_EQ(bytes[1], 0x00);
    EXPECT_EQ(bytes[2], 0xFF);
    EXPECT_EQ(bytes[3], 0x3F);
    free(ppData);
}

TEST_F(AudioUtilsSampleRateTest, prepareRawPcm16LEOutputMapsNaNToSilence) {
    std::array<float, 3> waveform{0.0f, std::numeric_limits<float>::quiet_NaN(), 0.5f};
    void* ppData = nullptr;
    size_t pDataSize = 0;
    ASSERT_NO_THROW(prepareRawPcm16LEOutput(&ppData, pDataSize, waveform.size(), waveform.data()));
    ASSERT_NE(ppData, nullptr);
    ASSERT_EQ(pDataSize, waveform.size() * sizeof(int16_t));
    const auto* pcm16 = reinterpret_cast<const int16_t*>(ppData);
    EXPECT_EQ(pcm16[0], 0);
    EXPECT_EQ(pcm16[1], 0);
    EXPECT_EQ(pcm16[2], 16383);
    free(ppData);
}

}  // namespace

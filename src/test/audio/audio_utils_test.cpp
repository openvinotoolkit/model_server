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
#include <limits>
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
    // For this frame, actual decoded size is 2304 samples (stereo or decoder output)
    size_t expectedDecodedSize = 2304 * sizeof(float);
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
    // For this frame, actual decoded size is 2304 samples (stereo or decoder output)
    size_t expectedDecodedSize = 2304 * sizeof(float);
    SetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES", std::to_string(expectedDecodedSize));
    std::vector<float> decoded;
    EXPECT_NO_THROW({ decoded = readMp3(view); });
    UnSetEnvironmentVar("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
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

}  // namespace

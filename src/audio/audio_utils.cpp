//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#define DR_WAV_IMPLEMENTATION
#define DR_MP3_IMPLEMENTATION
#include "src/port/dr_audio.hpp"
#include "audio_utils.hpp"
#include "src/timer.hpp"
#include "src/logging.hpp"
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#pragma warning(push)
#define PIPELINE_SUPPORTED_SAMPLE_RATE 16000

using namespace ovms;

static void validateAudioFileSizeAgainstMaxValue(size_t fileSize);

bool isWavBuffer(const std::string buf) {
    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    SPDLOG_TRACE("isWavBuffer: buf {}", buf.substr(0, 12));
    if (buf.size() < 12 || buf.substr(0, 4) != "RIFF" || buf.substr(8, 4) != "WAVE") {
        return false;
    }

    uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(buf.data() + 4);
    SPDLOG_TRACE("isWavBuffer: chunk_size {}", chunk_size);
    if (chunk_size + 8 != buf.size()) {
        return false;
    }

    return true;
}
// https://github.com/openvinotoolkit/openvino.genai/blob/8698683535fe32b5e3cb6953000c4e0175841bd3/samples/c/whisper_speech_recognition/whisper_utils.c#L105
void resample_audio(const float* input,
    size_t inputLength,
    float inputRate,
    float targetRate,
    std::vector<float>& output) {
    SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "Input file sample rate: {}. Resampling to {} required", inputRate, targetRate);
    float ratio = inputRate / targetRate;

    for (size_t i = 0; i < output.size(); i++) {
        float src_idx = i * ratio;
        size_t idx0 = (size_t)src_idx;
        size_t idx1 = idx0 + 1;

        if (idx1 >= inputLength) {
            output.data()[i] = input[inputLength - 1];
        } else {
            float frac = src_idx - idx0;
            output.data()[i] = input[idx0] * (1.0f - frac) + input[idx1] * frac;
        }
    }
}

enum : unsigned int {
    TENSOR_PREPARATION,
    RESAMPLING,
    TIMER_END
};

std::vector<float> readWav(const std::string_view& wavData) {
    Timer<TIMER_END> timer;
    timer.start(TENSOR_PREPARATION);
    drwav wav;
    auto result = drwav_init_memory(&wav, wavData.data(), wavData.size(), nullptr);
    if (result == false) {
        throw std::runtime_error("WAV file parsing failed");
    }
    if (wav.channels != 1 && wav.channels != 2) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file must be mono or stereo");
    }
    // dr_wav accepts bitsPerSample in [1, 64]; reject anything that is not a
    // whole-byte PCM width since the rest of this function and downstream
    // pipeline assume drwav_read_pcm_frames_s16 produces well-formed samples.
    if (wav.bitsPerSample < 8 || (wav.bitsPerSample % 8) != 0) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file has unsupported bits per sample");
    }

    const uint64_t blockAlign = wav.channels * (wav.bitsPerSample / 8);
    const uint64_t maxPossibleFrames = blockAlign ? (wavData.size() / blockAlign) : 0;
    if (wav.totalPCMFrameCount > maxPossibleFrames) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file header claims more frames than possible from data chunk size");
    }
    // Validate decoded buffer size before resampling (float32 mono output)
    validateAudioFileSize(wav.totalPCMFrameCount, wav.sampleRate, PIPELINE_SUPPORTED_SAMPLE_RATE, wav.channels, sizeof(float));
    
    const uint64_t n = wav.totalPCMFrameCount;
    std::vector<int16_t> pcm16;
    pcm16.resize(n * wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);

    // convert to mono, float
    std::vector<float> pcmf32;
    pcmf32.resize(n);
    if (wav.channels == 1) {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[i]) / 32768.0f;
        }
    } else {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
        }
    }
    timer.stop(TENSOR_PREPARATION);
    auto tensorPreparationTime = (timer.elapsed<std::chrono::microseconds>(TENSOR_PREPARATION)) / 1000;
    SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "Tensor preparation time: {} ms size: {}", tensorPreparationTime, pcmf32.size());
    if (wav.sampleRate == PIPELINE_SUPPORTED_SAMPLE_RATE) {
        return pcmf32;
    }

    timer.start(RESAMPLING);
    size_t outputLength = (size_t)(pcmf32.size() * PIPELINE_SUPPORTED_SAMPLE_RATE / wav.sampleRate);
    std::vector<float> output(outputLength);
    resample_audio(reinterpret_cast<float*>(pcmf32.data()), pcmf32.size(), wav.sampleRate, PIPELINE_SUPPORTED_SAMPLE_RATE, output);
    timer.stop(RESAMPLING);
    auto resamplingTime = (timer.elapsed<std::chrono::microseconds>(RESAMPLING)) / 1000;
    SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "Resampling time: {} ms", resamplingTime);
    SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "XXXXXXXXXXXXXX FINAL OUTPUT SIZE: {}", output.size());
    return output;
}
#pragma warning(push)
#pragma warning(disable : 6262)
std::vector<float> readMp3(const std::string_view& mp3Data) {
    Timer<TIMER_END> timer;
    timer.start(TENSOR_PREPARATION);
    drmp3 mp3;
    auto result = drmp3_init_memory(&mp3, mp3Data.data(), mp3Data.size(), nullptr);
    if (result == 0) {
        throw std::runtime_error("MP3 file parsing failed");
    }

    if (mp3.channels != 1 && mp3.channels != 2) {
        drmp3_uninit(&mp3);
        throw std::runtime_error("MP3 file must be mono or stereo");
    }
    // Validate expected output buffer size before any allocation or decoding, using metadata.
    // dr_mp3 sets totalPCMFrameCount to UINT64_MAX when no Xing/VBRI tag is present (unknown length).
    // In that case skip metadata-based validation; post-decode validation will guard against OOM.
    if (mp3.totalPCMFrameCount != std::numeric_limits<uint64_t>::max()) {
        try {
            validateAudioFileSize(mp3.totalPCMFrameCount, mp3.sampleRate, PIPELINE_SUPPORTED_SAMPLE_RATE, mp3.channels, sizeof(float));
        } catch (...) {
            drmp3_uninit(&mp3);
            throw;
        }
    }
    constexpr size_t MP3_DECODE_CHUNK_FRAMES = 1152;  // 1152 is the maximum number of PCM samples per channel produced by a single MPEG-1 Layer III MP3 frame. Reference: ISO/IEC 11172-3
    // We cannot know the decoded sample count up front, but we can check the maximum possible size based on file size and sample rate
    // For safety, check the decoded buffer after filling
    float tempBuffer[MP3_DECODE_CHUNK_FRAMES * 2];  // 2 is max channels we validated earlier
    std::vector<float> pcmf32;
    //pcmf32.reserve(mp3.totalPCMFrameCount * mp3.channels);
    try {
        for (;;) {
            drmp3_uint64 framesRead = drmp3_read_pcm_frames_f32(&mp3, MP3_DECODE_CHUNK_FRAMES, tempBuffer);
            if (framesRead == 0) {
                break;
            }
            pcmf32.insert(pcmf32.end(), tempBuffer, tempBuffer + framesRead * mp3.channels);
            validateAudioFileSizeAgainstMaxValue(pcmf32.size() * sizeof(float));
        }
    } catch (...) {
        drmp3_uninit(&mp3);
        throw;
    }
    drmp3_uninit(&mp3);
    timer.stop(TENSOR_PREPARATION);
    auto tensorPreparationTime = (timer.elapsed<std::chrono::microseconds>(TENSOR_PREPARATION)) / 1000;
    SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "Tensor preparation time: {} ms size: {}", tensorPreparationTime, pcmf32.size());
    //validateAudioFileSizeAgainstMaxValue(pcmf32.size() * sizeof(float));
    if (mp3.sampleRate == PIPELINE_SUPPORTED_SAMPLE_RATE) {
        return pcmf32;
    }
    timer.start(RESAMPLING);
    //validateAudioFileSize(mp3.totalPCMFrameCount, mp3.sampleRate, PIPELINE_SUPPORTED_SAMPLE_RATE, mp3.channels, sizeof(float));

    size_t outputLength = (size_t)(pcmf32.size() * PIPELINE_SUPPORTED_SAMPLE_RATE / mp3.sampleRate);
    validateAudioFileSizeAgainstMaxValue(outputLength * sizeof(float));
    std::vector<float> output(outputLength);
    resample_audio(reinterpret_cast<float*>(pcmf32.data()), pcmf32.size(), mp3.sampleRate, PIPELINE_SUPPORTED_SAMPLE_RATE, output);
    timer.stop(RESAMPLING);
    auto resamplingTime = (timer.elapsed<std::chrono::microseconds>(RESAMPLING)) / 1000;
    SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "Resampling time: {} ms", resamplingTime);
    return output;
}

void prepareAudioOutput(void** ppData, size_t& pDataSize, uint16_t bitsPerSample, size_t speechSize, const float* waveformPtr) {
    enum : unsigned int {
        OUTPUT_PREPARATION,
        TIMER_END
    };
    Timer<TIMER_END> timer;
    timer.start(OUTPUT_PREPARATION);
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = 16000;  // assume it is always 16 KHz
    format.bitsPerSample = bitsPerSample;
    drwav wav;
    size_t totalSamples = speechSize * format.channels;

    auto status = drwav_init_memory_write(&wav, ppData, &pDataSize, &format, nullptr);
    if (status == DRWAV_FALSE) {
        throw std::runtime_error("Failed to write all frames");
    }
    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, totalSamples, waveformPtr);
    if (framesWritten != totalSamples) {
        throw std::runtime_error("Failed to write all frames");
    }
    drwav_uninit(&wav);
    timer.stop(OUTPUT_PREPARATION);
    auto outputPreparationTime = (timer.elapsed<std::chrono::microseconds>(OUTPUT_PREPARATION)) / 1000;
    SPDLOG_LOGGER_DEBUG(t2s_calculator_logger, "Output preparation time: {} ms", outputPreparationTime);
}
static void validateAudioFileSizeAgainstMaxValue(size_t fileSize) {
    constexpr size_t DEFAULT_MAX_FILE_SIZE = 1024ull * 1024 * 1024; // 1GB
    size_t maxFileSize = DEFAULT_MAX_FILE_SIZE;
    const char* env = std::getenv("OVMS_AUDIO_MAX_FILE_SIZE_BYTES");
    if (env && *env) {
        try {
            size_t parsed = std::stoull(env);
            if (parsed > 0) {
                maxFileSize = parsed;
            }
        } catch (...) {
            // Ignore invalid env, use default
        }
    }
    SPDLOG_DEBUG("{} : {}", maxFileSize, fileSize);
    if (fileSize > maxFileSize) {
        throw std::runtime_error("Audio file size " + std::to_string(fileSize) +
            " exceeds maximum allowed size (" + std::to_string(maxFileSize) + ")");
    }
}

// Throws if the estimated resampled audio buffer size would exceed the maximum allowed size
void validateAudioFileSize(
    size_t inputSamples,
    uint32_t inputRate,
    uint32_t targetRate,
    uint32_t channels,
    size_t bytesPerSample) {
    // Detect overflow: if inputSamples is large enough that inputSamples * targetRate
    // would overflow size_t, the output is certainly too large.
    if (inputSamples > std::numeric_limits<size_t>::max() / targetRate) {
        throw std::runtime_error("Audio file estimated output size overflows maximum representable value");
    }
    size_t product = inputSamples * targetRate;
    // Guard the ceiling-division addition against overflow as well.
    if (product > std::numeric_limits<size_t>::max() - (inputRate - 1)) {
        throw std::runtime_error("Audio file estimated output size overflows maximum representable value");
    }
    // Estimate output samples after resampling (ceil division)
    size_t outputSamples = (product + inputRate - 1) / inputRate;
    size_t expectedSize = outputSamples * channels * bytesPerSample;
    validateAudioFileSizeAgainstMaxValue(expectedSize);
}

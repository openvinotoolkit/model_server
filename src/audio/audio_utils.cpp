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
// Default bounds for the sample rate read from an attacker-supplied audio header.
// Anything outside this range either makes no audio sense or would cause the resampling
// output buffer to balloon into a denial-of-service-grade allocation. Operators can
// override the bounds via the OVMS_AUDIO_MIN_SAMPLE_RATE / OVMS_AUDIO_MAX_SAMPLE_RATE
// environment variables.
#define DEFAULT_MIN_SAMPLE_RATE 4000u
#define DEFAULT_MAX_SAMPLE_RATE 384000u

using namespace ovms;

namespace {
uint32_t parseSampleRateEnv(const char* envName, uint32_t defaultValue) {
    const char* raw = std::getenv(envName);
    if (raw == nullptr || *raw == '\0') {
        return defaultValue;
    }
    try {
        uint64_t parsed = std::stoull(raw);
        if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
            SPDLOG_LOGGER_WARN(s2t_calculator_logger, "Ignoring out-of-range value '{}' for {}; using default {}", raw, envName, defaultValue);
            return defaultValue;
        }
        return static_cast<uint32_t>(parsed);
    } catch (const std::exception&) {
        SPDLOG_LOGGER_WARN(s2t_calculator_logger, "Ignoring invalid value '{}' for {}; using default {}", raw, envName, defaultValue);
        return defaultValue;
    }
}

uint32_t getMinSampleRate() {
    static const uint32_t value = parseSampleRateEnv("OVMS_AUDIO_MIN_SAMPLE_RATE", DEFAULT_MIN_SAMPLE_RATE);
    return value;
}

uint32_t getMaxSampleRate() {
    static const uint32_t value = parseSampleRateEnv("OVMS_AUDIO_MAX_SAMPLE_RATE", DEFAULT_MAX_SAMPLE_RATE);
    return value;
}

void validateSampleRate(uint32_t sampleRate, const char* sourceFormat) {
    const uint32_t minRate = getMinSampleRate();
    const uint32_t maxRate = getMaxSampleRate();
    if (sampleRate < minRate || sampleRate > maxRate) {
        throw std::runtime_error(std::string(sourceFormat) + " file sample rate " + std::to_string(sampleRate) +
                                 " is out of supported range [" + std::to_string(minRate) + ", " + std::to_string(maxRate) + "]");
    }
}
}  // namespace

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

    // dr_wav already parsed the `data` chunk header, so totalPCMFrameCount is the
    // authoritative sample count. The previous formula divided the whole-file size
    // by bytesPerSample, which over-counted by (header_bytes / bytesPerSample) and
    // produced a trailing tail of silence in the decoded output.
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
    validateSampleRate(wav.sampleRate, "WAV");

    timer.start(RESAMPLING);
    size_t outputLength = (size_t)(pcmf32.size() * PIPELINE_SUPPORTED_SAMPLE_RATE / wav.sampleRate);
    std::vector<float> output(outputLength);
    resample_audio(reinterpret_cast<float*>(pcmf32.data()), pcmf32.size(), wav.sampleRate, PIPELINE_SUPPORTED_SAMPLE_RATE, output);
    timer.stop(RESAMPLING);
    auto resamplingTime = (timer.elapsed<std::chrono::microseconds>(RESAMPLING)) / 1000;
    SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "Resampling time: {} ms", resamplingTime);
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
    // Decode in a streaming loop instead of trusting mp3.totalPCMFrameCount, which
    // dr_mp3 lifts verbatim from the Xing/Info "FRAMES" tag without any sanity
    // check. A malicious file can claim ~4.3e9 PCM frames in a few hundred bytes,
    // which would otherwise translate into a multi-TB pre-allocation. Output
    // capacity here grows in proportion to the bytes the decoder actually
    // produces, so it is naturally bounded by the input length.
    constexpr size_t MP3_DECODE_CHUNK_FRAMES = 1152;  // maximum PCM frames produced by one MPEG audio frame
    std::vector<float> pcmf32;
    float tempBuffer[MP3_DECODE_CHUNK_FRAMES * 2];  // *2 to accommodate stereo
    for (;;) {
        drmp3_uint64 framesRead = drmp3_read_pcm_frames_f32(&mp3, MP3_DECODE_CHUNK_FRAMES, tempBuffer);
        if (framesRead == 0) {
            break;
        }
        pcmf32.insert(pcmf32.end(), tempBuffer, tempBuffer + framesRead * mp3.channels);
    }
    drmp3_uninit(&mp3);
    timer.stop(TENSOR_PREPARATION);
    auto tensorPreparationTime = (timer.elapsed<std::chrono::microseconds>(TENSOR_PREPARATION)) / 1000;
    SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "Tensor preparation time: {} ms size: {}", tensorPreparationTime, pcmf32.size());
    if (mp3.sampleRate == PIPELINE_SUPPORTED_SAMPLE_RATE) {
        return pcmf32;
    }
    validateSampleRate(mp3.sampleRate, "MP3");
    timer.start(RESAMPLING);
    size_t outputLength = (size_t)(pcmf32.size() * PIPELINE_SUPPORTED_SAMPLE_RATE / mp3.sampleRate);
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

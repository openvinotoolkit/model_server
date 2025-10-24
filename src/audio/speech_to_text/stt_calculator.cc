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
#include <fstream>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "src/timer.hpp"
#include "src/http_payload.hpp"
#include "src/logging.hpp"
#include <mutex>
#include <thread>

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)
#define DR_WAV_IMPLEMENTATION
#define DR_MP3_IMPLEMENTATION
#include "src/port/dr_audio.hpp"

#include "stt_servable.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace ovms;

namespace mediapipe {

const std::string STT_SESSION_SIDE_PACKET_TAG = "STT_NODE_RESOURCES";

#define PIPELINE_SUPPORTED_SAMPLE_RATE 16000

bool is_wav_buffer(const std::string buf) {
    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    SPDLOG_TRACE("is_wav_buffer: buf {}", buf.substr(0, 12));
    if (buf.size() < 12 || buf.substr(0, 4) != "RIFF" || buf.substr(8, 4) != "WAVE") {
        return false;
    }

    uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(buf.data() + 4);
    SPDLOG_TRACE("is_wav_buffer: chunk_size {}", chunk_size);
    if (chunk_size + 8 != buf.size()) {
        return false;
    }

    return true;
}
// https://github.com/openvinotoolkit/openvino.genai/blob/8698683535fe32b5e3cb6953000c4e0175841bd3/samples/c/whisper_speech_recognition/whisper_utils.c#L105
float* resample_audio(const float* input,
    size_t input_length,
    float input_rate,
    float target_rate,
    size_t* output_length) {
    SPDLOG_LOGGER_DEBUG(stt_calculator_logger, "Input file sample rate: {}. Resampling to {} required", input_rate, target_rate);
    float ratio = input_rate / target_rate;
    *output_length = (size_t)(input_length / ratio);
    float* output = (float*)malloc(*output_length * sizeof(float));

    if (!output) {
        return NULL;
    }

    for (size_t i = 0; i < *output_length; i++) {
        float src_idx = i * ratio;
        size_t idx0 = (size_t)src_idx;
        size_t idx1 = idx0 + 1;

        if (idx1 >= input_length) {
            output[i] = input[input_length - 1];
        } else {
            float frac = src_idx - idx0;
            output[i] = input[idx0] * (1.0f - frac) + input[idx1] * frac;
        }
    }

    return output;
}

enum : unsigned int {
    TENSOR_PREPARATION,
    RESAMPLING,
    TIMER_END
};

ov::genai::RawSpeechInput read_wav(const std::string_view& wav_data) {
    Timer<TIMER_END> timer;
    timer.start(TENSOR_PREPARATION);
    drwav wav;
    auto result = drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr);
    if (result == false) {
        throw std::runtime_error("WAV file parsing failed");
    }
    if (wav.channels != 1 && wav.channels != 2) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file must be mono or stereo");
    }

    const uint64_t n =
        wav_data.empty() ? wav.totalPCMFrameCount : wav_data.size() / (wav.channels * wav.bitsPerSample / 8ul);

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
    SPDLOG_LOGGER_DEBUG(stt_calculator_logger, "Tensor preparation time: {} ms size: {}", tensorPreparationTime, pcmf32.size());
    if (wav.sampleRate == PIPELINE_SUPPORTED_SAMPLE_RATE) {
        return pcmf32;
    }

    size_t output_length;
    timer.start(RESAMPLING);
    auto buffer = resample_audio(reinterpret_cast<float*>(pcmf32.data()), pcmf32.size(), wav.sampleRate, PIPELINE_SUPPORTED_SAMPLE_RATE, &output_length);
    timer.stop(RESAMPLING);
    auto resamplingTime = (timer.elapsed<std::chrono::microseconds>(RESAMPLING)) / 1000;
    SPDLOG_LOGGER_DEBUG(stt_calculator_logger, "Resampling time: {} ms", resamplingTime);
    std::vector<float> output(buffer, buffer + output_length);
    return output;
}

ov::genai::RawSpeechInput read_mp3(const std::string_view& mp3_data) {
    Timer<TIMER_END> timer;
    timer.start(TENSOR_PREPARATION);
    drmp3 mp3;
    auto result = drmp3_init_memory(&mp3, mp3_data.data(), mp3_data.size(), nullptr);
    if (result == 0) {
        throw std::runtime_error("MP3 file parsing failed");
    }

    if (mp3.channels != 1 && mp3.channels != 2) {
        drmp3_uninit(&mp3);
        throw std::runtime_error("MP3 file must be mono or stereo");
    }
    const uint64_t n = mp3.totalPCMFrameCount;
    std::vector<float> pcmf32;
    pcmf32.resize(n * mp3.channels);
    drmp3_read_pcm_frames_f32(&mp3, n, pcmf32.data());
    drmp3_uninit(&mp3);
    timer.stop(TENSOR_PREPARATION);
    auto tensorPreparationTime = (timer.elapsed<std::chrono::microseconds>(TENSOR_PREPARATION)) / 1000;
    SPDLOG_LOGGER_DEBUG(stt_calculator_logger, "Tensor preparation time: {} ms size: {}", tensorPreparationTime, pcmf32.size());
    if (mp3.sampleRate == PIPELINE_SUPPORTED_SAMPLE_RATE) {
        return pcmf32;
    }
    timer.start(RESAMPLING);
    size_t output_length;
    auto buffer = resample_audio(reinterpret_cast<float*>(pcmf32.data()), pcmf32.size(), mp3.sampleRate, PIPELINE_SUPPORTED_SAMPLE_RATE, &output_length);
    timer.stop(RESAMPLING);
    auto resamplingTime = (timer.elapsed<std::chrono::microseconds>(RESAMPLING)) / 1000;
    SPDLOG_LOGGER_DEBUG(stt_calculator_logger, "Resampling time: {} ms", resamplingTime);
    std::vector<float> output(buffer, buffer + output_length);
    return output;
}

class SttCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        cc->InputSidePackets().Tag(STT_SESSION_SIDE_PACKET_TAG).Set<SttServableMap>();  // TODO: template?
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<std::string>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(stt_calculator_logger, "SpeechToTextCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(stt_calculator_logger, "SpeechToTextCalculator  [Node: {}] Open start", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(stt_calculator_logger, "SpeechToTextCalculator  [Node: {}] Process start", cc->NodeName());

        SttServableMap pipelinesMap = cc->InputSidePackets().Tag(STT_SESSION_SIDE_PACKET_TAG).Get<SttServableMap>();
        auto it = pipelinesMap.find(cc->NodeName());
        RET_CHECK(it != pipelinesMap.end()) << "Could not find initialized STT node named: " << cc->NodeName();
        auto pipe = it->second;

        auto payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();

        std::unique_ptr<std::string> output;
        if (absl::StartsWith(payload.uri, "/v3/audio/transcriptions")) {
            if (payload.multipartParser->hasParseError())
                return absl::InvalidArgumentError("Failed to parse multipart data");

            std::string_view stream = payload.multipartParser->getFileContentByFieldName("stream");
            if (!stream.empty()) {
                return absl::InvalidArgumentError("streaming is not supported");
            }
            std::string_view file = payload.multipartParser->getFileContentByFieldName("file");
            if (file.empty()) {
                return absl::InvalidArgumentError(absl::StrCat("File parsing fails"));
            }

            ov::genai::RawSpeechInput raw_speech;
            try {
                if (is_wav_buffer(std::string(file))) {
                    SPDLOG_DEBUG("Received file format: wav");
                    raw_speech = read_wav(file);
                } else {
                    raw_speech = read_mp3(file);
                    SPDLOG_DEBUG("Received file format: mp3");
                }
            } catch (std::exception&) {
                return absl::InvalidArgumentError("Received input file is not valid wav nor mp3 audio file");
            }
            std::string result = "{\"text\": \"";
            std::unique_lock lock(pipe->sttPipelineMutex);
            result += pipe->sttPipeline->generate(raw_speech);
            result.append("\"}");
            output = std::make_unique<std::string>(result);
        } else {
            return absl::InvalidArgumentError(absl::StrCat("Unsupported URI: ", payload.uri));
        }

        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), cc->InputTimestamp());
        SPDLOG_LOGGER_DEBUG(stt_calculator_logger, "SpeechToTextCalculator  [Node: {}] Process end", cc->NodeName());

        return absl::OkStatus();
    }
};

const std::string SttCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string SttCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(SttCalculator);

}  // namespace mediapipe

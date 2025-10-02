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

#include "../http_payload.hpp"
#include "../logging.hpp"
#include <mutex>
#include <thread>

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#define DR_MP3_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable : 6386 6262)
#include "dr_mp3.h"
#pragma warning(pop)
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "speech_servable.hpp"

#ifdef _WIN32
#    include <fcntl.h>
#    include <io.h>
#endif

using namespace ovms;

namespace mediapipe {

// using SpeechPipelinesMap = std::unordered_map<std::string, std::shared_ptr<SpeechPipelines>>;


const std::string SPEECH_SESSION_SIDE_PACKET_TAG = "SPEECH_NODE_RESOURCES";

#define COMMON_SAMPLE_RATE 16000

bool is_wav_buffer(const std::string buf) {
    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    if (buf.size() < 12 || buf.substr(0, 4) != "RIFF" || buf.substr(8, 4) != "WAVE") {
        return false;
    }

    uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(buf.data() + 4);
    if (chunk_size + 8 != buf.size()) {
        return false;
    }

    return true;
}

ov::genai::RawSpeechInput read_wav(const std::string_view& wav_data) {
    drwav wav;

//     if (filename == "-") {
//         {
// #ifdef _WIN32
//             _setmode(_fileno(stdin), _O_BINARY);
// #endif

//             uint8_t buf[1024];
//             while (true) {
//                 const size_t n = fread(buf, 1, sizeof(buf), stdin);
//                 if (n == 0) {
//                     break;
//                 }
//                 wav_data.insert(wav_data.end(), buf, buf + n);
//             }
//         }

//         OPENVINO_ASSERT(drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr),
//                         "Failed to open WAV file from stdin");

//         fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, wav_data.size());
//     } else if (is_wav_buffer(filename)) {
//         OPENVINO_ASSERT(drwav_init_memory(&wav, filename.c_str(), filename.size(), nullptr),
//                         "Failed to open WAV file from fname buffer");
//     } else if (!drwav_init_file(&wav, filename.c_str(), nullptr)) {
// #if defined(WHISPER_FFMPEG)
//         OPENVINO_ASSERT(ffmpeg_decode_audio(fname, wav_data) == 0, "Failed to ffmpeg decode")

//         OPENVINO_ASSERT(drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr),
//                         "Failed to read wav data as wav")
// #else
//         throw std::runtime_error("failed to open as WAV file");
// #endif
//     }
    auto result =  drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr);
    if(result == false){
        SPDLOG_ERROR("FILE PARSING FAILED {}", result);
        throw std::runtime_error("FILE PARSING FAILED");
    }
    if (wav.channels != 1 && wav.channels != 2) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file must be mono or stereo");
    }

    if (wav.sampleRate != COMMON_SAMPLE_RATE) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file must be " + std::string{COMMON_SAMPLE_RATE / 1000} + " kHz");
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

    return pcmf32;
}

float* resample_audio(const float* input,
                      size_t input_length,
                      float input_rate,
                      float target_rate,
                      size_t* output_length) {
    if (input_rate == target_rate) {
        *output_length = input_length;
        float* output = (float*)malloc(input_length * sizeof(float));
        if (output) {
            memcpy(output, input, input_length * sizeof(float));
        }
        return output;
    }

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

ov::genai::RawSpeechInput read_mp3(const std::string_view& mp3_data) {
    drmp3 mp3;

    SPDLOG_ERROR("1");
    auto result =  drmp3_init_memory(&mp3, mp3_data.data(), mp3_data.size(), nullptr);
    if(result == false){
        SPDLOG_ERROR("FILE PARSING FAILED {}", result);
        throw std::runtime_error("FILE PARSING FAILED");
    }
    SPDLOG_ERROR("2");
    if (mp3.channels != 1 && mp3.channels != 2) {
        drmp3_uninit(&mp3);
        throw std::runtime_error("WAV file must be mono or stereo");
    }
    SPDLOG_ERROR("3 {}", mp3.sampleRate);
    // if (mp3.sampleRate != COMMON_SAMPLE_RATE) {
    //     drmp3_uninit(&mp3);
    //     throw std::runtime_error("WAV file must be " + std::string{COMMON_SAMPLE_RATE / 1000} + " kHz");
    // }
    SPDLOG_ERROR("4");
    const uint64_t n = mp3.totalPCMFrameCount;
    SPDLOG_ERROR("mp3.totalPCMFrameCount {} : n {}",  mp3.totalPCMFrameCount, n);
    std::vector<float> pcmf32;
    pcmf32.resize(n * mp3.channels);
    drmp3_read_pcm_frames_f32(&mp3, n, pcmf32.data());
    drmp3_uninit(&mp3);
    SPDLOG_ERROR("5");
    // convert to mono, float
    // std::vector<float> pcmf32;
    // pcmf32.resize(n);
    // if (mp3.channels == 1) {
    //     for (uint64_t i = 0; i < n; i++) {
    //         pcmf32[i] = float(pcm16[i]) / 32768.0f;
    //     }
    // } else {
    //     for (uint64_t i = 0; i < n; i++) {
    //         pcmf32[i] = float(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
    //     }
    // }
    size_t output_length;
    auto buffer = resample_audio(reinterpret_cast<float*>(pcmf32.data()), pcmf32.size(), mp3.sampleRate, 16000, &output_length);
    std::vector<float> output(buffer, buffer + output_length);
    return output;
}

std::variant<absl::Status, std::optional<std::string_view>> getFileFromPayload(const ovms::MultiPartParser& parser, const std::string& keyName) {
    std::string_view value = parser.getFileContentByFieldName(keyName);
    if (value.empty()) {
        return std::nullopt;
    }
    return value;
}

#define SET_OR_RETURN(TYPE, NAME, RHS) \
    auto NAME##_OPT = RHS;             \
    RETURN_IF_HOLDS_STATUS(NAME##_OPT) \
    auto NAME = std::get<TYPE>(NAME##_OPT);

#define RETURN_IF_HOLDS_STATUS(NAME)                  \
    if (std::holds_alternative<absl::Status>(NAME)) { \
        return std::get<absl::Status>(NAME);          \
    }

class SpeechCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        cc->InputSidePackets().Tag(SPEECH_SESSION_SIDE_PACKET_TAG).Set<SpeechServableMap>();  // TODO: template?
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<std::string>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "SpeechCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "SpeechCalculator  [Node: {}] Open start", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "SpeechCalculator  [Node: {}] Process start", cc->NodeName());

        SpeechServableMap pipelinesMap = cc->InputSidePackets().Tag(SPEECH_SESSION_SIDE_PACKET_TAG).Get<SpeechServableMap>();
        auto it = pipelinesMap.find(cc->NodeName());
        RET_CHECK(it != pipelinesMap.end()) << "Could not find initialized Speech node named: " << cc->NodeName();
        auto pipe = it->second;

        auto payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();

        std::unique_ptr<ov::Tensor> images;  // output
        std::unique_ptr<std::string> output;
        if (absl::StartsWith(payload.uri, "/v3/audio/transcriptions")) {
            if (payload.multipartParser->hasParseError())
                return absl::InvalidArgumentError("Failed to parse multipart data");

            SET_OR_RETURN(std::optional<std::string_view>, file, getFileFromPayload(*payload.multipartParser, "file"));
            auto stream = getFileFromPayload(*payload.multipartParser, "stream");
            if(!std::holds_alternative<absl::Status>(stream)){
                SPDLOG_ERROR("NO VALUE");
            }
            else{
                SPDLOG_ERROR("{}", (std::get<std::optional<std::string_view>>(stream)).value());
            }
            if(!file.has_value()){
                return absl::InvalidArgumentError(absl::StrCat("File parsing fails"));
            }
            // ov::genai::WhisperGenerationConfig config = pipe->whisperPipeline->get_generation_config();
            // // 'task' and 'language' parameters are supported for multilingual models only
            // config.language = "<|en|>";  // can switch to <|zh|> for Chinese language
            // config.task = "transcribe";
            // config.return_timestamps = true;
            ov::genai::RawSpeechInput raw_speech;
            try {
                raw_speech = read_mp3(file.value());
            } catch(std::exception&){
                return absl::InvalidArgumentError("Audio file pasing failed");
            }
            std::string result = "{\"text\": \"";
            std::unique_lock lock(pipe->whisperPipelineMutex);
            result += pipe->whisperPipeline->generate(raw_speech);
            result.append("\"}");
            SPDLOG_ERROR("{}",result);
            output = std::make_unique<std::string>(result);
        } else if(absl::StartsWith(payload.uri, "/v3/audio/speech")){
            if (payload.parsedJson->HasParseError())
                return absl::InvalidArgumentError("Failed to parse JSON");

            if (!payload.parsedJson->IsObject()) {
                return absl::InvalidArgumentError("JSON body must be an object");
            }
            auto inputIt = payload.parsedJson->FindMember("input");
            if (inputIt == payload.parsedJson->MemberEnd()) {
                return absl::InvalidArgumentError("input field is missing in JSON body");
            }
            if (!inputIt->value.IsString()) {
                return absl::InvalidArgumentError("input field is not a string");
            }
            auto streamIt = payload.parsedJson->FindMember("stream_format");
            if (streamIt != payload.parsedJson->MemberEnd()) {
                SPDLOG_ERROR("STREAM: {}", streamIt->value.GetString());
            }
            else{
                SPDLOG_ERROR("NO STREAM");
            }
            SPDLOG_ERROR("INPUT: {}", inputIt->value.GetString());
            std::unique_lock lock(pipe->text2SpeechPipelineMutex);
            auto gen_speech = pipe->text2SpeechPipeline->generate(inputIt->value.GetString());
            lock.unlock();
            drwav_data_format format;
            format.container = drwav_container_riff;
            format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
            format.channels = 1;
            format.sampleRate = 16000;  // assume it is always 16 KHz
            format.bitsPerSample = gen_speech.speeches[0].get_element_type().bitwidth();
            SPDLOG_ERROR("1");
            drwav wav;
            void* ppData;
            size_t pDataSize;

            auto waveform_size = gen_speech.speeches[0].get_size();
            size_t total_samples = waveform_size * format.channels;
            auto waveform_ptr = gen_speech.speeches[0].data<const float>();
            OPENVINO_ASSERT(drwav_init_memory_write_sequential_pcm_frames(&wav, &ppData, &pDataSize, &format, total_samples, nullptr),
                            "Failed to initialize WAV writer");
            SPDLOG_ERROR("2");
            drwav_uint64 frames_written = drwav_write_pcm_frames(&wav, total_samples, waveform_ptr);
            OPENVINO_ASSERT(frames_written == total_samples, "Failed to write not all frames");
             SPDLOG_ERROR("3");
            output = std::make_unique<std::string>(reinterpret_cast<char*>(ppData), pDataSize);
            drwav_uninit(&wav);
            SPDLOG_ERROR("4");
            //drwav_free(ppData, NULL);
        }else {
            return absl::InvalidArgumentError(absl::StrCat("Unsupported URI: ", payload.uri));
        }

        // auto outputOrStatus = generateJSONResponseFromOvTensor(*images);
        // RETURN_IF_HOLDS_STATUS(outputOrStatus);
        // auto output = std::move(std::get<std::unique_ptr<std::string>>(outputOrStatus));
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), cc->InputTimestamp());
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "SpeechCalculator  [Node: {}] Process end", cc->NodeName());

        return absl::OkStatus();
    }
};

const std::string SpeechCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string SpeechCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(SpeechCalculator);

}  // namespace mediapipe

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

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

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
    OPENVINO_ASSERT(drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr), "Failed to open WAV file from stdin");
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
        // cc->InputSidePackets().Tag(IMAGE_GEN_SESSION_SIDE_PACKET_TAG).Set<SpeechPipelinesMap>();  // TODO: template?
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

        // ImageGenerationPipelinesMap pipelinesMap = cc->InputSidePackets().Tag(SPEECH_SESSION_SIDE_PACKET_TAG).Get<SpeechPipelinesMap>();
        // auto it = pipelinesMap.find(cc->NodeName());
        // RET_CHECK(it != pipelinesMap.end()) << "Could not find initialized Speech node named: " << cc->NodeName();
        // auto pipe = it->second;

        auto payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();

        std::unique_ptr<ov::Tensor> images;  // output
        std::unique_ptr<std::string> output;
        if (absl::StartsWith(payload.uri, "/v3/audio/transcriptions")) {
            if (payload.multipartParser->hasParseError())
                return absl::InvalidArgumentError("Failed to parse multipart data");

            SET_OR_RETURN(std::optional<std::string_view>, file, getFileFromPayload(*payload.multipartParser, "file"));
            if(!file.has_value()){
                return absl::InvalidArgumentError(absl::StrCat("File parsing fails"));
            }
            ov::genai::WhisperPipeline pipeline("/models/audio/transcriptions", "CPU");
            ov::genai::WhisperGenerationConfig config = pipeline.get_generation_config();
            // 'task' and 'language' parameters are supported for multilingual models only
            config.language = "<|en|>";  // can switch to <|zh|> for Chinese language
            config.task = "transcribe";
            config.return_timestamps = true;
            ov::genai::RawSpeechInput raw_speech = read_wav(file.value());
            output = std::make_unique<std::string>(pipeline.generate(raw_speech));
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
            ov::genai::Text2SpeechPipeline pipeline("/models/audio/speech", "CPU");
            SPDLOG_ERROR("1");
            auto gen_speech = pipeline.generate(inputIt->value.GetString());
            drwav_data_format format;
            format.container = drwav_container_riff;
            format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
            format.channels = 1;
            format.sampleRate = 16000;  // assume it is always 16 KHz
            format.bitsPerSample = gen_speech.speeches[0].get_element_type().bitwidth();

            drwav wav;
            void* ppData;
            size_t pDataSize;
            OPENVINO_ASSERT(drwav_init_memory_write(&wav, &ppData, &pDataSize, &format, nullptr),
                            "Failed to initialize WAV writer");
            auto waveform_size = gen_speech.speeches[0].get_size();
            size_t total_samples = waveform_size * format.channels;
            auto waveform_ptr = gen_speech.speeches[0].data<const float>();

            drwav_uint64 frames_written = drwav_write_pcm_frames(&wav, total_samples, waveform_ptr);
            OPENVINO_ASSERT(frames_written == total_samples, "Failed to write not all frames");

            SPDLOG_ERROR("SIZE {}", gen_speech.speeches[0].get_size());
            output = std::make_unique<std::string>(reinterpret_cast<char*>(ppData), pDataSize);
            //drwav_free(&wav) TODO: ??
            drwav_uninit(&wav);
            SPDLOG_ERROR("3");
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

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

#include "../../http_payload.hpp"
#include "../../logging.hpp"
#include <mutex>
#include <thread>

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)

#define DR_WAV_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable : 4245 4220)
#include "dr_wav.h"  // NOLINT
#define DR_MP3_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable : 6386 6262)
#include "dr_mp3.h"  // NOLINT
#pragma warning(pop)
#include "tts_servable.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace ovms;

namespace mediapipe {

const std::string TTS_SESSION_SIDE_PACKET_TAG = "TTS_NODE_RESOURCES";

class TtsCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        cc->InputSidePackets().Tag(TTS_SESSION_SIDE_PACKET_TAG).Set<TtsServableMap>();  // TODO: template?
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<std::string>();
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(tts_calculator_logger, "TtsCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(tts_calculator_logger, "TtsCalculator  [Node: {}] Open start", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(tts_calculator_logger, "TtsCalculator  [Node: {}] Process start", cc->NodeName());

        TtsServableMap pipelinesMap = cc->InputSidePackets().Tag(TTS_SESSION_SIDE_PACKET_TAG).Get<TtsServableMap>();
        auto it = pipelinesMap.find(cc->NodeName());
        RET_CHECK(it != pipelinesMap.end()) << "Could not find initialized TTS node named: " << cc->NodeName();
        auto pipe = it->second;

        auto payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();

        std::unique_ptr<std::string> output;
        if (absl::StartsWith(payload.uri, "/v3/audio/speech")) {
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
                return absl::InvalidArgumentError("streaming is not supported");
            }
            std::unique_lock lock(pipe->ttsPipelineMutex);
            auto gen_speech = pipe->ttsPipeline->generate(inputIt->value.GetString());

            drwav_data_format format;
            format.container = drwav_container_riff;
            format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
            format.channels = 1;
            format.sampleRate = 16000;  // assume it is always 16 KHz
            format.bitsPerSample = gen_speech.speeches[0].get_element_type().bitwidth();
            drwav wav;
            void* ppData;
            size_t pDataSize;
            auto waveform_size = gen_speech.speeches[0].get_size();
            size_t total_samples = waveform_size * format.channels;
            ov::Tensor cpu_tensor(gen_speech.speeches[0].get_element_type(), gen_speech.speeches[0].get_shape());
            // copy results to release inference request
            gen_speech.speeches[0].copy_to(cpu_tensor);
            lock.unlock();

            auto waveform_ptr = cpu_tensor.data<const float>();
            OPENVINO_ASSERT(drwav_init_memory_write_sequential_pcm_frames(&wav, &ppData, &pDataSize, &format, total_samples, nullptr),
                "Failed to initialize WAV writer");
            drwav_uint64 frames_written = drwav_write_pcm_frames(&wav, total_samples, waveform_ptr);
            OPENVINO_ASSERT(frames_written == total_samples, "Failed to write not all frames");
            output = std::make_unique<std::string>(reinterpret_cast<char*>(ppData), pDataSize);
            drwav_uninit(&wav);
            // drwav_free(ppData, NULL); TODO: is needed?
        } else {
            return absl::InvalidArgumentError(absl::StrCat("Unsupported URI: ", payload.uri));
        }

        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), cc->InputTimestamp());
        SPDLOG_LOGGER_DEBUG(tts_calculator_logger, "TtsCalculator  [Node: {}] Process end", cc->NodeName());

        return absl::OkStatus();
    }
};

const std::string TtsCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string TtsCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(TtsCalculator);

}  // namespace mediapipe

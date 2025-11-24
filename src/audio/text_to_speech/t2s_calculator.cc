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

#include "src/audio/audio_utils.hpp"
#include "src/http_payload.hpp"
#include "src/logging.hpp"
#include <mutex>
#include <thread>

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#pragma warning(pop)

#include "src/port/dr_audio.hpp"

#include "t2s_servable.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace ovms;

namespace mediapipe {

const std::string TTS_SESSION_SIDE_PACKET_TAG = "TTS_NODE_RESOURCES";

class T2sCalculator : public CalculatorBase {
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
        SPDLOG_LOGGER_DEBUG(t2s_calculator_logger, "T2sCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(t2s_calculator_logger, "T2sCalculator  [Node: {}] Open start", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(t2s_calculator_logger, "T2sCalculator  [Node: {}] Process start", cc->NodeName());

        TtsServableMap pipelinesMap = cc->InputSidePackets().Tag(TTS_SESSION_SIDE_PACKET_TAG).Get<TtsServableMap>();
        auto it = pipelinesMap.find(cc->NodeName());
        RET_CHECK(it != pipelinesMap.end()) << "Could not find initialized TTS node named: " << cc->NodeName();
        auto pipe = it->second;
        const auto& payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();

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
            auto generatedSpeech = pipe->ttsPipeline->generate(inputIt->value.GetString());
            auto bitsPerSample = generatedSpeech.speeches[0].get_element_type().bitwidth();
            auto speechSize = generatedSpeech.speeches[0].get_size();
            ov::Tensor cpuTensor(generatedSpeech.speeches[0].get_element_type(), generatedSpeech.speeches[0].get_shape());
            // copy results to release inference request
            generatedSpeech.speeches[0].copy_to(cpuTensor);
            lock.unlock();
            void* ppData;
            size_t pDataSize;
            prepareAudioOutput(&ppData, pDataSize, bitsPerSample, speechSize, cpuTensor.data<const float>());
            output = std::make_unique<std::string>(reinterpret_cast<char*>(ppData), pDataSize);
            drwav_free(ppData, NULL);
        } else {
            return absl::InvalidArgumentError(absl::StrCat("Unsupported URI: ", payload.uri));
        }

        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), cc->InputTimestamp());
        SPDLOG_LOGGER_DEBUG(t2s_calculator_logger, "T2sCalculator  [Node: {}] Process end", cc->NodeName());

        return absl::OkStatus();
    }
};

const std::string T2sCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string T2sCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(T2sCalculator);

}  // namespace mediapipe

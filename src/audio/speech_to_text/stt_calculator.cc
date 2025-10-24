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

#include "stt_servable.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace ovms;

namespace mediapipe {

const std::string STT_SESSION_SIDE_PACKET_TAG = "STT_NODE_RESOURCES";

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

            ov::genai::RawSpeechInput rawSpeech;
            try {
                if (isWavBuffer(std::string(file))) {
                    SPDLOG_DEBUG("Received file format: wav");
                    rawSpeech = readWav(file);
                } else {
                    rawSpeech = readMp3(file);
                    SPDLOG_DEBUG("Received file format: mp3");
                }
            } catch (std::exception&) {
                return absl::InvalidArgumentError("Received input file is not valid wav nor mp3 audio file");
            }
            std::string result = "{\"text\": \"";
            std::unique_lock lock(pipe->sttPipelineMutex);
            result += pipe->sttPipeline->generate(rawSpeech);
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

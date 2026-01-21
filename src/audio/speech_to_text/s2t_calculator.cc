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

#include "src/port/rapidjson_writer.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "s2t_servable.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace ovms;

namespace mediapipe {

const std::string STT_SESSION_SIDE_PACKET_TAG = "STT_NODE_RESOURCES";

enum Endpoint {
    TRANSCRIPTIONS,
    TRANSLATIONS,
    UNSUPPORTED
};

Endpoint getEndpoint(const std::string& url) {
    if (absl::StartsWith(url, "/v3/audio/transcriptions")) {
        return Endpoint::TRANSCRIPTIONS;
    }
    if (absl::StartsWith(url, "/v3/audio/translations")) {
        return Endpoint::TRANSLATIONS;
    }
    return Endpoint::UNSUPPORTED;
}

size_t ISO_LANG_CODE_MAX = 3;

class S2tCalculator : public CalculatorBase {
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
        SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "SpeechToTextCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "SpeechToTextCalculator  [Node: {}] Open start", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "SpeechToTextCalculator  [Node: {}] Process start", cc->NodeName());

        SttServableMap pipelinesMap = cc->InputSidePackets().Tag(STT_SESSION_SIDE_PACKET_TAG).Get<SttServableMap>();
        auto it = pipelinesMap.find(cc->NodeName());
        RET_CHECK(it != pipelinesMap.end()) << "Could not find initialized STT node named: " << cc->NodeName();
        auto pipe = it->second;

        const auto& payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();
        auto endpoint = getEndpoint(payload.uri);
        if (endpoint == Endpoint::UNSUPPORTED) {
            return absl::InvalidArgumentError(absl::StrCat("Unsupported URI: ", payload.uri));
        }

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

        std::vector<float> rawSpeech;
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
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.StartObject();
        writer.String("text");
        if (endpoint == Endpoint::TRANSCRIPTIONS) {
            std::string_view language = payload.multipartParser->getFileContentByFieldName("language");
            if (!language.empty()) {
                if (language.size() > ISO_LANG_CODE_MAX) {
                    return absl::InvalidArgumentError("Invalid language code.");
                }
                std::string genaiLanguage = "<|" + std::string(language) + "|>";
                std::unique_lock lock(pipe->sttPipelineMutex);
                std::string generatedText = pipe->sttPipeline->generate(rawSpeech, ov::genai::language(genaiLanguage.c_str()));
                lock.unlock();
                writer.String(generatedText.c_str());
            } else {
                std::unique_lock lock(pipe->sttPipelineMutex);
                std::string generatedText = pipe->sttPipeline->generate(rawSpeech);
                lock.unlock();
                writer.String(generatedText.c_str());
            }
        }
        if (endpoint == Endpoint::TRANSLATIONS) {
            std::unique_lock lock(pipe->sttPipelineMutex);
            std::string generatedText = pipe->sttPipeline->generate(rawSpeech, ov::genai::task("translate"));
            lock.unlock();
            writer.String(generatedText.c_str());
        }
        writer.EndObject();
        std::unique_ptr<std::string> output = std::make_unique<std::string>(buffer.GetString());

        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), cc->InputTimestamp());
        SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "SpeechToTextCalculator  [Node: {}] Process end", cc->NodeName());

        return absl::OkStatus();
    }
};

const std::string S2tCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string S2tCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};

REGISTER_CALCULATOR(S2tCalculator);
}  // namespace mediapipe

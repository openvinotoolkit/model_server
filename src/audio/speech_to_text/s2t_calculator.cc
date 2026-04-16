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
#include <string>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "src/audio/audio_utils.hpp"
#include "src/client_connection.hpp"
#include "src/http_payload.hpp"
#include "src/logging.hpp"

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/strings/str_cat.h"
#pragma warning(pop)

#include "src/port/rapidjson_writer.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "s2t_servable.hpp"
#include "s2t_streaming_handler.hpp"

using namespace ovms;

namespace mediapipe {

const std::string STT_SESSION_SIDE_PACKET_TAG = "STT_NODE_RESOURCES";

enum Endpoint {
    TRANSCRIPTIONS,
    TRANSLATIONS,
    UNSUPPORTED
};

static Endpoint getEndpoint(const std::string& url) {
    if (absl::StartsWith(url, "/v3/audio/transcriptions")) {
        return Endpoint::TRANSCRIPTIONS;
    }
    if (absl::StartsWith(url, "/v3/audio/translations")) {
        return Endpoint::TRANSLATIONS;
    }
    return Endpoint::UNSUPPORTED;
}

static absl::Status checkClientDisconnected(const ovms::HttpPayload& payload, const std::string& nodeName, const char* context) {
    if (payload.client && payload.client->isDisconnected()) {
        SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "Client disconnected {} [Node: {}]", context, nodeName);
        return absl::CancelledError("Client disconnected");
    }
    return absl::OkStatus();
}

class S2tCalculator : public CalculatorBase {
    static const std::string INPUT_TAG_NAME;
    static const std::string OUTPUT_TAG_NAME;
    static const std::string LOOPBACK_TAG_NAME;

    bool hasLoopback_ = false;
    S2tStreamingHandler streamingHandler_;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Inputs().Tag(INPUT_TAG_NAME).Set<ovms::HttpPayload>();
        if (cc->Inputs().HasTag(LOOPBACK_TAG_NAME)) {
            cc->Inputs().Tag(LOOPBACK_TAG_NAME).Set<bool>();
        }
        cc->InputSidePackets().Tag(STT_SESSION_SIDE_PACKET_TAG).Set<SttServableMap>();
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Set<std::string>();
        if (cc->Outputs().HasTag(LOOPBACK_TAG_NAME)) {
            cc->Outputs().Tag(LOOPBACK_TAG_NAME).Set<bool>();
        }
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "SpeechToTextCalculator [Node: {} ] Close", cc->NodeName());
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "SpeechToTextCalculator  [Node: {}] Open start", cc->NodeName());
        hasLoopback_ = cc->Inputs().HasTag(LOOPBACK_TAG_NAME);
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "SpeechToTextCalculator  [Node: {}] Process start", cc->NodeName());

        bool loopbackEmpty = !hasLoopback_ || cc->Inputs().Tag(LOOPBACK_TAG_NAME).IsEmpty();
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty() && loopbackEmpty) {
            return absl::OkStatus();
        }

        // --- LOOPBACK iteration: drain streaming queue ---
        if (hasLoopback_ && !cc->Inputs().Tag(LOOPBACK_TAG_NAME).IsEmpty() && streamingHandler_.isActive()) {
            return streamingHandler_.processIteration(cc, LOOPBACK_TAG_NAME, OUTPUT_TAG_NAME);
        }

        // --- First iteration: new request ---
        if (cc->Inputs().Tag(INPUT_TAG_NAME).IsEmpty()) {
            return absl::OkStatus();
        }

        SttServableMap pipelinesMap = cc->InputSidePackets().Tag(STT_SESSION_SIDE_PACKET_TAG).Get<SttServableMap>();
        auto it = pipelinesMap.find(cc->NodeName());
        RET_CHECK(it != pipelinesMap.end()) << "Could not find initialized STT node named: " << cc->NodeName();
        auto pipe = it->second;

        const auto& payload = cc->Inputs().Tag(INPUT_TAG_NAME).Get<ovms::HttpPayload>();
        auto endpoint = getEndpoint(payload.uri);
        if (endpoint == Endpoint::UNSUPPORTED) {
            return absl::InvalidArgumentError(absl::StrCat("Unsupported URI: ", payload.uri));
        }
        try {
            if (payload.multipartParser->hasParseError())
                return absl::InvalidArgumentError("Failed to parse multipart data");

            std::string streamField = payload.multipartParser->getFieldByName("stream");
            bool requestStreaming = (streamField == "true") && hasLoopback_;
            if (streamField == "true" && !hasLoopback_) {
                return absl::InvalidArgumentError("streaming is not supported for this graph configuration (LOOPBACK not configured)");
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

            if (requestStreaming) {
                return streamingHandler_.start(cc, pipe, payload, std::move(rawSpeech),
                    endpoint == Endpoint::TRANSCRIPTIONS, LOOPBACK_TAG_NAME, OUTPUT_TAG_NAME);
            }

            return processUnaryRequest(cc, pipe, endpoint, payload, rawSpeech);

        } catch (ov::AssertFailure& e) {
            return absl::InvalidArgumentError(e.what());
        } catch (...) {
            return absl::InvalidArgumentError("Response generation failed");
        }
        return absl::OkStatus();
    }

private:
    absl::Status processUnaryRequest(CalculatorContext* cc, std::shared_ptr<ovms::SttServable> pipe,
        Endpoint endpoint, const ovms::HttpPayload& payload, const std::vector<float>& rawSpeech) {
        auto client = payload.client;
        auto disconnectCallback = [client](std::string) -> ov::genai::StreamingStatus {
            if (client && client->isDisconnected()) {
                return ov::genai::StreamingStatus::CANCEL;
            }
            return ov::genai::StreamingStatus::RUNNING;
        };
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.StartObject();
        writer.String("text");
        if (endpoint == Endpoint::TRANSCRIPTIONS) {
            ov::genai::WhisperGenerationConfig config = pipe->sttPipeline->get_generation_config();
            auto status = S2tStreamingHandler::applyTranscriptionConfig(config, pipe, payload);
            if (status != absl::OkStatus())
                return status;

            std::unique_lock lock(pipe->sttPipelineMutex);
            auto disconnectStatus = checkClientDisconnected(payload, cc->NodeName(), "before transcription");
            if (!disconnectStatus.ok())
                return disconnectStatus;
            const ov::genai::WhisperDecodedResults result = pipe->sttPipeline->generate(rawSpeech, config, disconnectCallback);
            lock.unlock();
            disconnectStatus = checkClientDisconnected(payload, cc->NodeName(), "after transcription");
            if (!disconnectStatus.ok())
                return disconnectStatus;
            const std::string generatedText = result;
            writer.String(generatedText.c_str());
            serializeTimestamps(writer, result, config);
        }
        if (endpoint == Endpoint::TRANSLATIONS) {
            std::unique_lock lock(pipe->sttPipelineMutex);
            auto disconnectStatus = checkClientDisconnected(payload, cc->NodeName(), "before translation");
            if (!disconnectStatus.ok())
                return disconnectStatus;
            std::string generatedText = pipe->sttPipeline->generate(rawSpeech, ov::genai::task("translate"), ov::genai::streamer(disconnectCallback));
            lock.unlock();
            disconnectStatus = checkClientDisconnected(payload, cc->NodeName(), "after translation");
            if (!disconnectStatus.ok())
                return disconnectStatus;
            writer.String(generatedText.c_str());
        }
        writer.EndObject();
        auto output = std::make_unique<std::string>(buffer.GetString());
        cc->Outputs().Tag(OUTPUT_TAG_NAME).Add(output.release(), cc->InputTimestamp());
        SPDLOG_LOGGER_DEBUG(s2t_calculator_logger, "SpeechToTextCalculator  [Node: {}] Process end", cc->NodeName());
        return absl::OkStatus();
    }

    static void serializeTimestamps(rapidjson::Writer<rapidjson::StringBuffer>& writer,
        const ov::genai::WhisperDecodedResults& result, const ov::genai::WhisperGenerationConfig& config) {
        if (config.word_timestamps) {
            writer.String("words");
            writer.StartArray();
            if (result.words.has_value()) {
                for (const auto& word : *result.words) {
                    writer.StartObject();
                    writer.String("word");
                    writer.String(word.word.c_str());
                    writer.String("start");
                    writer.Double(word.start_ts);
                    writer.String("end");
                    writer.Double(word.end_ts);
                    writer.EndObject();
                }
            }
            writer.EndArray();
        }
        if (config.return_timestamps) {
            writer.String("segments");
            writer.StartArray();
            if (result.chunks.has_value()) {
                for (const auto& chunk : *result.chunks) {
                    writer.StartObject();
                    writer.String("text");
                    writer.String(chunk.text.c_str());
                    writer.String("start");
                    writer.Double(chunk.start_ts);
                    writer.String("end");
                    writer.Double(chunk.end_ts);
                    writer.EndObject();
                }
            }
            writer.EndArray();
        }
    }
};

const std::string S2tCalculator::INPUT_TAG_NAME{"HTTP_REQUEST_PAYLOAD"};
const std::string S2tCalculator::OUTPUT_TAG_NAME{"HTTP_RESPONSE_PAYLOAD"};
const std::string S2tCalculator::LOOPBACK_TAG_NAME{"LOOPBACK"};

REGISTER_CALCULATOR(S2tCalculator);
}  // namespace mediapipe

//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "src/audio/audio_utils.hpp"
#include "src/http_payload.hpp"
#include "src/logging.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include "src/sse_utils.hpp"
#include "src/webrtc/voxtral_audio_adapter.hpp"
#include "s2t_servable.hpp"
#include "s2t_streaming_handler.hpp"
#include "streaming_s2t.hpp"

namespace mediapipe {

class StreamingS2TCalculator : public CalculatorBase {
    struct Utterance {
        std::shared_ptr<ovms::VoxtralAudioAdapter::Stream> stream;
        bool finishing = false;
    };
    std::shared_ptr<ovms::SttServable> servable_;
    std::shared_ptr<ovms::VoxtralAudioAdapter> adapter_;
    std::shared_ptr<ovms::StreamingS2TBridge> bridge_;
    std::unordered_map<std::string, Utterance> utterances_;
    std::vector<std::shared_ptr<ovms::VoxtralAudioAdapter::Stream>> retired_;
    std::shared_ptr<ovms::ClientConnection> client_;
    std::shared_ptr<std::atomic<bool>> httpActive_;
    bool http_ = false;
    bool httpStreaming_ = false;
    int64_t outputTimestamp_ = 0;
    uint64_t requestId_ = 0;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        const bool http = cc->Inputs().HasTag("HTTP_REQUEST_PAYLOAD");
        RET_CHECK(http != cc->Inputs().HasTag("AUDIO"));
        if (http) {
            cc->Inputs().Tag("HTTP_REQUEST_PAYLOAD").Set<ovms::HttpPayload>();
            cc->Outputs().Tag("HTTP_RESPONSE_PAYLOAD").Set<std::string>();
        } else {
            cc->Inputs().Tag("AUDIO").Set<ovms::StreamingS2TInput>();
            cc->Outputs().Tag("EVENT").Set<ovms::StreamingS2TEvent>();
        }
        // WAKE and its side packet are injected by StreamingS2TGraphExecution.
        if (cc->Inputs().HasTag("WAKE")) {
            cc->Inputs().Tag("WAKE").Set<bool>();
            cc->InputSidePackets().Tag("STREAMING_S2T_RESOURCES").Set<ovms::StreamingS2TBridgeMap>();
        }
        if (cc->Inputs().HasTag("LOOPBACK"))
            cc->Inputs().Tag("LOOPBACK").Set<bool>();
        if (cc->Outputs().HasTag("LOOPBACK"))
            cc->Outputs().Tag("LOOPBACK").Set<bool>();
        cc->InputSidePackets().Tag("STT_NODE_RESOURCES").Set<ovms::SttServableMap>();
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) override {
        RET_CHECK(cc->Inputs().HasTag("WAKE")) << "Streaming STT requires the result-ready graph bridge";
        const auto& servables = cc->InputSidePackets().Tag("STT_NODE_RESOURCES").Get<ovms::SttServableMap>();
        const auto servable = servables.find(cc->NodeName());
        RET_CHECK(servable != servables.end()) << "Missing initialized STT servable";
        const auto& bridges = cc->InputSidePackets().Tag("STREAMING_S2T_RESOURCES").Get<ovms::StreamingS2TBridgeMap>();
        const auto bridge = bridges.find(cc->NodeName());
        RET_CHECK(bridge != bridges.end()) << "Missing streaming STT result bridge";
        servable_ = servable->second;
        bridge_ = bridge->second;
        http_ = cc->Inputs().HasTag("HTTP_REQUEST_PAYLOAD");
        try {
            adapter_ = servable_->createStreamingAdapter();
        } catch (const std::exception& e) {
            return absl::FailedPreconditionError(e.what());
        }
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) override {
        retired_.erase(std::remove_if(retired_.begin(), retired_.end(),
                           [](const auto& stream) { return stream->isCompleted(); }),
            retired_.end());
        try {
            if (http_ && !cc->Inputs().Tag("HTTP_REQUEST_PAYLOAD").IsEmpty()) {
                const auto status = startHttp(cc->Inputs().Tag("HTTP_REQUEST_PAYLOAD").Get<ovms::HttpPayload>());
                if (!status.ok())
                    return status;
            } else if (!http_ && !cc->Inputs().Tag("AUDIO").IsEmpty()) {
                const auto& input = cc->Inputs().Tag("AUDIO").Get<ovms::StreamingS2TInput>();
                try {
                    handleInput(input);
                } catch (const std::exception& e) {
                    SPDLOG_ERROR("Streaming STT utterance {} rejected: {}", input.utteranceId, e.what());
                    const auto it = utterances_.find(input.utteranceId);
                    if (it != utterances_.end()) {
                        it->second.stream->cancel();
                        retired_.push_back(std::move(it->second.stream));
                        utterances_.erase(it);
                        bridge_->complete();
                    }
                    bridge_->publish({ovms::StreamingS2TEvent::Type::Error, input.utteranceId, e.what()});
                }
            }
            for (auto& event : bridge_->drain()) {
                if (event.type == ovms::StreamingS2TEvent::Type::Error &&
                    event.text == "Streaming STT result queue is full") {
                    for (auto& entry : utterances_) {
                        entry.second.stream->cancel();
                        retired_.push_back(std::move(entry.second.stream));
                        bridge_->complete();
                    }
                    utterances_.clear();
                }
                if (event.type != ovms::StreamingS2TEvent::Type::Delta) {
                    if (httpActive_)
                        httpActive_->store(false);
                    const auto it = utterances_.find(event.utteranceId);
                    if (it != utterances_.end()) {
                        if (event.type == ovms::StreamingS2TEvent::Type::Error)
                            it->second.stream->cancel();
                        retired_.push_back(std::move(it->second.stream));
                        utterances_.erase(it);
                        bridge_->complete();
                    }
                }
                if (http_) {
                    if (client_ && client_->isDisconnected())
                        return absl::CancelledError("HTTP transcription client disconnected");
                    if (event.type == ovms::StreamingS2TEvent::Type::Error)
                        return absl::InternalError(event.text);
                    if (!httpStreaming_ && event.type == ovms::StreamingS2TEvent::Type::Delta)
                        continue;
                    std::string response;
                    if (httpStreaming_) {
                        response = ovms::wrapTextInServerSideEventMessage(
                            event.type == ovms::StreamingS2TEvent::Type::Delta ?
                                S2tStreamingHandler::serializeDeltaEvent(event.text) :
                                S2tStreamingHandler::serializeDoneEvent(event.text));
                    } else {
                        rapidjson::StringBuffer buffer;
                        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                        writer.StartObject();
                        writer.Key("text");
                        writer.String(event.text.data(), static_cast<rapidjson::SizeType>(event.text.size()));
                        writer.EndObject();
                        response = buffer.GetString();
                    }
                    cc->Outputs().Tag("HTTP_RESPONSE_PAYLOAD").Add(
                        new std::string(std::move(response)), Timestamp(++outputTimestamp_));
                } else {
                    cc->Outputs().Tag("EVENT").Add(
                        new ovms::StreamingS2TEvent(std::move(event)), Timestamp(++outputTimestamp_));
                }
            }
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Streaming STT processing failed: {}", e.what());
            return absl::InternalError(e.what());
        }
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext*) override {
        if (bridge_)
            bridge_->invalidate();
        if (httpActive_)
            httpActive_->store(false);
        for (const auto& entry : utterances_)
            entry.second.stream->cancel();
        for (const auto& stream : retired_)
            stream->cancel();
        utterances_.clear();
        retired_.clear();
        return absl::OkStatus();
    }

private:
    Utterance& startUtterance(const std::string& id) {
        if (id.empty())
            throw std::invalid_argument("Streaming STT requires an utterance ID");
        if (utterances_.size() + retired_.size() >= 8)
            throw std::runtime_error("Too many active or finishing STT utterances");
        const auto bridge = bridge_;
        auto stream = adapter_->createStream(
            [bridge, id](const std::string& text) {
                bridge->publish({ovms::StreamingS2TEvent::Type::Delta, id, text});
            },
            [bridge, id](const std::string& text) {
                bridge->publish({ovms::StreamingS2TEvent::Type::Final, id, text});
            },
            [bridge, id](std::exception_ptr error) {
                std::string message = "Unknown streaming STT inference failure";
                try {
                    if (error)
                        std::rethrow_exception(error);
                } catch (const std::exception& e) {
                    message = e.what();
                } catch (...) {
                    SPDLOG_ERROR("Streaming STT inference threw an unknown exception");
                }
                SPDLOG_ERROR("Streaming STT utterance {} failed: {}", id, message);
                bridge->publish({ovms::StreamingS2TEvent::Type::Error, id, std::move(message)});
            });
        const auto inserted = utterances_.emplace(id, Utterance{std::move(stream), false});
        if (!inserted.second)
            throw std::invalid_argument("Duplicate active STT utterance ID");
        try {
            bridge_->begin();
        } catch (...) {
            utterances_.erase(inserted.first);
            throw;
        }
        return inserted.first->second;
    }

    void handleInput(const ovms::StreamingS2TInput& input) {
        auto it = utterances_.find(input.utteranceId);
        if (input.type == ovms::StreamingS2TInput::Type::Audio) {
            auto& utterance = it == utterances_.end() ? startUtterance(input.utteranceId) : it->second;
            if (utterance.finishing)
                throw std::invalid_argument("Audio received after utterance finish");
            if (!input.audio)
                throw std::invalid_argument("Streaming STT audio packet has no audio");
            utterance.stream->push(std::move(*input.audio));
        } else {
            if (it == utterances_.end())
                throw std::invalid_argument("Unknown or empty STT utterance");
            if (input.type == ovms::StreamingS2TInput::Type::Finish) {
                it->second.finishing = true;
                it->second.stream->finish();
            } else {
                it->second.stream->cancel();
                retired_.push_back(std::move(it->second.stream));
                utterances_.erase(it);
                bridge_->complete();
            }
        }
    }

    absl::Status startHttp(const ovms::HttpPayload& payload) {
        if (!utterances_.empty())
            return absl::FailedPreconditionError("An HTTP transcription is already active");
        if (payload.uri.rfind("/v1/audio/transcriptions", 0) != 0 &&
            payload.uri.rfind("/v3/audio/transcriptions", 0) != 0)
            return absl::InvalidArgumentError("Voxtral realtime supports only audio transcriptions");
        if (!payload.multipartParser || payload.multipartParser->hasParseError())
            return absl::InvalidArgumentError("Failed to parse multipart data");
        const auto& parser = *payload.multipartParser;
        for (const auto* field : {"language", "prompt"}) {
            if (!parser.getFieldByName(field).empty())
                return absl::InvalidArgumentError(std::string("Voxtral realtime does not support: ") + field);
        }
        if (!parser.getArrayFieldByName("timestamp_granularities[]").empty())
            return absl::InvalidArgumentError("Voxtral realtime does not support timestamps");
        const std::string format = parser.getFieldByName("response_format");
        if (!format.empty() && format != "json")
            return absl::InvalidArgumentError("Voxtral realtime supports only JSON response format");
        auto config = servable_->sttPipeline->get_generation_config();
        const std::string requestedTemperature = parser.getFieldByName("temperature");
        auto status = ovms::SttServable::parseTemperature(payload, config);
        if (!status.ok())
            return status;
        if (!requestedTemperature.empty() && config.temperature != 0)
            return absl::InvalidArgumentError("Voxtral realtime does not support nonzero temperature");
        const std::string streaming = parser.getFieldByName("stream");
        if (!streaming.empty() && streaming != "true" && streaming != "false")
            return absl::InvalidArgumentError("Invalid stream field; expected true or false");
        const auto file = parser.getFileContentByFieldName("file");
        if (file.empty())
            return absl::InvalidArgumentError("Missing or empty audio file");
        std::vector<float> audio;
        try {
            audio = ovms::audio_utils::isWavBuffer(std::string(file)) ?
                ovms::audio_utils::readWav(file) : ovms::audio_utils::readMp3(file);
        } catch (const std::exception& e) {
            SPDLOG_DEBUG("Streaming STT audio decoding failed: {}", e.what());
            return absl::InvalidArgumentError("Received input file is not valid wav nor mp3 audio file");
        }
        if (audio.empty())
            return absl::InvalidArgumentError("Decoded audio is empty");
        client_ = payload.client;
        if (client_ && client_->isDisconnected())
            return absl::CancelledError("HTTP transcription client disconnected");
        httpStreaming_ = streaming == "true";
        httpActive_ = std::make_shared<std::atomic<bool>>(true);
        const std::string id = "http_" + std::to_string(++requestId_);
        auto& utterance = startUtterance(id);
        if (client_) {
            const std::weak_ptr<ovms::VoxtralAudioAdapter::Stream> weakStream = utterance.stream;
            const std::weak_ptr<ovms::StreamingS2TBridge> weakBridge = bridge_;
            const std::weak_ptr<std::atomic<bool>> weakActive = httpActive_;
            client_->registerDisconnectionCallback([weakStream, weakBridge, weakActive, id] {
                const auto active = weakActive.lock();
                if (!active || !active->exchange(false))
                    return;
                if (const auto stream = weakStream.lock()) {
                    stream->cancel();
                    if (const auto bridge = weakBridge.lock())
                        bridge->publish({ovms::StreamingS2TEvent::Type::Error, id, "HTTP transcription client disconnected"});
                }
            });
        }
        utterance.stream->push(ovms::AudioChunk{std::move(audio), 16000, 0, 1});
        utterance.finishing = true;
        utterance.stream->finish();
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(StreamingS2TCalculator);
}  // namespace mediapipe

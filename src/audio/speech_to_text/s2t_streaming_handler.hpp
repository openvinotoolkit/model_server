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
#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 6246 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "openvino/genai/whisper_pipeline.hpp"

#include "src/client_connection.hpp"
#include "src/http_payload.hpp"
#include "src/sse_utils.hpp"
#include "src/port/rapidjson_writer.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "streaming_text_queue.hpp"
#include "s2t_servable.hpp"

namespace mediapipe {

// Encapsulates all streaming state and logic for the S2tCalculator.
// Manages the background generation thread, the text queue, SSE
// serialization and LOOPBACK signaling.
class S2tStreamingHandler {
public:
    bool isActive() const { return isStreaming_; }

    static absl::Status parseTemperature(const ovms::HttpPayload& payload, float& temperature);

    static std::string serializeTextChunk(const std::string& text) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.StartObject();
        writer.String("text");
        writer.String(text.c_str());
        writer.EndObject();
        return buffer.GetString();
    }

    absl::Status start(CalculatorContext* cc,
        std::shared_ptr<ovms::SttServable> pipe,
        const ovms::HttpPayload& payload,
        std::vector<float> rawSpeech,
        bool isTranscription,
        const std::string& loopbackTag,
        const std::string& outputTag) {
        isStreaming_ = true;
        accumulatedText_.clear();
        streamingQueue_ = std::make_shared<ovms::StreamingTextQueue>();

        auto queue = streamingQueue_;
        auto client = payload.client;
        auto streamerCallback = [queue, client](std::string text) -> ov::genai::StreamingStatus {
            if (client && client->isDisconnected()) {
                queue->setDone();
                return ov::genai::StreamingStatus::CANCEL;
            }
            if (!text.empty()) {
                queue->push(std::move(text));
            }
            return ov::genai::StreamingStatus::RUNNING;
        };

        if (isTranscription) {
            ov::genai::WhisperGenerationConfig config = pipe->sttPipeline->get_generation_config();
            auto status = applyTranscriptionConfig(config, pipe, payload);
            if (status != absl::OkStatus()) {
                isStreaming_ = false;
                return status;
            }
            ovms::SttServable::StreamingJob job(
                [pipe, rawSpeech = std::move(rawSpeech), config, streamerCallback, queue]() mutable -> ov::genai::WhisperDecodedResults {
                    try {
                        std::unique_lock lock(pipe->sttPipelineMutex);
                        auto result = pipe->sttPipeline->generate(rawSpeech, config, streamerCallback);
                        lock.unlock();
                        queue->setDone();
                        return result;
                    } catch (...) {
                        queue->setDone();
                        throw;
                    }
                });
            try {
                generateFuture_ = pipe->addRequest(std::move(job));
            } catch (const std::exception& e) {
                isStreaming_ = false;
                return absl::InternalError(e.what());
            }
        } else {
            float temperature = 1.0f;
            auto tempStatus = parseTemperature(payload, temperature);
            if (tempStatus != absl::OkStatus()) {
                isStreaming_ = false;
                return tempStatus;
            }
            ovms::SttServable::StreamingJob job(
                [pipe, rawSpeech = std::move(rawSpeech), streamerCallback, queue, temperature]() mutable -> ov::genai::WhisperDecodedResults {
                    try {
                        std::unique_lock lock(pipe->sttPipelineMutex);
                        auto result = pipe->sttPipeline->generate(rawSpeech,
                            ov::genai::task("translate"),
                            ov::genai::temperature(temperature),
                            ov::genai::streamer(streamerCallback));
                        lock.unlock();
                        queue->setDone();
                        return result;
                    } catch (...) {
                        queue->setDone();
                        throw;
                    }
                });
            try {
                generateFuture_ = pipe->addRequest(std::move(job));
            } catch (const std::exception& e) {
                isStreaming_ = false;
                return absl::InternalError(e.what());
            }
        }

        // Trigger first LOOPBACK iteration
        iterationTimestamp_ = cc->InputTimestamp();
        cc->Outputs().Tag(loopbackTag).Add(new bool{true}, iterationTimestamp_);
        return absl::OkStatus();
    }

    absl::Status processIteration(CalculatorContext* cc,
        const std::string& loopbackTag,
        const std::string& outputTag) {
        std::string chunk;
        bool hasData = streamingQueue_->waitAndPop(chunk);

        if (hasData) {
            accumulatedText_ += chunk;
            std::string ssePayload = ovms::wrapTextInServerSideEventMessage(serializeTextChunk(chunk));
            cc->Outputs().Tag(outputTag).Add(new std::string{std::move(ssePayload)}, iterationTimestamp_);

            // Continue looping
            auto now = std::chrono::system_clock::now();
            iterationTimestamp_ = ::mediapipe::Timestamp(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
            cc->Outputs().Tag(loopbackTag).Add(new bool{true}, iterationTimestamp_);
        } else {
            // Generation complete — send final event and stop
            try {
                if (generateFuture_.valid()) {
                    generateFuture_.get();  // propagate any exceptions
                }
            } catch (ov::AssertFailure& e) {
                isStreaming_ = false;
                return absl::InvalidArgumentError(e.what());
            } catch (...) {
                isStreaming_ = false;
                return absl::InvalidArgumentError("Response generation failed");
            }

            std::string doneEvent = ovms::wrapTextInServerSideEventMessage("[DONE]");
            cc->Outputs().Tag(outputTag).Add(new std::string{std::move(doneEvent)}, iterationTimestamp_);
            isStreaming_ = false;
        }

        return absl::OkStatus();
    }

    // Reused by both streaming start (for config) and unary path.
    // Kept here to avoid duplicating the parsing logic.
    static absl::Status applyTranscriptionConfig(ov::genai::WhisperGenerationConfig& config,
        const std::shared_ptr<ovms::SttServable>& pipe, const ovms::HttpPayload& payload);

private:
    bool isStreaming_ = false;
    std::shared_ptr<ovms::StreamingTextQueue> streamingQueue_;
    std::future<ov::genai::WhisperDecodedResults> generateFuture_;
    std::string accumulatedText_;
    ::mediapipe::Timestamp iterationTimestamp_{0};
};

}  // namespace mediapipe

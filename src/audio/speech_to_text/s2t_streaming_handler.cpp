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
#include "s2t_streaming_handler.hpp"

#include <chrono>
#include <utility>
#include <vector>

#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include "src/http_payload.hpp"
#include "src/logging.hpp"
#include "src/sse_utils.hpp"
#include "src/stringutils.hpp"
#include "streaming_text_queue.hpp"
#include "s2t_servable.hpp"

namespace mediapipe {

std::string S2tStreamingHandler::serializeDeltaEvent(const std::string& delta) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.String("type");
    writer.String("transcript.text.delta");
    writer.String("delta");
    writer.String(delta.c_str());
    writer.String("logprobs");
    writer.StartArray();
    writer.EndArray();
    writer.EndObject();
    return buffer.GetString();
}

std::string S2tStreamingHandler::serializeDoneEvent(const std::string& text) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.String("type");
    writer.String("transcript.text.done");
    writer.String("text");
    writer.String(text.c_str());
    writer.String("logprobs");
    writer.StartArray();
    writer.EndArray();
    writer.EndObject();
    return buffer.GetString();
}

absl::Status S2tStreamingHandler::start(CalculatorContext* cc,
    std::shared_ptr<ovms::SttServable> pipe,
    const ovms::HttpPayload& payload,
    std::vector<float> rawSpeech,
    bool isTranscription,
    const std::string& loopbackTag,
    const std::string&) {
    if (isStreaming_) {
        return absl::FailedPreconditionError("Streaming request is already active");
    }
    isStreaming_ = true;
    streamingQueue_ = std::make_shared<ovms::StreamingTextQueue>();
    executionContext_.reset();

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
        auto status = ovms::SttServable::applyTranscriptionConfig(config, pipe, payload);
        if (status != absl::OkStatus()) {
            isStreaming_ = false;
            return status;
        }
        auto executionContext = std::make_shared<ovms::SttServableExecutionContext>(ovms::SttServable::StreamingJob(
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
            }));
        try {
            pipe->addRequest(executionContext);
            executionContext_ = std::move(executionContext);
        } catch (const std::exception& e) {
            isStreaming_ = false;
            return absl::InternalError(e.what());
        }
    } else {
        float temperature = pipe->sttPipeline->get_generation_config().temperature;
        auto tempStatus = ovms::SttServable::parseTemperature(payload, temperature);
        if (tempStatus != absl::OkStatus()) {
            isStreaming_ = false;
            return tempStatus;
        }
        auto executionContext = std::make_shared<ovms::SttServableExecutionContext>(ovms::SttServable::StreamingJob(
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
            }));
        try {
            pipe->addRequest(executionContext);
            executionContext_ = std::move(executionContext);
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

absl::Status S2tStreamingHandler::processIteration(CalculatorContext* cc,
    const std::string& loopbackTag,
    const std::string& outputTag) {
    std::string chunk;
    bool hasData = streamingQueue_->waitAndPop(chunk);

    if (hasData) {
        std::string ssePayload = ovms::wrapTextInServerSideEventMessage(serializeDeltaEvent(chunk));
        cc->Outputs().Tag(outputTag).Add(new std::string{std::move(ssePayload)}, iterationTimestamp_);

        // Continue looping
        auto now = std::chrono::system_clock::now();
        iterationTimestamp_ = ::mediapipe::Timestamp(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
        cc->Outputs().Tag(loopbackTag).Add(new bool{true}, iterationTimestamp_);
    } else {
        // Generation complete — send final event and stop
        std::string finalText;
        try {
            if (executionContext_ && executionContext_->finished.valid()) {
                const ov::genai::WhisperDecodedResults result = executionContext_->finished.get();
                finalText = result;
            }
        } catch (ov::AssertFailure& e) {
            isStreaming_ = false;
            executionContext_.reset();
            return absl::InvalidArgumentError(e.what());
        } catch (...) {
            isStreaming_ = false;
            executionContext_.reset();
            return absl::InvalidArgumentError("Response generation failed");
        }

        std::string doneEvent = ovms::wrapTextInServerSideEventMessage(serializeDoneEvent(finalText));
        cc->Outputs().Tag(outputTag).Add(new std::string{std::move(doneEvent)}, iterationTimestamp_);
        isStreaming_ = false;
        executionContext_.reset();
    }

    return absl::OkStatus();
}

}  // namespace mediapipe

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

static constexpr size_t ISO_LANG_CODE_MAX = 3;

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
        float temperature = pipe->sttPipeline->get_generation_config().temperature;
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

absl::Status S2tStreamingHandler::processIteration(CalculatorContext* cc,
    const std::string& loopbackTag,
    const std::string& outputTag) {
    std::string chunk;
    bool hasData = streamingQueue_->waitAndPop(chunk);

    if (hasData) {
        accumulatedText_ += chunk;
        std::string ssePayload = ovms::wrapTextInServerSideEventMessage(serializeDeltaEvent(chunk));
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

        std::string doneEvent = ovms::wrapTextInServerSideEventMessage(serializeDoneEvent(accumulatedText_));
        cc->Outputs().Tag(outputTag).Add(new std::string{std::move(doneEvent)}, iterationTimestamp_);
        isStreaming_ = false;
    }

    return absl::OkStatus();
}

absl::Status S2tStreamingHandler::parseTemperature(const ovms::HttpPayload& payload, float& temperature) {
    std::string temperatureStr = payload.multipartParser->getFieldByName("temperature");
    if (temperatureStr.size() > 0) {
        SPDLOG_LOGGER_TRACE(ovms::s2t_calculator_logger, "Received temperature: {}", temperatureStr);
        auto temp = ovms::stof(temperatureStr);
        if (!temp.has_value()) {
            temp = ovms::stou32(temperatureStr);
            if (!temp.has_value())
                return absl::InvalidArgumentError("Invalid temperature type.");
        }
        if (temp.value() < 0.0f || temp.value() > 2.0f)
            return absl::InvalidArgumentError("Temperature out of range(0.0, 2.0)");
        temperature = temp.value();
    }
    return absl::OkStatus();
}

absl::Status S2tStreamingHandler::applyTranscriptionConfig(ov::genai::WhisperGenerationConfig& config,
    const std::shared_ptr<ovms::SttServable>& pipe, const ovms::HttpPayload& payload) {
    std::string language = payload.multipartParser->getFieldByName("language");
    if (language.size() > 0) {
        if (language.size() > ISO_LANG_CODE_MAX) {
            return absl::InvalidArgumentError("Invalid language code.");
        }
        SPDLOG_LOGGER_TRACE(ovms::s2t_calculator_logger, "Received language: {}", language);
        config.language = "<|" + language + "|>";
    }
    std::vector<std::string> timestampsTypes = payload.multipartParser->getArrayFieldByName("timestamp_granularities[]");
    config.word_timestamps = false;
    for (const auto& timestampsType : timestampsTypes) {
        SPDLOG_LOGGER_TRACE(ovms::s2t_calculator_logger, "Received timestamp type: {}", timestampsType);
        if (timestampsType == "segment") {
            config.return_timestamps = true;
        } else if (timestampsType == "word") {
            if (!pipe->enableWordTimestamps)
                return absl::InvalidArgumentError("Word timestamps not supported for this model");
            config.word_timestamps = true;
        } else {
            return absl::InvalidArgumentError("Invalid timestamp_granularities type. Allowed types: \"segment\", \"word\"");
        }
    }
    auto status = parseTemperature(payload, config.temperature);
    if (status != absl::OkStatus())
        return status;
    return absl::OkStatus();
}

}  // namespace mediapipe

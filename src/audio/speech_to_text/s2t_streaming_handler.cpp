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

absl::Status S2tStreamingHandler::start(std::shared_ptr<ovms::SttServable> pipe,
    const ovms::HttpPayload& payload,
    std::vector<float> rawSpeech,
    const ov::genai::WhisperGenerationConfig& config) {
    if (isStreaming_) {
        return absl::FailedPreconditionError("Streaming request is already active");
    }
    isStreaming_ = true;
    streamingQueue_ = std::make_shared<ovms::StreamingTextQueue>();
    executionContext_.reset();

    auto client = payload.client;
    auto streamerCallback = [queue = streamingQueue_, client](std::string text) -> ov::genai::StreamingStatus {
        if (client && client->isDisconnected()) {
            queue->endStreaming();
            return ov::genai::StreamingStatus::CANCEL;
        }
        if (!text.empty()) {
            queue->push(std::move(text));
        }
        return ov::genai::StreamingStatus::RUNNING;
    };

    auto guardedStreamerCallback = [streamerCallback = std::move(streamerCallback), queue = streamingQueue_](std::string text) mutable -> ov::genai::StreamingStatus {
        try {
            return streamerCallback(std::move(text));
        } catch (...) {
            queue->endStreaming();
            throw;
        }
    };
    auto executionContext = std::make_shared<ovms::SttServableExecutionContext>(
        std::move(rawSpeech),
        config,
        std::move(guardedStreamerCallback),
        [queue = streamingQueue_]() { queue->endStreaming(); });
    try {
        pipe->addRequest(executionContext);
        executionContext_ = std::move(executionContext);
    } catch (const std::exception& e) {
        isStreaming_ = false;
        return absl::InternalError(e.what());
    }

    return absl::OkStatus();
}

absl::Status S2tStreamingHandler::processIteration(std::string& ssePayload,
    bool& shouldContinueLoopback,
    bool& hasOutput) {
    hasOutput = false;
    shouldContinueLoopback = false;

    std::string chunk;
    bool hasData = streamingQueue_->waitAndPop(chunk);

    if (hasData) {
        ssePayload = ovms::wrapTextInServerSideEventMessage(serializeDeltaEvent(chunk));
        hasOutput = true;
        shouldContinueLoopback = true;
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

        ssePayload = ovms::wrapTextInServerSideEventMessage(serializeDoneEvent(finalText));
        hasOutput = true;
        isStreaming_ = false;
        executionContext_.reset();
    }

    return absl::OkStatus();
}

}  // namespace mediapipe

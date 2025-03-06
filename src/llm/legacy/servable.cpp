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

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../logging.hpp"
#include "../../status.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../http_payload.hpp"
#include "../../mediapipe_internal/mediapipe_utils.hpp"
#include "../apis/openai_completions.hpp"
#include "../text_processor.hpp"
#include "servable.hpp"

namespace ovms {

static std::string wrapTextInServerSideEventMessage(const std::string& text) {
    std::stringstream ss;
    ss << "data: " << text << "\n\n";
    return ss.str();
}

// Node resources interface start
std::shared_ptr<GenAiServableExecutionContext> LegacyServable::createExecutionContext() {
    return std::make_shared<LegacyServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> LegacyServable::getProperties() {
    return properties;
}

absl::Status LegacyServable::scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    properties->legacyExecutor->addRequest(legacyExecutionContext);
    return absl::OkStatus();
}

absl::Status LegacyServable::readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    legacyExecutionContext->finished.wait();
    // std::lock_guard<std::mutex> guard(pipelineMutex);
    // legacyExecutionContext->results = properties->pipeline->generate(legacyExecutionContext->inputIds, legacyExecutionContext->apiHandler->createGenerationConfig());
    return absl::OkStatus();
}

absl::Status LegacyServable::parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    executionContext->apiHandler = std::make_shared<OpenAIChatCompletionsHandler>(*executionContext->payload.parsedJson,
        executionContext->endpoint,
        std::chrono::system_clock::now(),
        getProperties()->tokenizer);

    auto status = executionContext->apiHandler->parseRequest(getProperties()->maxTokensLimit, getProperties()->bestOfLimit, getProperties()->isSpeculativePipeline);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to parse request: {}", status.message());
        return status;
    }

    if (executionContext->apiHandler->isStream()) {
        executionContext->lastStreamerCallbackOutput = "";  // initialize with empty string
        auto callback = [& lastStreamerCallbackOutput = executionContext->lastStreamerCallbackOutput](std::string text) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Streamer callback executed with text: [{}]", text);
            lastStreamerCallbackOutput += text;
            return ov::genai::StreamingStatus::RUNNING;
        };

        executionContext->textStreamer = std::make_shared<ov::genai::TextStreamer>(getProperties()->tokenizer, callback);
    }
    return absl::OkStatus();
}

absl::Status LegacyServable::prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    executionContext->response = executionContext->apiHandler->serializeUnaryResponse(legacyExecutionContext->results);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", executionContext->response);
    return absl::OkStatus();
}

absl::Status LegacyServable::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    std::stringstream ss;
    ss << executionContext->lastStreamerCallbackOutput;
    std::string lastTextChunk = ss.str();
    executionContext->lastStreamerCallbackOutput = "";
    auto generationStatus = legacyExecutionContext->finished.wait_for(std::chrono::nanoseconds::zero());
    if (generationStatus != std::future_status::ready) {  // continue
        if (lastTextChunk.size() > 0) {
            executionContext->response = wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingChunk(lastTextChunk, ov::genai::GenerationFinishReason::NONE));
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", executionContext->response);
        }
        executionContext->sendLoopbackSignal = true;
    } else {  // finish generation
        OVMS_PROFILE_SCOPE("Generation of last streaming response");
        executionContext->textStreamer->end();
        // if streamer::put returned a value, streamer::end() result will not contain it, so we add it manually
        if (!executionContext->lastStreamerCallbackOutput.empty())
            lastTextChunk = lastTextChunk + executionContext->lastStreamerCallbackOutput;
        executionContext->response = wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingChunk(lastTextChunk, ov::genai::GenerationFinishReason::STOP));
        if (executionContext->apiHandler->getStreamOptions().includeUsage)
            executionContext->response += wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingUsageChunk());

        executionContext->response += wrapTextInServerSideEventMessage("[DONE]");

        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", lastTextChunk);
        executionContext->sendLoopbackSignal = false;
        return absl::OkStatus();
    }
    return absl::OkStatus();
}

absl::Status LegacyServable::readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Relevant properties read");
    return absl::OkStatus();
}

bool LegacyServable::supportsSpeculativeDecoding() const {
    return true;
}
}  // namespace ovms

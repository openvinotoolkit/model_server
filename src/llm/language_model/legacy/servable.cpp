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

#include "../../../logging.hpp"
#include "../../../status.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../../http_payload.hpp"
#include "../../../mediapipe_internal/mediapipe_utils.hpp"
#include "../../apis/openai_completions.hpp"
#include "../../text_processor.hpp"
#include "servable.hpp"

namespace ovms {

absl::Status LegacyServable::validateInputComplianceWithProperties(const ov::Tensor& inputIds) const {
    if (properties->device == "NPU") {
        int64_t inputLength = inputIds.get_size();
        if (inputLength > properties->maxPromptLength) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Input length exceeds the maximum allowed length: {} > {}", inputLength, properties->maxPromptLength);
            return absl::InvalidArgumentError("Input length exceeds the maximum allowed length");
        }
    }
    return absl::OkStatus();
}

// Node resources interface start
std::shared_ptr<GenAiServableExecutionContext> LegacyServable::createExecutionContext() {
    return std::make_shared<LegacyServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> LegacyServable::getProperties() {
    return properties;
}

absl::Status LegacyServable::parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    legacyExecutionContext->apiHandler = std::make_shared<OpenAIChatCompletionsHandler>(*legacyExecutionContext->payload.parsedJson,
        legacyExecutionContext->endpoint,
        std::chrono::system_clock::now(),
        getProperties()->tokenizer);

    auto status = executionContext->apiHandler->parseRequest(getProperties()->maxTokensLimit, getProperties()->bestOfLimit, getProperties()->maxModelLength);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to parse request: {}", status.message());
        return status;
    }

    if (legacyExecutionContext->apiHandler->isStream()) {
        legacyExecutionContext->lastStreamerCallbackOutput = "";  // initialize with empty string
        auto callback = [& executionInProgress = legacyExecutionContext->executionInProgress, &mutex = legacyExecutionContext->mutex, &lastStreamerCallbackOutput = legacyExecutionContext->lastStreamerCallbackOutput](std::string text) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Streamer callback executed with text: [{}]", text);
            {
                std::lock_guard<std::mutex> lock(mutex);
                lastStreamerCallbackOutput += text;
                executionInProgress.notify_one();
            }
            return ov::genai::StreamingStatus::RUNNING;
        };
        legacyExecutionContext->textStreamer = std::make_shared<ov::genai::TextStreamer>(getProperties()->tokenizer, callback);
    }
    return absl::OkStatus();
}

absl::Status LegacyServable::prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    // Use the base class implementation to prepare inputs
    auto status = GenAiServable::prepareInputs(executionContext);
    if (!status.ok()) {
        return status;
    }
    // Additional validation layer for NPU specific properties
    status = validateInputComplianceWithProperties(executionContext->inputIds);
    return status;
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
    if (!legacyExecutionContext->success) {
        return absl::InvalidArgumentError("Request processing failed, check its correctness.");
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

absl::Status LegacyServable::readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    return absl::OkStatus();
}

absl::Status LegacyServable::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    std::string lastTextChunk;
    auto generationStatus = legacyExecutionContext->finished.wait_for(std::chrono::nanoseconds::zero());
    {
        std::unique_lock lock(legacyExecutionContext->mutex);
        while (executionContext->lastStreamerCallbackOutput.size() == 0 && generationStatus != std::future_status::ready) {
            SPDLOG_LOGGER_TRACE(llm_executor_logger, "Waiting for partial data...");
            legacyExecutionContext->executionInProgress.wait(lock);
            generationStatus = legacyExecutionContext->finished.wait_for(std::chrono::nanoseconds::zero());
        }
        lastTextChunk = executionContext->lastStreamerCallbackOutput;
        if (!lastTextChunk.empty()) {
            auto tokensTensor = properties->tokenizer.encode(lastTextChunk, ov::genai::add_special_tokens(false)).input_ids;
            auto numTokens = tokensTensor.get_size();
            executionContext->apiHandler->incrementProcessedTokens(numTokens);
        }
        executionContext->lastStreamerCallbackOutput = "";
    }
    if (generationStatus != std::future_status::ready) {  // continue
        if (lastTextChunk.size() > 0) {
            executionContext->response = wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingChunk(lastTextChunk, ov::genai::GenerationFinishReason::NONE));
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", executionContext->response);
        }
        executionContext->sendLoopbackSignal = true;
    } else {  // finish generation
        if (!legacyExecutionContext->success) {
            return absl::InvalidArgumentError("Request processing failed, check its correctness.");
        }
        OVMS_PROFILE_SCOPE("Generation of last streaming response");
        executionContext->textStreamer->end();
        // if streamer::put returned a value, streamer::end() result will not contain it, so we add it manually
        if (!executionContext->lastStreamerCallbackOutput.empty())
            lastTextChunk = lastTextChunk + executionContext->lastStreamerCallbackOutput;
        executionContext->response = wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingChunk(lastTextChunk, ov::genai::GenerationFinishReason::STOP));
        if (executionContext->apiHandler->getStreamOptions().includeUsage)
            executionContext->response += wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingUsageChunk());

        executionContext->response += wrapTextInServerSideEventMessage("[DONE]");

        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", executionContext->response);
        executionContext->sendLoopbackSignal = false;
        return absl::OkStatus();
    }
    return absl::OkStatus();
}

}  // namespace ovms

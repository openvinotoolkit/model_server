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

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../http_payload.hpp"
#include "../logging.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "../profiler.hpp"
#include "apis/openai_completions.hpp"
#include "servable.hpp"
#include "text_processor.hpp"

namespace ovms {
absl::Status GenAiServable::loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const ovms::HttpPayload& payload) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", payload.body);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
    if (payload.uri == "/v3/chat/completions" || payload.uri == "/v3/v1/chat/completions") {
        executionContext->endpoint = Endpoint::CHAT_COMPLETIONS;
    } else if (payload.uri == "/v3/completions" || payload.uri == "/v3/v1/completions") {
        executionContext->endpoint = Endpoint::COMPLETIONS;
    } else {
        return absl::InvalidArgumentError("Wrong endpoint. Allowed endpoints: /v3/chat/completions, /v3/completions");
    }
    executionContext->payload = payload;
    return absl::OkStatus();
}

absl::Status GenAiServable::parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    executionContext->apiHandler = std::make_shared<OpenAIChatCompletionsHandler>(*executionContext->payload.parsedJson,
        executionContext->endpoint,
        std::chrono::system_clock::now(),
        getProperties()->tokenizer);

    auto status = executionContext->apiHandler->parseRequest(getProperties()->maxTokensLimit, getProperties()->bestOfLimit, getProperties()->maxModelLength);
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to parse request: {}", status.message());
        return status;
    }

    if (executionContext->apiHandler->isStream()) {
        executionContext->lastStreamerCallbackOutput = "";  // initialize with empty string
        auto callback = [& lastStreamerCallbackOutput = executionContext->lastStreamerCallbackOutput](std::string text) {
            SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Streamer callback executed with text: [{}]", text);
            lastStreamerCallbackOutput = text;
            return ov::genai::StreamingStatus::RUNNING;
        };

        executionContext->textStreamer = std::make_shared<ov::genai::TextStreamer>(getProperties()->tokenizer, callback);
    }
    return absl::OkStatus();
}

absl::Status GenAiServable::prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    if (executionContext->apiHandler == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "API handler is not initialized");
    }

    // Base servable cannot process images
    if (executionContext->apiHandler->getImageHistory().size() > 0) {
        return absl::InternalError("This servable supports only text input, but image_url has been provided");
    }

    std::string inputText;
    switch (executionContext->endpoint) {
    case Endpoint::CHAT_COMPLETIONS: {
        bool success;
        if (executionContext->apiHandler->getProcessedJson().size() > 0) {
            success = TextProcessor::applyChatTemplate(getProperties()->textProcessor, getProperties()->modelsPath, executionContext->apiHandler->getProcessedJson(), inputText);
        } else {
            success = TextProcessor::applyChatTemplate(getProperties()->textProcessor, getProperties()->modelsPath, executionContext->payload.body, inputText);
        }
        if (!success) {
            return absl::Status(absl::StatusCode::kInvalidArgument, inputText);
        }
        if (inputText.size() == 0) {
            return absl::Status(absl::StatusCode::kInvalidArgument, "Final prompt after applying chat template is empty");
        }
        break;
    }
    case Endpoint::COMPLETIONS: {
        inputText = executionContext->apiHandler->getPrompt().value();
    }
    }

    bool encodeAddSpecialTokens = (executionContext->endpoint == Endpoint::COMPLETIONS);
    executionContext->inputIds = getProperties()->tokenizer.encode(inputText, ov::genai::add_special_tokens(encodeAddSpecialTokens)).input_ids;
    if (getProperties()->maxModelLength.has_value()) {
        if (executionContext->inputIds.get_size() > getProperties()->maxModelLength.value()) {
            std::stringstream ss;
            ss << "Number of prompt tokens: " << executionContext->inputIds.get_size() << " exceeds model max length: " << getProperties()->maxModelLength.value();
            SPDLOG_LOGGER_ERROR(llm_calculator_logger, ss.str());
            return absl::Status(absl::StatusCode::kInvalidArgument, ss.str());
        }
        if (executionContext->apiHandler->getMaxTokens().has_value() && executionContext->inputIds.get_size() + executionContext->apiHandler->getMaxTokens().value() > getProperties()->maxModelLength.value()) {
            std::stringstream ss;
            ss << "Number of prompt tokens: " << executionContext->inputIds.get_size() << " + max tokens value: " << executionContext->apiHandler->getMaxTokens().value() << " exceeds model max length: " << getProperties()->maxModelLength.value();
            SPDLOG_LOGGER_ERROR(llm_calculator_logger, ss.str());
            return absl::Status(absl::StatusCode::kInvalidArgument, ss.str());
        }
    }

    executionContext->apiHandler->setPromptTokensUsage(executionContext->inputIds.get_size());
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "{}", getPromptTokensString(executionContext->inputIds));

    return absl::OkStatus();
}

absl::Status GenAiServable::prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    executionContext->response = executionContext->apiHandler->serializeUnaryResponse(executionContext->generationOutputs);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", executionContext->response);
    return absl::OkStatus();
}

absl::Status GenAiServable::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    if (executionContext->generationOutputs.size() != 1) {
        return absl::InternalError("For streaming we expect exactly one generation output");
    }
    auto& generationOutput = executionContext->generationOutputs[0];
    executionContext->apiHandler->incrementProcessedTokens(generationOutput.generated_ids.size());

    std::stringstream ss;
    executionContext->textStreamer->write(generationOutput.generated_ids);
    ss << executionContext->lastStreamerCallbackOutput;
    // OpenVINO GenAI TextStreamer dose not trigger callback if text is empty: https://github.com/openvinotoolkit/openvino.genai/blob/434c2a9494fb1ee83ca7a36fe8315cfc2691c232/src/cpp/src/text_streamer.cpp#L102-L108
    // Reset lastStreamerCallbackOutput as "" to avoid repeated sending previous text if lastStreamerCallbackOutput not updated by callback
    executionContext->lastStreamerCallbackOutput = "";

    std::string lastTextChunk = ss.str();
    ov::genai::GenerationFinishReason finishReason = generationOutput.finish_reason;
    if (finishReason == ov::genai::GenerationFinishReason::NONE) {  // continue
        if (lastTextChunk.size() > 0) {
            executionContext->response = wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingChunk(lastTextChunk, finishReason));
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", executionContext->response);
        }
        executionContext->sendLoopbackSignal = true;
    } else {  // finish generation
        OVMS_PROFILE_SCOPE("Generation of last streaming response");
        executionContext->textStreamer->end();
        // if streamer::put returned a value, streamer::end() result will not contain it, so we add it manually
        if (!executionContext->lastStreamerCallbackOutput.empty())
            lastTextChunk = lastTextChunk + executionContext->lastStreamerCallbackOutput;
        executionContext->response = wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingChunk(lastTextChunk, finishReason));
        if (executionContext->apiHandler->getStreamOptions().includeUsage)
            executionContext->response += wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingUsageChunk());

        executionContext->response += wrapTextInServerSideEventMessage("[DONE]");

        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", executionContext->response);
        executionContext->sendLoopbackSignal = false;
    }
    return absl::OkStatus();
}

#pragma warning(push)
#pragma warning(disable : 4505)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function";
std::string wrapTextInServerSideEventMessage(const std::string& text) {
    std::stringstream ss;
    ss << "data: " << text << "\n\n";
    return ss.str();
}
#pragma GCC diagnostic pop
#pragma warning(push)

}  // namespace ovms

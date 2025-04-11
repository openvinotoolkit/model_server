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
#include <unordered_map>
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

absl::Status VisualLanguageModelLegacyServable::loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const ovms::HttpPayload& payload) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", payload.body);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
    if (payload.uri == "/v3/chat/completions" || payload.uri == "/v3/v1/chat/completions") {
        executionContext->endpoint = Endpoint::CHAT_COMPLETIONS;
    } else {
        return absl::InvalidArgumentError("Wrong endpoint. VLM Servable allowed only on /v3/chat/completions endpoint");
    }
    executionContext->payload = payload;
    return absl::OkStatus();
}

// Node resources interface start
std::shared_ptr<GenAiServableExecutionContext> VisualLanguageModelLegacyServable::createExecutionContext() {
    return std::make_shared<VisualLanguageModelLegacyServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> VisualLanguageModelLegacyServable::getProperties() {
    return properties;
}

bool VisualLanguageModelLegacyServable::supportsSpeculativeDecoding() const {
    return false;
}

absl::Status VisualLanguageModelLegacyServable::parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<VisualLanguageModelLegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    legacyExecutionContext->apiHandler = std::make_shared<OpenAIChatCompletionsHandler>(*legacyExecutionContext->payload.parsedJson,
        legacyExecutionContext->endpoint,
        std::chrono::system_clock::now(),
        getProperties()->tokenizer);

    auto status = executionContext->apiHandler->parseRequest(getProperties()->maxTokensLimit, getProperties()->bestOfLimit, getProperties()->isSpeculativePipeline,
        getProperties()->isPromptLookupPipeline, getProperties()->maxModelLength);
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

absl::Status VisualLanguageModelLegacyServable::scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<VisualLanguageModelLegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    properties->legacyExecutor->addRequest(legacyExecutionContext);
    return absl::OkStatus();
}

absl::Status VisualLanguageModelLegacyServable::readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<VisualLanguageModelLegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    legacyExecutionContext->finished.wait();
    if (!legacyExecutionContext->success) {
        return absl::InvalidArgumentError("Request processing failed, check its correctness.");
    }
    return absl::OkStatus();
}

absl::Status VisualLanguageModelLegacyServable::prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<VisualLanguageModelLegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    size_t completionTokens = 0;
    for (std::string text : legacyExecutionContext->results.texts) {
        auto tokensTensor = properties->tokenizer.encode(text, ov::genai::add_special_tokens(false)).input_ids;
        completionTokens += tokensTensor.get_size();
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated tokens number: {}", completionTokens);
    executionContext->response = executionContext->apiHandler->serializeUnaryResponse(legacyExecutionContext->results, completionTokens);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", executionContext->response);
    return absl::OkStatus();
}

absl::Status VisualLanguageModelLegacyServable::readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    return absl::OkStatus();
}

absl::Status VisualLanguageModelLegacyServable::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<VisualLanguageModelLegacyServableExecutionContext>(executionContext);
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
        executionContext->lastStreamerCallbackOutput = "";
    }
    if (!lastTextChunk.empty()) {
        auto tokensTensor = properties->tokenizer.encode(lastTextChunk, ov::genai::add_special_tokens(false)).input_ids;
        auto numTokens = tokensTensor.get_size();
        executionContext->apiHandler->incrementProcessedTokens(numTokens);
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

        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", lastTextChunk);
        executionContext->sendLoopbackSignal = false;
        return absl::OkStatus();
    }
    return absl::OkStatus();
}

absl::Status VisualLanguageModelLegacyServable::prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto vlmExecutionContext = std::static_pointer_cast<VisualLanguageModelLegacyServableExecutionContext>(executionContext);
    if (vlmExecutionContext->apiHandler == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "API handler is not initialized");
    }
    if (executionContext->endpoint == Endpoint::CHAT_COMPLETIONS) {
        ov::genai::ChatHistory& chatHistory = vlmExecutionContext->apiHandler->getChatHistory();

        // Validate chat history for restricted tags
        for (const auto& historyEntry : chatHistory) {
            for (const auto& [_, content] : historyEntry) {
                if (content.find("<ov_genai_image_") != std::string::npos) {
                    return absl::InvalidArgumentError("Message contains restricted <ov_genai_image> tag");
                }
            }
        }

        const ImageHistory& imageHistory = vlmExecutionContext->apiHandler->getImageHistory();
        size_t imageIndex = 0;
        std::unordered_map<size_t, std::string> imageTags;
        for (const auto& image : imageHistory) {
            const auto& [chatTurnIndex, imageTensor] = image;
            std::string imageTag = "<ov_genai_image_" + std::to_string(imageIndex++) + ">\n";
            imageTags[chatTurnIndex] = imageTags[chatTurnIndex] + imageTag;
            vlmExecutionContext->inputImages.push_back(imageTensor);
        }
        for (const auto& [chatTurnIndex, imageTagString] : imageTags) {
            chatHistory[chatTurnIndex]["content"] = imageTagString + chatHistory[chatTurnIndex]["content"];
        }
        constexpr bool add_generation_prompt = true;  // confirm it should be hardcoded
        vlmExecutionContext->inputText = properties->tokenizer.apply_chat_template(chatHistory, add_generation_prompt);
    } else {
        return absl::InvalidArgumentError("Unsupported endpoint");
    }

    // Below logic is used only for the statistics and debugging purposes and does not affect the model execution.
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "VLM input text: {}", vlmExecutionContext->inputText);
    bool encodeAddSpecialTokens = false;  // assuming chat template application added special tokens
    ov::Tensor inputTextIds = getProperties()->tokenizer.encode(vlmExecutionContext->inputText, ov::genai::add_special_tokens(encodeAddSpecialTokens)).input_ids;
    vlmExecutionContext->apiHandler->setPromptTokensUsage(inputTextIds.get_size());
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "{}", getPromptTokensString(inputTextIds));

    return absl::OkStatus();
}

}  // namespace ovms

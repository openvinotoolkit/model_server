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
#include <utility>
#include <vector>

#include "../../../logging.hpp"
#include "../../../profiler.hpp"
#include "../../../status.hpp"
#include "../../apis/openai_completions.hpp"
#include "../../apis/openai_responses.hpp"
#include "../../ovms_text_streamer.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../../http_payload.hpp"
#include "../../../mediapipe_internal/mediapipe_utils.hpp"
#include "../../text_utils.hpp"
#if (PYTHON_DISABLE == 0)
#include "../../py_jinja_template_processor.hpp"
#endif
#include "../../io_processing/generation_config_builder.hpp"
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

    legacyExecutionContext->baseGenerationConfig = properties->baseGenerationConfig;
    try {
        if (legacyExecutionContext->endpoint == Endpoint::RESPONSES) {
            legacyExecutionContext->apiHandler = std::make_shared<OpenAIResponsesHandler>(*legacyExecutionContext->payload.parsedJson,
                legacyExecutionContext->endpoint,
                std::chrono::system_clock::now(),
                getProperties()->tokenizer,
                getProperties()->toolParserName,
                getProperties()->reasoningParserName);
        } else {
            legacyExecutionContext->apiHandler = std::make_shared<OpenAIChatCompletionsHandler>(*legacyExecutionContext->payload.parsedJson,
                legacyExecutionContext->endpoint,
                std::chrono::system_clock::now(),
                getProperties()->tokenizer,
                getProperties()->toolParserName,
                getProperties()->reasoningParserName);
        }
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to create API handler: {}", e.what());
        return absl::InvalidArgumentError(std::string("Failed to create API handler: ") + e.what());
    }

    auto status = executionContext->apiHandler->parseRequest(getProperties()->maxTokensLimit, getProperties()->bestOfLimit, getProperties()->maxModelLength);
    if (!status.ok()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to parse request: {}", status.message());
        return status;
    }

    ov::AnyMap streamerConfig;
    {
        if (!legacyExecutionContext->apiHandler->getRequest().skipSpecialTokens) {
            streamerConfig.insert(ov::genai::skip_special_tokens(false));
        }
        auto ovmsCallback = [& ctx = *legacyExecutionContext](rapidjson::Document delta, bool isLast) -> ov::genai::StreamingStatus {
            if (ctx.clientDisconnected.load()) {
                ctx.deltaChannel.signalComplete();
                return ov::genai::StreamingStatus::CANCEL;
            }
            ctx.deltaChannel.push(std::move(delta), isLast);
            return ov::genai::StreamingStatus::RUNNING;
        };
        legacyExecutionContext->textStreamer = std::make_shared<OVMSTextStreamer>(
            getProperties()->tokenizer,
            legacyExecutionContext->apiHandler->getOutputParser(),
            legacyExecutionContext->apiHandler->areToolsAvailable(),
            std::move(ovmsCallback),
            streamerConfig);
    }
    GenerationConfigBuilder configBuilder(getProperties()->baseGenerationConfig,
        getProperties()->toolParserName,
        getProperties()->enableToolGuidedGeneration,
        getProperties()->decodingMethod);
    auto inputRequestResult = legacyExecutionContext->apiHandler->extractInputRequest(configBuilder);
    if (!inputRequestResult.ok()) {
        return inputRequestResult.status();
    }
    legacyExecutionContext->inputRequest = std::move(*inputRequestResult);
    return absl::OkStatus();
}

absl::Status LegacyServable::prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    // Use the base class implementation to prepare inputs
    auto status = GenAiServable::prepareInputs(executionContext);
    if (!status.ok()) {
        return status;
    }
    // Additional validation layer for NPU specific properties
    status = validateInputComplianceWithProperties(executionContext->inputRequest.inputIds);
    return status;
}

absl::Status LegacyServable::scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<LegacyServableExecutionContext>(executionContext);
    std::weak_ptr<LegacyServableExecutionContext> weakContext = legacyExecutionContext;
    legacyExecutionContext->payload.client->registerDisconnectionCallback([weakContext]() {
        if (auto context = weakContext.lock()) {
            context->signalDisconnection();
        }
    });
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        legacyExecutionContext->signalDisconnection();
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
    // By the time prepareCompleteResponse is called, readCompleteExecutionResults has
    // already waited on finished — results and perf_metrics are fully populated.
    executionContext->apiHandler->setPromptTokensUsage(
        legacyExecutionContext->results.perf_metrics.get_num_input_tokens());
    executionContext->apiHandler->setCompletionTokensUsage(
        legacyExecutionContext->results.perf_metrics.get_num_generated_tokens());

    if (legacyExecutionContext->results.finish_reasons.empty()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Missing finish reason in legacy LLM unary generation result, defaulting to STOP");
    }
    const ov::genai::GenerationFinishReason finishReason =
        legacyExecutionContext->results.finish_reasons.empty() ? ov::genai::GenerationFinishReason::STOP : legacyExecutionContext->results.finish_reasons[0];

    if (executionContext->apiHandler->isVerboseResponse() && !legacyExecutionContext->results.tokens.empty()) {
        executionContext->apiHandler->appendVerboseRawTokens(legacyExecutionContext->results.tokens[0]);
    }

    std::vector<rapidjson::Document> deltas = executionContext->deltaChannel.drain();
    executionContext->response = executionContext->apiHandler->serializeUnaryResponse(deltas, finishReason);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", executionContext->response);
    return absl::OkStatus();
}

absl::Status LegacyServable::readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    executionContext->deltaChannel.waitForData();
    return absl::OkStatus();
}

}  // namespace ovms

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

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

#include "../../../config.hpp"
#include "../../../http_payload.hpp"
#include "../../../mediapipe_internal/mediapipe_utils.hpp"
#include "../../text_utils.hpp"
#include "../../../tokenize/tokenize_parser.hpp"
#if (PYTHON_DISABLE == 0)
#include "../../py_jinja_template_processor.hpp"
#endif
#include "../../io_processing/generation_config_builder.hpp"
#include "servable.hpp"

namespace ovms {

absl::Status VisualLanguageModelLegacyServable::loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const ovms::HttpPayload& payload) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", payload.body);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
    // Parsed JSON is not guaranteed to be valid, we may reach this point via multipart content type request with no valid JSON parser
    if (payload.parsedJson->HasParseError()) {
        return absl::InvalidArgumentError("Non-json request received in text generation calculator");
    }
    if (payload.uri == "/v3/chat/completions" || payload.uri == "/v3/v1/chat/completions") {
        executionContext->endpoint = Endpoint::CHAT_COMPLETIONS;
    } else if (payload.uri == "/v3/responses" || payload.uri == "/v3/v1/responses") {
        executionContext->endpoint = Endpoint::RESPONSES;
    } else if (TokenizeParser::isTokenizeEndpoint(payload.uri)) {
        executionContext->endpoint = Endpoint::TOKENIZE;
    } else {
        return absl::InvalidArgumentError("Wrong endpoint. VLM Servable allowed only on /v3/chat/completions, /v3/responses endpoint or /v3/tokenize");
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

absl::Status VisualLanguageModelLegacyServable::parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<VisualLanguageModelLegacyServableExecutionContext>(executionContext);
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
    auto& config = ovms::Config::instance();

    auto status = executionContext->apiHandler->parseRequest(getProperties()->maxTokensLimit, getProperties()->bestOfLimit, getProperties()->maxModelLength, config.getServerSettings().allowedLocalMediaPath, config.getServerSettings().allowedMediaDomains);
    if (!status.ok()) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to parse request: {}", status.message());
        return status;
    }

    ov::AnyMap streamerConfig;
    if (legacyExecutionContext->apiHandler->isStream()) {
        if ((legacyExecutionContext->apiHandler->getOutputParser() != nullptr &&
                legacyExecutionContext->apiHandler->getOutputParser()->requiresStreamingWithSpecialTokens()) ||
            !legacyExecutionContext->apiHandler->getRequest().skipSpecialTokens) {
            streamerConfig.insert(ov::genai::skip_special_tokens(false));
        }
        auto ovmsCallback = [& ctx = *legacyExecutionContext](rapidjson::Document delta) -> ov::genai::StreamingStatus {
            if (ctx.clientDisconnected.load()) {
                ctx.deltaChannel.signalComplete();
                return ov::genai::StreamingStatus::CANCEL;
            }
            ctx.deltaChannel.push(std::move(delta));
            return ov::genai::StreamingStatus::RUNNING;
        };
        legacyExecutionContext->textStreamer = std::make_shared<OVMSTextStreamer>(
            getProperties()->tokenizer,
            legacyExecutionContext->apiHandler->getOutputParser(),
            legacyExecutionContext->apiHandler->areToolsAvailable(),
            std::move(ovmsCallback),
            streamerConfig);
    } else {
        // For the unary path we still need OVMSTextStreamer so that the tokenizer
        // decode params (e.g. skip_special_tokens) from the request are applied.
        // results.texts[0] is decoded by the VLM pipeline with its own hardcoded
        // config — using the streamer callback is the only way to respect the user's
        // setting here.
        //
        // Crucially, we pass nullptr as the output parser: serializeUnaryResponse
        // feeds accumulatedUnaryText back through encodeTextToTokens() and the
        // batch parser (parseOutputIfNeeded), which expects raw decoded text with
        // structural tags intact (e.g. <think>, <tool_call>). Passing a non-null
        // parser here would strip those tags via parseChunk before accumulation
        // and break the downstream unary parsing of reasoning/tool_calls.
        // Will be further reworked in next refactor phases.
        if ((legacyExecutionContext->apiHandler->getOutputParser() != nullptr &&
                legacyExecutionContext->apiHandler->getOutputParser()->requiresStreamingWithSpecialTokens()) ||
            !legacyExecutionContext->apiHandler->getRequest().skipSpecialTokens) {
            streamerConfig.insert(ov::genai::skip_special_tokens(false));
        }
        auto unaryCallback = [& ctx = *legacyExecutionContext](rapidjson::Document delta) -> ov::genai::StreamingStatus {
            if (delta.HasMember("delta") && delta["delta"].IsObject() &&
                delta["delta"].HasMember("content") && delta["delta"]["content"].IsString()) {
                ctx.accumulatedUnaryText += delta["delta"]["content"].GetString();
            }
            return ov::genai::StreamingStatus::RUNNING;
        };
        legacyExecutionContext->textStreamer = std::make_shared<OVMSTextStreamer>(
            getProperties()->tokenizer,
            nullptr,  // no parser: accumulate raw decoded text for batch unary parsing
            false,
            std::move(unaryCallback),
            streamerConfig);
    }
    GenerationConfigBuilder configBuilder(getProperties()->baseGenerationConfig,
        getProperties()->toolParserName,
        getProperties()->enableToolGuidedGeneration,
        getProperties()->decodingMethod);
    legacyExecutionContext->inputRequest = legacyExecutionContext->apiHandler->extractInputRequest(configBuilder);
    return absl::OkStatus();
}

absl::Status VisualLanguageModelLegacyServable::scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<VisualLanguageModelLegacyServableExecutionContext>(executionContext);
    std::weak_ptr<VisualLanguageModelLegacyServableExecutionContext> weakContext = legacyExecutionContext;
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

    // pipe->generate() called streamer->end() before returning, so accumulatedUnaryText is
    // already fully populated by the callbacks fired from OVMSTextStreamer::write()/end().
    const std::string& completeText = legacyExecutionContext->accumulatedUnaryText;
    executionContext->response = executionContext->apiHandler->serializeUnaryResponse(
        legacyExecutionContext->results, completeText);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", executionContext->response);
    return absl::OkStatus();
}

absl::Status VisualLanguageModelLegacyServable::readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    executionContext->deltaChannel.waitForData();
    return absl::OkStatus();
}

absl::Status VisualLanguageModelLegacyServable::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyExecutionContext = std::static_pointer_cast<VisualLanguageModelLegacyServableExecutionContext>(executionContext);
    if (legacyExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    std::vector<rapidjson::Document> deltas = executionContext->deltaChannel.drain();
    const bool isFinishing = executionContext->deltaChannel.complete();
    if (!isFinishing) {
        // For RESPONSES endpoint, always call serializeStreamingChunk so that
        // output item initialization events are emitted even before the tokenizer produces text.
        if (deltas.size() > 0 || executionContext->apiHandler->getEndpoint() == Endpoint::RESPONSES) {
            for (auto& delta : deltas) {
                if (executionContext->apiHandler->isVerboseResponse() &&
                    delta.HasMember("delta") && delta["delta"].IsObject() &&
                    delta["delta"].HasMember("content") && delta["delta"]["content"].IsString()) {
                    executionContext->apiHandler->appendVerboseRawText(delta["delta"]["content"].GetString());
                }
                std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                    std::move(delta), ov::genai::GenerationFinishReason::NONE);
                if (!serialized.empty()) {
                    executionContext->response += wrapTextInServerSideEventMessage(serialized);
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", serialized);
                }
            }
            if (deltas.empty()) {
                // No delta generated yet — emit lifecycle events for RESPONSES endpoint.
                if (!executionContext->lifecyclePrimed) {
                    std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                        rapidjson::Document{}, ov::genai::GenerationFinishReason::NONE);
                    if (!serialized.empty()) {
                        executionContext->response = wrapTextInServerSideEventMessage(serialized);
                        executionContext->lifecyclePrimed = true;
                    }
                }
            }
        }
        executionContext->sendLoopbackSignal = true;
    } else {
        if (!legacyExecutionContext->success) {
            return absl::InvalidArgumentError("Request processing failed, check its correctness.");
        }
        OVMS_PROFILE_SCOPE("Generation of last streaming response");
        // end() was already called by pipe->generate() internally; all deltas are
        // already in deltaChannel before signalComplete() fired. Drain any remaining.
        for (auto& d : executionContext->deltaChannel.drain()) {
            deltas.push_back(std::move(d));
        }
        if (legacyExecutionContext->results.finish_reasons.empty()) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Missing finish reason in legacy VLM streaming generation result, defaulting to STOP");
        }
        // Legacy generation path always runs with deltas=1, so we read the single finish reason at index 0.
        ov::genai::GenerationFinishReason finishReason = legacyExecutionContext->results.finish_reasons.empty() ? ov::genai::GenerationFinishReason::STOP : legacyExecutionContext->results.finish_reasons[0];
        executionContext->apiHandler->setPromptTokensUsage(legacyExecutionContext->results.perf_metrics.get_num_input_tokens());
        executionContext->apiHandler->setCompletionTokensUsage(legacyExecutionContext->results.perf_metrics.get_num_generated_tokens());
        if (!deltas.empty()) {
            for (size_t i = 0; i < deltas.size(); ++i) {
                const bool isLast = (i == deltas.size() - 1);
                if (executionContext->apiHandler->isVerboseResponse() &&
                    deltas[i].HasMember("delta") && deltas[i]["delta"].IsObject() &&
                    deltas[i]["delta"].HasMember("content") && deltas[i]["delta"]["content"].IsString()) {
                    executionContext->apiHandler->appendVerboseRawText(deltas[i]["delta"]["content"].GetString());
                }
                std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                    std::move(deltas[i]),
                    isLast ? finishReason : ov::genai::GenerationFinishReason::NONE);
                if (!serialized.empty()) {
                    executionContext->response += wrapTextInServerSideEventMessage(serialized);
                }
            }
        } else {
            // Parser produced no delta (generation ended on a swallowed token).
            std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                rapidjson::Document{}, finishReason);
            if (!serialized.empty()) {
                executionContext->response += wrapTextInServerSideEventMessage(serialized);
            }
        }
        if (executionContext->apiHandler->getStreamOptions().includeUsage)
            executionContext->response += wrapTextInServerSideEventMessage(executionContext->apiHandler->serializeStreamingUsageChunk());
        executionContext->response += wrapTextInServerSideEventMessage("[DONE]");
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", executionContext->response);
        executionContext->sendLoopbackSignal = false;
    }
    return absl::OkStatus();
}

}  // namespace ovms

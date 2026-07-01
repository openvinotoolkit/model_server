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

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005 4456 6246 6313)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../config.hpp"
#include "../http_payload.hpp"
#include "../logging.hpp"
#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "../profiler.hpp"
#include "apis/openai_completions.hpp"
#include "apis/openai_responses.hpp"
#include "io_processing/generation_config_builder.hpp"
#include "io_processing/input_processor.hpp"
#include "ovms_text_streamer.hpp"
#include "servable.hpp"
#include "text_utils.hpp"
#include "../tokenize/tokenize_parser.hpp"

namespace ovms {

void GenAiServable::determineDecodingMethod() {
    getProperties()->decodingMethod = DecodingMethod::STANDARD;
    auto& pluginConfig = getProperties()->pluginConfig;
    if (pluginConfig.find("draft_model") != pluginConfig.end()) {
        if (getProperties()->eagle3Mode) {
            getProperties()->decodingMethod = DecodingMethod::EAGLE3;
        } else {
            getProperties()->decodingMethod = DecodingMethod::SPECULATIVE_DECODING;
        }
    }
    auto it = pluginConfig.find("prompt_lookup");
    if (it != pluginConfig.end() && it->second.as<bool>() == true) {
        getProperties()->decodingMethod = DecodingMethod::PROMPT_LOOKUP;
    }
}

absl::Status GenAiServable::loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const ovms::HttpPayload& payload) {
    if (spdlog::default_logger_raw()->level() <= spdlog::level::debug) {
        logRequestDetails(payload);
    }
    // Parsed JSON is not guaranteed to be valid, we may reach this point via multipart content type request with no valid JSON parser
    if (payload.parsedJson->HasParseError()) {
        return absl::InvalidArgumentError("Non-json request received in text generation calculator");
    }
    if (payload.uri == "/v3/chat/completions" || payload.uri == "/v3/v1/chat/completions") {
        executionContext->endpoint = Endpoint::CHAT_COMPLETIONS;
    } else if (payload.uri == "/v3/completions" || payload.uri == "/v3/v1/completions") {
        executionContext->endpoint = Endpoint::COMPLETIONS;
    } else if (payload.uri == "/v3/responses" || payload.uri == "/v3/v1/responses") {
        executionContext->endpoint = Endpoint::RESPONSES;
    } else if (TokenizeParser::isTokenizeEndpoint(payload.uri)) {
        executionContext->endpoint = Endpoint::TOKENIZE;
    } else {
        return absl::InvalidArgumentError("Wrong endpoint. Allowed endpoints: /v3/chat/completions, /v3/completions, /v3/responses, /v3/tokenize");
    }
    executionContext->payload = payload;
    return absl::OkStatus();
}

absl::Status GenAiServable::processTokenizeRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    ovms::TokenizeRequest tokenizeRequest;
    auto status = ovms::TokenizeParser::parseTokenizeRequest(*executionContext->payload.parsedJson, tokenizeRequest);
    if (status != absl::OkStatus()) {
        return status;
    }

    ov::genai::TokenizedInputs tokens;

    if (auto strings = std::get_if<std::vector<std::string>>(&tokenizeRequest.input)) {
        tokens = getProperties()->tokenizer.encode(*strings, tokenizeRequest.parameters);
        RET_CHECK(tokens.input_ids.get_shape().size() == 2);
    } else {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "LLM tokenize input is of not supported type");
        return absl::InvalidArgumentError("Input should be string or array of strings");
    }

    StringBuffer responseBuffer;
    auto responseStatus = ovms::TokenizeParser::parseTokenizeResponse(responseBuffer, tokens, tokenizeRequest.parameters);

    if (!responseStatus.ok()) {
        return responseStatus;
    }

    executionContext->response = responseBuffer.GetString();

    return absl::OkStatus();
}

absl::Status GenAiServable::parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    try {
        if (executionContext->endpoint == Endpoint::RESPONSES) {
            executionContext->apiHandler = std::make_shared<OpenAIResponsesHandler>(*executionContext->payload.parsedJson,
                executionContext->endpoint,
                std::chrono::system_clock::now(),
                getProperties()->tokenizer,
                getProperties()->toolParserName,
                getProperties()->reasoningParserName);
        } else {
            executionContext->apiHandler = std::make_shared<OpenAIChatCompletionsHandler>(*executionContext->payload.parsedJson,
                executionContext->endpoint,
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

    {
        auto ovmsCallback = [& ctx = *executionContext](rapidjson::Document delta, bool isLast) -> ov::genai::StreamingStatus {
            ctx.deltaChannel.push(std::move(delta), isLast);
            return ov::genai::StreamingStatus::RUNNING;
        };
        ov::AnyMap streamerConfig;
        if ((executionContext->apiHandler->getOutputParser() != nullptr &&
                executionContext->apiHandler->getOutputParser()->requiresStreamingWithSpecialTokens()) ||
            !executionContext->apiHandler->getRequest().skipSpecialTokens) {
            streamerConfig.insert(ov::genai::skip_special_tokens(false));
        }
        executionContext->textStreamer = std::make_shared<OVMSTextStreamer>(
            getProperties()->tokenizer,
            executionContext->apiHandler->getOutputParser(),
            executionContext->apiHandler->areToolsAvailable(),
            std::move(ovmsCallback),
            streamerConfig);
    }
    GenerationConfigBuilder configBuilder(getProperties()->baseGenerationConfig,
        getProperties()->toolParserName,
        getProperties()->enableToolGuidedGeneration,
        getProperties()->decodingMethod);
    auto inputRequestResult = executionContext->apiHandler->extractInputRequest(configBuilder);
    if (!inputRequestResult.ok()) {
        return inputRequestResult.status();
    }
    executionContext->inputRequest = std::move(*inputRequestResult);
    return absl::OkStatus();
}

absl::Status GenAiServable::validateInputCompatibility(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    // LM servables reject requests containing image content. Images are preserved
    // as JsonContainer arrays in chatHistory. Reject if any message's content array
    // contains an image_url entry.
    if (!getProperties()->inputProcessorContext.config.isVLM &&
        std::holds_alternative<ov::genai::ChatHistory>(executionContext->inputRequest.input)) {
        const auto& ch = std::get<ov::genai::ChatHistory>(executionContext->inputRequest.input);
        for (size_t i = 0; i < ch.size(); i++) {
            const auto content = ch[i]["content"];
            if (content.is_array()) {
                for (size_t j = 0; j < content.size(); j++) {
                    if (content[j]["type"].as_string().value_or("") == "image_url") {
                        return absl::Status(absl::StatusCode::kInvalidArgument, "This servable supports only text input, but image_url has been provided");
                    }
                }
            }
        }
    }
    return absl::OkStatus();
}

absl::Status GenAiServable::prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    if (executionContext->apiHandler == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "API handler is not initialized");
    }

    auto status = validateInputCompatibility(executionContext);
    if (!status.ok()) {
        return status;
    }

    InputRequest& req = executionContext->inputRequest;
    InputProcessor processor(getProperties()->inputProcessorContext, req);
    status = processor.process(req);
    if (!status.ok()) {
        return status;
    }

    if (executionContext->apiHandler->getOutputParser() != nullptr) {
        executionContext->apiHandler->getOutputParser()->detectAndSetImplicitReasoningStart(req.promptText);
    }
    if (Config::instance().getServerSettings().verboseResponse) {
        executionContext->apiHandler->enableVerboseResponse(req.promptText);
    }
    if (getProperties()->maxModelLength.has_value()) {
        if (req.inputIds.get_size() > getProperties()->maxModelLength.value()) {
            std::stringstream ss;
            ss << "Number of prompt tokens: " << req.inputIds.get_size()
               << " exceeds model max length: " << getProperties()->maxModelLength.value();
            SPDLOG_LOGGER_WARN(llm_calculator_logger, ss.str());
            return absl::Status(absl::StatusCode::kInvalidArgument, ss.str());
        }
        if (executionContext->apiHandler->getMaxTokens().has_value() &&
            req.inputIds.get_size() + static_cast<size_t>(executionContext->apiHandler->getMaxTokens().value()) >
                getProperties()->maxModelLength.value()) {
            std::stringstream ss;
            ss << "Number of prompt tokens: " << req.inputIds.get_size()
               << " + max tokens value: " << executionContext->apiHandler->getMaxTokens().value()
               << " exceeds model max length: " << getProperties()->maxModelLength.value();
            SPDLOG_LOGGER_WARN(llm_calculator_logger, ss.str());
            return absl::Status(absl::StatusCode::kInvalidArgument, ss.str());
        }
    }
    executionContext->apiHandler->setPromptTokensUsage(req.inputIds.get_size());
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "{}", getPromptTokensString(req.inputIds));
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Pipeline input text: {}", req.promptText);

    return absl::OkStatus();
}

absl::Status GenAiServable::prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    const bool hasLogprobs = executionContext->apiHandler->getRequest().logprobschat ||
                             executionContext->apiHandler->getRequest().logprobs;
    const size_t numOutputs = executionContext->generationOutputs.size();

    // Build streamer config once; shared across all per-sequence streamers.
    ov::AnyMap streamerConfig;
    if ((executionContext->apiHandler->getOutputParser() != nullptr &&
            executionContext->apiHandler->getOutputParser()->requiresStreamingWithSpecialTokens()) ||
        !executionContext->apiHandler->getRequest().skipSpecialTokens) {
        streamerConfig.insert(ov::genai::skip_special_tokens(false));
    }

    std::vector<std::vector<rapidjson::Document>> allDeltas;
    std::vector<ov::genai::GenerationFinishReason> finishReasons;
    std::vector<UnaryChoiceLogprobs> logprobData;
    allDeltas.reserve(numOutputs);
    finishReasons.reserve(numOutputs);

    for (size_t i = 0; i < numOutputs; ++i) {
        const auto& output = executionContext->generationOutputs[i];

        if (executionContext->apiHandler->isVerboseResponse()) {
            executionContext->apiHandler->appendVerboseRawTokens(output.generated_ids);
        }
        executionContext->apiHandler->incrementProcessedTokens(output.generated_ids.size());

        std::vector<rapidjson::Document> localDeltas;
        if (numOutputs == 1) {
            // Single sequence: reuse the OVMSTextStreamer and deltaChannel built in parseRequest.
            executionContext->textStreamer->write(output.generated_ids);
            executionContext->textStreamer->end();
            localDeltas = executionContext->deltaChannel.drain();
        } else {
            // Multiple sequences: each beam requires its own independent stateful streamer
            // (hold-back buffer, parser state are per-sequence).
            auto cb = [&localDeltas](rapidjson::Document delta, bool) -> ov::genai::StreamingStatus {
                localDeltas.push_back(std::move(delta));
                return ov::genai::StreamingStatus::RUNNING;
            };
            auto tempStreamer = std::make_shared<OVMSTextStreamer>(
                getProperties()->tokenizer,
                executionContext->apiHandler->getOutputParser(),
                executionContext->apiHandler->areToolsAvailable(),
                std::move(cb),
                streamerConfig);
            tempStreamer->write(output.generated_ids);
            tempStreamer->end();
        }

        allDeltas.push_back(std::move(localDeltas));
        finishReasons.push_back(output.finish_reason);
        if (hasLogprobs) {
            logprobData.push_back({output.generated_ids, output.generated_log_probs});
        }
    }

    if (hasLogprobs) {
        executionContext->response = executionContext->apiHandler->serializeUnaryResponse(
            allDeltas, finishReasons, logprobData);
    } else {
        executionContext->response = executionContext->apiHandler->serializeUnaryResponse(
            allDeltas, finishReasons);
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", executionContext->response);
    return absl::OkStatus();
}

absl::Status GenAiServable::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    if (executionContext->generationOutputs.size() != 1) {
        return absl::InternalError("For streaming we expect exactly one generation output");
    }
    auto& generationOutput = executionContext->generationOutputs[0];
    executionContext->apiHandler->incrementProcessedTokens(generationOutput.generated_ids.size());
    if (executionContext->apiHandler->isVerboseResponse()) {
        executionContext->apiHandler->appendVerboseRawTokens(generationOutput.generated_ids);
    }

    bool isFirstToken = GenerationPhase::INPUT_TOKEN_PROCESSING == executionContext->generationPhase;
    if (isFirstToken) {
        executionContext->generationPhase = GenerationPhase::OUTPUT_TOKEN_PROCESSING;
    }

    ov::genai::GenerationFinishReason finishReason = generationOutput.finish_reason;
    const bool isFinishing = (finishReason != ov::genai::GenerationFinishReason::NONE);

    // OVMSTextStreamer::write() fires the callback for each flush event, pushing
    // Documents into executionContext->deltaChannel.
    executionContext->textStreamer->write(generationOutput.generated_ids);

    if (isFinishing) {
        OVMS_PROFILE_SCOPE("Generation of last streaming response");
        // end() flushes held-back tokens and calls parseChunk(STOP). Any resulting
        // Document is pushed into deltaChannel by the callback.
        executionContext->textStreamer->end();
    }

    // Drain all deltas accumulated during this write()/end() cycle.
    std::vector<rapidjson::Document> deltas = executionContext->deltaChannel.drain();
    const size_t count = deltas.size();

    if (!isFinishing) {
        // For RESPONSES endpoint, always call serializeStreamingChunk so lifecycle
        // events (output_item.added, content_part.added) are emitted on the first
        // call, even before the tokenizer produces text.
        if (count > 0 || executionContext->apiHandler->getEndpoint() == Endpoint::RESPONSES) {
            // Emit each delta. All are mid-stream so finishReason is NONE.
            for (size_t i = 0; i < count; ++i) {
                std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                    std::move(deltas[i]),
                    ov::genai::GenerationFinishReason::NONE);
                if (!serialized.empty()) {
                    executionContext->response += wrapTextInServerSideEventMessage(serialized);
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", serialized);
                }
            }
            if (count == 0) {
                // No delta generated yet — emit lifecycle events (response.created, response.in_progress)
                // for the RESPONSES endpoint before any content arrives.
                if (!executionContext->lifecyclePrimed) {
                    std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                        rapidjson::Document{}, ov::genai::GenerationFinishReason::NONE);
                    if (!serialized.empty()) {
                        executionContext->response += wrapTextInServerSideEventMessage(serialized);
                        executionContext->lifecyclePrimed = true;
                    }
                }
            }
        } else if (isFirstToken) {
            std::string serializedChunk = executionContext->apiHandler->serializeStreamingHandshakeChunk();
            if (!serializedChunk.empty()) {
                executionContext->response = wrapTextInServerSideEventMessage(serializedChunk);
            }
        }
        executionContext->sendLoopbackSignal = true;
    } else {
        // Finishing: emit all pending deltas; the last one gets the real finishReason.
        if (count > 0) {
            for (size_t i = 0; i < count; ++i) {
                const bool isLast = (i == count - 1);
                std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                    std::move(deltas[i]),
                    isLast ? finishReason : ov::genai::GenerationFinishReason::NONE);
                if (!serialized.empty()) {
                    executionContext->response += wrapTextInServerSideEventMessage(serialized);
                }
            }
        } else {
            // No delta produced (generation ended on a swallowed token).
            // Still emit a chunk carrying the finish_reason with an empty Document.
            std::string serialized = executionContext->apiHandler->serializeStreamingChunk(
                rapidjson::Document{}, finishReason);
            if (!serialized.empty()) {
                executionContext->response += wrapTextInServerSideEventMessage(serialized);
            }
        }
        if (executionContext->apiHandler->getStreamOptions().includeUsage) {
            std::string usageChunk = executionContext->apiHandler->serializeStreamingUsageChunk();
            if (!usageChunk.empty()) {
                executionContext->response += wrapTextInServerSideEventMessage(usageChunk);
            }
        }
        executionContext->response += wrapTextInServerSideEventMessage("[DONE]");
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", executionContext->response);
        executionContext->sendLoopbackSignal = false;
    }

    return absl::OkStatus();
}

absl::Status prepareLegacyPartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto legacyCtx = std::static_pointer_cast<LegacyServableExecutionContextBase>(executionContext);
    if (legacyCtx->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    std::vector<rapidjson::Document> deltas = executionContext->deltaChannel.drain();
    const bool isFinishing = executionContext->deltaChannel.complete();

    // Helper: accumulate verbose raw text from a delta's content field.
    // Both LLM-Legacy (switched from token-based) and VLM-Legacy use per-delta
    // text extraction, which is correct because OVMSTextStreamer is configured with
    // skip_special_tokens(false) in verbose mode, so delta content already includes
    // special tokens.
    auto appendVerboseContent = [&](const rapidjson::Document& delta) {
        if (executionContext->apiHandler->isVerboseResponse() &&
            delta.HasMember("delta") && delta["delta"].IsObject() &&
            delta["delta"].HasMember("content") && delta["delta"]["content"].IsString()) {
            executionContext->apiHandler->appendVerboseRawText(delta["delta"]["content"].GetString());
        }
    };

    if (!isFinishing) {
        // For RESPONSES endpoint, always call serializeStreamingChunk so that
        // output item initialization events are emitted even before the tokenizer produces text.
        if (deltas.size() > 0 || executionContext->apiHandler->getEndpoint() == Endpoint::RESPONSES) {
            for (auto& delta : deltas) {
                appendVerboseContent(delta);
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
        // Wait for the readySignal
        // (set right after pipe->generate() returns and results are assigned)
        // to guarantee results is populated before we read finish_reasons and perf_metrics.
        // Also ensures success flag is accurate.
        legacyCtx->finished.wait();
        if (!legacyCtx->success) {
            return absl::InvalidArgumentError("Request processing failed, check its correctness.");
        }
        OVMS_PROFILE_SCOPE("Generation of last streaming response");
        // end() was already called by pipe->generate() internally; all deltas are
        // already in deltaChannel before signalComplete() fired. Drain any remaining.
        for (auto& d : executionContext->deltaChannel.drain()) {
            deltas.push_back(std::move(d));
        }
        // Legacy generation path always runs with deltas=1, so we read the single finish reason at index 0.
        const ov::genai::GenerationFinishReason finishReason = legacyCtx->legacyFinishReason();
        legacyCtx->setLegacyUsage(*executionContext->apiHandler);
        if (!deltas.empty()) {
            for (size_t i = 0; i < deltas.size(); ++i) {
                const bool isLast = (i == deltas.size() - 1);
                appendVerboseContent(deltas[i]);
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

absl::Status LegacyServableBase::preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    return prepareLegacyPartialResponse(executionContext);
}

void logRequestDetails(const ovms::HttpPayload& payload) {
    auto parsedJson = payload.parsedJson;
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    parsedJson->Accept(writer);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", buffer.GetString());
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
}

}  // namespace ovms

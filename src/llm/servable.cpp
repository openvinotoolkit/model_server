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
#include "input_workarounds.hpp"
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

    if (executionContext->apiHandler->isStream()) {
        auto ovmsCallback = [& ctx = *executionContext](rapidjson::Document delta) -> ov::genai::StreamingStatus {
            ctx.deltaChannel.push(std::move(delta));
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
    executionContext->generationConfigBuilder = std::make_shared<GenerationConfigBuilder>(getProperties()->baseGenerationConfig,
        getProperties()->toolParserName,
        getProperties()->enableToolGuidedGeneration,
        getProperties()->decodingMethod);
    executionContext->generationConfigBuilder->parseConfigFromRequest(executionContext->apiHandler->getRequest());
    executionContext->generationConfigBuilder->adjustConfigForDecodingMethod();
    try {
        executionContext->generationConfigBuilder->validateStructuredOutputConfig(getProperties()->tokenizer);
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Tool guided generation will not be applied due to JSON schema validation failure: {}", e.what());
        executionContext->generationConfigBuilder->unsetStructuredOutputConfig();
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

    // Apply input workarounds based on detected chat template capabilities.
    // This modifies the request JSON before chat template application.
    // Currently effective only on the Python Jinja path; GenAI C++ path workarounds
    // will be added during the pre-processing refactoring.
    auto applyInputWorkarounds = [this](const std::string& jsonBody) -> std::string {
        const auto& caps = getProperties()->chatTemplateCaps;
        if (!caps.requiresObjectArguments && !caps.requiresNonNullContent) {
            return jsonBody;  // no workarounds needed
        }
        rapidjson::Document doc;
        doc.Parse(jsonBody.c_str());
        if (doc.HasParseError()) {
            return jsonBody;
        }
        input_workarounds::applyToJson(caps, getProperties()->detectedModelFamily, doc);
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        return buffer.GetString();
    };

    std::string inputText;
    switch (executionContext->endpoint) {
    case Endpoint::CHAT_COMPLETIONS: {
#if (PYTHON_DISABLE == 0)
        if (getProperties()->chatTemplateMode == ChatTemplateMode::JINJA) {
            bool success;
            auto tplStart = std::chrono::steady_clock::now();
            if (executionContext->apiHandler->getProcessedJson().size() > 0) {
                std::string modifiedJson = applyInputWorkarounds(executionContext->apiHandler->getProcessedJson());
                success = PyJinjaTemplateProcessor::applyChatTemplate(getProperties()->templateProcessor, getProperties()->modelsPath, modifiedJson, inputText);
            } else {
                std::string modifiedJson = applyInputWorkarounds(executionContext->payload.body);
                success = PyJinjaTemplateProcessor::applyChatTemplate(getProperties()->templateProcessor, getProperties()->modelsPath, modifiedJson, inputText);
            }
            auto tplEnd = std::chrono::steady_clock::now();
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "applyChatTemplate Jinja (chat): {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(tplEnd - tplStart).count());
            if (!success) {
                return absl::Status(absl::StatusCode::kInvalidArgument, inputText);
            }
        } else  // NOLINT(readability/braces)
#endif
        {
            ov::genai::ChatHistory& chatHistory = executionContext->apiHandler->getChatHistory();
            input_workarounds::applyToHistory(getProperties()->chatTemplateCaps, getProperties()->detectedModelFamily, chatHistory);
            constexpr bool addGenerationPrompt = true;
            auto toolParsingResult = executionContext->apiHandler->parseToolsToJsonContainer();
            if (!toolParsingResult.ok()) {
                return toolParsingResult.status();
            }
            const auto& tools = toolParsingResult.value();
            auto chatTemplateKwargsParsingResult = executionContext->apiHandler->parseChatTemplateKwargsToJsonContainer();
            if (!chatTemplateKwargsParsingResult.ok()) {
                return chatTemplateKwargsParsingResult.status();
            }
            const auto& chatTemplateKwargs = chatTemplateKwargsParsingResult.value();
            try {
                auto tplStart = std::chrono::steady_clock::now();
                inputText = getProperties()->tokenizer.apply_chat_template(chatHistory, addGenerationPrompt, {}, tools, chatTemplateKwargs);
                auto tplEnd = std::chrono::steady_clock::now();
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "apply_chat_template (chat): {} us",
                    std::chrono::duration_cast<std::chrono::microseconds>(tplEnd - tplStart).count());
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to apply chat template: {}", e.what());
                return absl::Status(absl::StatusCode::kInvalidArgument, "Failed to apply chat template. The model either does not have chat template or has an invalid one.");
            }
        }
        if (inputText.size() == 0) {
            return absl::Status(absl::StatusCode::kInvalidArgument, "Final prompt after applying chat template is empty");
        }
        if (executionContext->apiHandler->getOutputParser() != nullptr) {
            executionContext->apiHandler->getOutputParser()->detectAndSetImplicitReasoningStart(inputText);
        }
        break;
    }
    case Endpoint::RESPONSES: {
        if (executionContext->apiHandler->getChatHistory().size() > 0) {
#if (PYTHON_DISABLE == 0)
            if (getProperties()->chatTemplateMode == ChatTemplateMode::JINJA) {
                std::string modifiedJson = applyInputWorkarounds(executionContext->apiHandler->getProcessedJson());
                auto tplStart = std::chrono::steady_clock::now();
                bool success = PyJinjaTemplateProcessor::applyChatTemplate(getProperties()->templateProcessor, getProperties()->modelsPath, modifiedJson, inputText);
                auto tplEnd = std::chrono::steady_clock::now();
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "applyChatTemplate Jinja (responses): {} us",
                    std::chrono::duration_cast<std::chrono::microseconds>(tplEnd - tplStart).count());
                if (!success) {
                    return absl::Status(absl::StatusCode::kInvalidArgument, inputText);
                }
            } else  // NOLINT(readability/braces)
#endif
            {
                ov::genai::ChatHistory& chatHistory = executionContext->apiHandler->getChatHistory();
                input_workarounds::applyToHistory(getProperties()->chatTemplateCaps, getProperties()->detectedModelFamily, chatHistory);
                constexpr bool addGenerationPrompt = true;
                auto toolParsingResult = executionContext->apiHandler->parseToolsToJsonContainer();
                if (!toolParsingResult.ok()) {
                    return toolParsingResult.status();
                }
                const auto& tools = toolParsingResult.value();
                auto chatTemplateKwargsParsingResult = executionContext->apiHandler->parseChatTemplateKwargsToJsonContainer();
                if (!chatTemplateKwargsParsingResult.ok()) {
                    return chatTemplateKwargsParsingResult.status();
                }
                const auto& chatTemplateKwargs = chatTemplateKwargsParsingResult.value();
                try {
                    auto tplStart = std::chrono::steady_clock::now();
                    inputText = getProperties()->tokenizer.apply_chat_template(chatHistory, addGenerationPrompt, {}, tools, chatTemplateKwargs);
                    auto tplEnd = std::chrono::steady_clock::now();
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "apply_chat_template (responses): {} us",
                        std::chrono::duration_cast<std::chrono::microseconds>(tplEnd - tplStart).count());
                } catch (const std::exception& e) {
                    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to apply chat template: {}", e.what());
                    return absl::Status(absl::StatusCode::kInvalidArgument, "Failed to apply chat template. The model either does not have chat template or has an invalid one.");
                }
            }
            if (inputText.size() == 0) {
                return absl::Status(absl::StatusCode::kInvalidArgument, "Final prompt after applying chat template is empty");
            }
            if (executionContext->apiHandler->getOutputParser() != nullptr) {
                executionContext->apiHandler->getOutputParser()->detectAndSetImplicitReasoningStart(inputText);
            }
        } else {
            auto prompt = executionContext->apiHandler->getPrompt();
            if (!prompt.has_value()) {
                return absl::Status(absl::StatusCode::kInvalidArgument, "input is missing");
            }
            inputText = prompt.value();
        }
        break;
    }
    case Endpoint::COMPLETIONS: {
        inputText = executionContext->apiHandler->getPrompt().value();
        break;
    }
    case Endpoint::TOKENIZE:
        return absl::InternalError("Tokenize endpoint should not reach prepareInputs stage");
    }
    if (Config::instance().getServerSettings().verboseResponse) {
        executionContext->apiHandler->enableVerboseResponse(inputText);
    }
    bool encodeAddSpecialTokens = (executionContext->endpoint == Endpoint::COMPLETIONS);
    executionContext->inputIds = getProperties()->tokenizer.encode(inputText, ov::genai::add_special_tokens(encodeAddSpecialTokens)).input_ids;
    if (getProperties()->maxModelLength.has_value()) {
        if (executionContext->inputIds.get_size() > getProperties()->maxModelLength.value()) {
            std::stringstream ss;
            ss << "Number of prompt tokens: " << executionContext->inputIds.get_size() << " exceeds model max length: " << getProperties()->maxModelLength.value();
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
            return absl::Status(absl::StatusCode::kInvalidArgument, ss.str());
        }
        if (executionContext->apiHandler->getMaxTokens().has_value() && executionContext->inputIds.get_size() + executionContext->apiHandler->getMaxTokens().value() > getProperties()->maxModelLength.value()) {
            std::stringstream ss;
            ss << "Number of prompt tokens: " << executionContext->inputIds.get_size() << " + max tokens value: " << executionContext->apiHandler->getMaxTokens().value() << " exceeds model max length: " << getProperties()->maxModelLength.value();
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, ss.str());
            return absl::Status(absl::StatusCode::kInvalidArgument, ss.str());
        }
    }

    executionContext->apiHandler->setPromptTokensUsage(executionContext->inputIds.get_size());
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Pipeline input text: {}", inputText);
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

void logRequestDetails(const ovms::HttpPayload& payload) {
    auto parsedJson = payload.parsedJson;
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    parsedJson->Accept(writer);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", buffer.GetString());
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
}

}  // namespace ovms

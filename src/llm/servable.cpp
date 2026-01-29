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
#include "servable.hpp"

#include <memory>
#include <stdexcept>
#include <string>
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
#include "builtin_tool_executor.hpp"
#include "text_utils.hpp"
#include "../tokenize/tokenize_parser.hpp"

namespace ovms {

GenAiServable::GenAiServable() {
    SPDLOG_LOGGER_INFO(llm_calculator_logger, "GenAiServable: Constructor called, BuiltInToolExecutor initialized with mock handlers");
}

bool GenAiServable::initializeMcpClient(const std::string& url, const std::string& sseEndpoint) {
    SPDLOG_LOGGER_INFO(llm_calculator_logger, "GenAiServable::initializeMcpClient called with url={}, sseEndpoint={}", url, sseEndpoint);
    return builtInToolExecutor.initializeMcpClient(url, sseEndpoint);
}

bool GenAiServable::isMcpClientReady() const {
    return builtInToolExecutor.isMcpClientReady();
}

void GenAiServable::determineDecodingMethod() {
    getProperties()->decodingMethod = DecodingMethod::STANDARD;
    auto& pluginConfig = getProperties()->pluginConfig;
    if (pluginConfig.find("draft_model") != pluginConfig.end()) {
        getProperties()->decodingMethod = DecodingMethod::SPECULATIVE_DECODING;
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
    } else if (TokenizeParser::isTokenizeEndpoint(payload.uri)) {
        executionContext->endpoint = Endpoint::TOKENIZE;
    } else {
        return absl::InvalidArgumentError("Wrong endpoint. Allowed endpoints: /v3/chat/completions, /v3/completions");
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
        executionContext->apiHandler = std::make_shared<OpenAIChatCompletionsHandler>(*executionContext->payload.parsedJson,
            executionContext->endpoint,
            std::chrono::system_clock::now(),
            getProperties()->tokenizer,
            getProperties()->toolParserName,
            getProperties()->reasoningParserName);
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Failed to create API handler: {}", e.what());
        return absl::InvalidArgumentError(std::string("Failed to create API handler: ") + e.what());
    }
    auto& config = ovms::Config::instance();

    auto status = executionContext->apiHandler->parseRequest(getProperties()->maxTokensLimit, getProperties()->bestOfLimit, getProperties()->maxModelLength, config.getServerSettings().allowedLocalMediaPath);
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
        ov::AnyMap streamerConfig;
        if (executionContext->apiHandler->getOutputParser() != nullptr &&
            (executionContext->apiHandler->getOutputParser()->requiresStreamingWithSpecialTokens())) {
            streamerConfig.insert(ov::genai::skip_special_tokens(false));
        }
        executionContext->textStreamer = std::make_shared<ov::genai::TextStreamer>(getProperties()->tokenizer, callback, streamerConfig);
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

    std::string inputText;
    switch (executionContext->endpoint) {
    case Endpoint::CHAT_COMPLETIONS: {
#if (PYTHON_DISABLE == 0)
        bool success;
        if (executionContext->apiHandler->getProcessedJson().size() > 0) {
            success = PyJinjaTemplateProcessor::applyChatTemplate(getProperties()->templateProcessor, getProperties()->modelsPath, executionContext->apiHandler->getProcessedJson(), inputText);
        } else {
            success = PyJinjaTemplateProcessor::applyChatTemplate(getProperties()->templateProcessor, getProperties()->modelsPath, executionContext->payload.body, inputText);
        }
        if (!success) {
            return absl::Status(absl::StatusCode::kInvalidArgument, inputText);
        }
#else
        ov::genai::ChatHistory& chatHistory = executionContext->apiHandler->getChatHistory();
        constexpr bool add_generation_prompt = true;  // confirm it should be hardcoded
        try {
            inputText = getProperties()->tokenizer.apply_chat_template(chatHistory, add_generation_prompt);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to apply chat template: {}", e.what());
            return absl::Status(absl::StatusCode::kInvalidArgument, "Failed to apply chat template. The model either does not have chat template or has an invalid one.");
        }
#endif
        if (inputText.size() == 0) {
            return absl::Status(absl::StatusCode::kInvalidArgument, "Final prompt after applying chat template is empty");
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
    SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Pipeline input text: {}", inputText);
    SPDLOG_LOGGER_ERROR(llm_calculator_logger, "{}", getPromptTokensString(executionContext->inputIds));

    return absl::OkStatus();
}

absl::Status GenAiServable::prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "prepareCompleteResponse called, generationOutputs size: {}", executionContext->generationOutputs.size());
    
    if (executionContext->generationOutputs.empty()) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "prepareCompleteResponse called but generationOutputs is empty");
        return absl::InternalError("No generation outputs available");
    }
    
    // Parse the first generation output and store it for later use (e.g., built-in tool detection)
    const auto& generationOutput = executionContext->generationOutputs[0];
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsing generation output, generated_ids size: {}", generationOutput.generated_ids.size());
    
    executionContext->lastParsedOutput = executionContext->apiHandler->parseGenerationOutput(generationOutput.generated_ids);
    executionContext->hasLastParsedOutput = true;
    
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsed output - content size: {}, toolCalls: {}, builtInToolCalls: {}, reasoning size: {}",
        executionContext->lastParsedOutput.content.size(), 
        executionContext->lastParsedOutput.toolCalls.size(), 
        executionContext->lastParsedOutput.builtInToolCalls.size(),
        executionContext->lastParsedOutput.reasoning.size());
    
    // Log first 200 chars of content for debugging
    if (!executionContext->lastParsedOutput.content.empty()) {
        std::string contentPreview = executionContext->lastParsedOutput.content.substr(0, std::min(size_t(200), executionContext->lastParsedOutput.content.size()));
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Parsed content preview: {}", contentPreview);
    }
    
    // Log each built-in tool call
    for (size_t i = 0; i < executionContext->lastParsedOutput.builtInToolCalls.size(); ++i) {
        const auto& call = executionContext->lastParsedOutput.builtInToolCalls[i];
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Built-in tool call [{}]: name='{}', id='{}', arguments='{}'",
            i, call.name, call.id, call.arguments);
    }
    
    // Serialize response, passing the pre-parsed output to avoid double parsing
    executionContext->response = executionContext->apiHandler->serializeUnaryResponse(executionContext->generationOutputs, &executionContext->lastParsedOutput);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response prepared, length: {}", executionContext->response.size());
    return absl::OkStatus();
}

bool GenAiServable::hasBuiltInToolCalls(const std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    if (!executionContext->hasLastParsedOutput) {
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "hasBuiltInToolCalls: no parsed output available");
        return false;
    }
    bool result = !executionContext->lastParsedOutput.builtInToolCalls.empty();
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "hasBuiltInToolCalls check: {} (count: {})", result, executionContext->lastParsedOutput.builtInToolCalls.size());
    return result;
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
            std::string serializedChunk = executionContext->apiHandler->serializeStreamingChunk(lastTextChunk, finishReason);
            if (!serializedChunk.empty()) {
                executionContext->response = wrapTextInServerSideEventMessage(serializedChunk);
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", executionContext->response);
            }
        }
        executionContext->sendLoopbackSignal = true;
    } else {  // finish generation
        OVMS_PROFILE_SCOPE("Generation of last streaming response");
        executionContext->textStreamer->end();
        // if streamer::put returned a value, streamer::end() result will not contain it, so we add it manually
        if (!executionContext->lastStreamerCallbackOutput.empty()) {
            lastTextChunk = lastTextChunk + executionContext->lastStreamerCallbackOutput;
        }
        std::string serializedChunk = executionContext->apiHandler->serializeStreamingChunk(lastTextChunk, finishReason);
        if (!serializedChunk.empty()) {
            executionContext->response = wrapTextInServerSideEventMessage(serializedChunk);
        }
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
void logRequestDetails(const ovms::HttpPayload& payload) {
    auto parsedJson = payload.parsedJson;
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    parsedJson->Accept(writer);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request body: {}", buffer.GetString());
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Request uri: {}", payload.uri);
}
#pragma GCC diagnostic pop
#pragma warning(push)

// ----------- Built-in tool execution methods ------------

BuiltInToolResults_t GenAiServable::executeBuiltInTools(const ToolCalls_t& builtInToolCalls) {
    SPDLOG_LOGGER_INFO(llm_calculator_logger, "GenAiServable::executeBuiltInTools called with {} tool calls, MCP ready: {}",
                       builtInToolCalls.size(), builtInToolExecutor.isMcpClientReady() ? "YES" : "NO");
    return builtInToolExecutor.execute(builtInToolCalls);
}

void GenAiServable::appendToolResultsToChatHistory(std::shared_ptr<GenAiServableExecutionContext>& executionContext,
                                                    const std::string& assistantContent,
                                                    const ToolCalls_t& builtInToolCalls,
                                                    const BuiltInToolResults_t& toolResults) {
#if (PYTHON_DISABLE == 0)
    // When Python is enabled, we need to modify the JSON document
    // Get the document and append messages to the "messages" array
    rapidjson::Document& doc = executionContext->apiHandler->getDocument();

    if (!doc.HasMember("messages") || !doc["messages"].IsArray()) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "Cannot append tool results: messages array not found in request");
        return;
    }

    rapidjson::Value& messages = doc["messages"];
    auto& allocator = doc.GetAllocator();

    // Add assistant message with tool calls
    rapidjson::Value assistantMessage(rapidjson::kObjectType);
    assistantMessage.AddMember("role", rapidjson::Value("assistant", allocator), allocator);

    if (!assistantContent.empty()) {
        assistantMessage.AddMember("content", rapidjson::Value(assistantContent.c_str(), allocator), allocator);
    } else {
        assistantMessage.AddMember("content", rapidjson::Value("", allocator), allocator);
    }

    // Add tool_calls array to assistant message
    if (!builtInToolCalls.empty()) {
        rapidjson::Value toolCallsArray(rapidjson::kArrayType);
        for (const auto& toolCall : builtInToolCalls) {
            rapidjson::Value toolCallObj(rapidjson::kObjectType);
            toolCallObj.AddMember("id", rapidjson::Value(toolCall.id.c_str(), allocator), allocator);
            toolCallObj.AddMember("type", rapidjson::Value("function", allocator), allocator);

            rapidjson::Value functionObj(rapidjson::kObjectType);
            functionObj.AddMember("name", rapidjson::Value(toolCall.name.c_str(), allocator), allocator);
            functionObj.AddMember("arguments", rapidjson::Value(toolCall.arguments.c_str(), allocator), allocator);
            toolCallObj.AddMember("function", functionObj, allocator);

            toolCallsArray.PushBack(toolCallObj, allocator);
        }
        assistantMessage.AddMember("tool_calls", toolCallsArray, allocator);
    }

    messages.PushBack(assistantMessage, allocator);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Added assistant message to JSON with {} built-in tool calls", builtInToolCalls.size());

    // Add tool result messages
    for (const auto& result : toolResults) {
        rapidjson::Value toolMessage(rapidjson::kObjectType);
        toolMessage.AddMember("role", rapidjson::Value("tool", allocator), allocator);
        toolMessage.AddMember("tool_call_id", rapidjson::Value(result.toolCallId.c_str(), allocator), allocator);
        toolMessage.AddMember("name", rapidjson::Value(result.toolName.c_str(), allocator), allocator);
        toolMessage.AddMember("content", rapidjson::Value(result.content.c_str(), allocator), allocator);
        messages.PushBack(toolMessage, allocator);
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Added tool result message for tool: {} with id: {}", result.toolName, result.toolCallId);
    }

    // Serialize the updated document to processedJson for the template processor
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    executionContext->apiHandler->setProcessedJson(buffer.GetString());

#else
    // When Python is disabled, use ChatHistory
    auto& chatHistory = executionContext->apiHandler->getChatHistory();

    // Add assistant message with the content and tool calls
    ov::AnyMap assistantMessage;
    assistantMessage["role"] = std::string("assistant");

    if (!assistantContent.empty()) {
        assistantMessage["content"] = assistantContent;
    } else {
        assistantMessage["content"] = std::string("");
    }

    // Add tool_calls to assistant message as a formatted string representing the calls
    // Note: The exact format depends on what the chat template expects
    if (!builtInToolCalls.empty()) {
        std::stringstream toolCallsStr;
        toolCallsStr << "[";
        for (size_t i = 0; i < builtInToolCalls.size(); ++i) {
            if (i > 0) toolCallsStr << ", ";
            toolCallsStr << "{\"id\": \"" << builtInToolCalls[i].id << "\", "
                         << "\"type\": \"function\", "
                         << "\"function\": {\"name\": \"" << builtInToolCalls[i].name << "\", "
                         << "\"arguments\": " << builtInToolCalls[i].arguments << "}}";
        }
        toolCallsStr << "]";
        assistantMessage["tool_calls"] = toolCallsStr.str();
    }

    chatHistory.push_back(assistantMessage);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Added assistant message to chat history with {} built-in tool calls", builtInToolCalls.size());

    // Add tool result messages
    for (const auto& result : toolResults) {
        ov::AnyMap toolMessage;
        toolMessage["role"] = std::string("tool");
        toolMessage["tool_call_id"] = result.toolCallId;
        toolMessage["name"] = result.toolName;
        toolMessage["content"] = result.content;
        chatHistory.push_back(toolMessage);
        SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Added tool result message for tool: {} with id: {}", result.toolName, result.toolCallId);
    }
#endif
}

}  // namespace ovms

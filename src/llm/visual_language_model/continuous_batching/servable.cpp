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
#include <unordered_map>
#include <vector>

#include "src/port/rapidjson_document.hpp"
#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

#include "../../../config.hpp"
#include "../../../logging.hpp"
#include "../../../tokenize/tokenize_parser.hpp"
#include "../../text_utils.hpp"
#if (PYTHON_DISABLE == 0)
#include "../../py_jinja_template_processor.hpp"
#endif

namespace ovms {

absl::Status VisualLanguageModelServable::addRequestToPipeline(std::shared_ptr<ContinuousBatchingServableExecutionContext>& executionContext) {
    auto vlmExecutionContext = std::static_pointer_cast<VisualLanguageModelServableExecutionContext>(executionContext);
    vlmExecutionContext->generationHandle = properties->pipeline->add_request(currentRequestId++,  // to be removed from API?
        vlmExecutionContext->inputText, vlmExecutionContext->inputImages,
        vlmExecutionContext->generationConfigBuilder->getConfig());
    return absl::OkStatus();
}

absl::Status VisualLanguageModelServable::loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const ovms::HttpPayload& payload) {
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

std::shared_ptr<GenAiServableExecutionContext> VisualLanguageModelServable::createExecutionContext() {
    return std::make_shared<VisualLanguageModelServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> VisualLanguageModelServable::getProperties() {
    return properties;
}

absl::Status VisualLanguageModelServable::prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto vlmExecutionContext = std::static_pointer_cast<VisualLanguageModelServableExecutionContext>(executionContext);
    if (vlmExecutionContext->apiHandler == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "API handler is not initialized");
    }
    if (executionContext->endpoint == Endpoint::CHAT_COMPLETIONS || executionContext->endpoint == Endpoint::RESPONSES) {
        ov::genai::ChatHistory& chatHistory = vlmExecutionContext->apiHandler->getChatHistory();
        const ImageHistory& imageHistory = vlmExecutionContext->apiHandler->getImageHistory();

        // Responses parsing keeps images in imageHistory only; Chat Completions parsing
        // already inserts placeholders in content to preserve multipart order.
        if (executionContext->endpoint == Endpoint::RESPONSES) {
            for (size_t i = 0; i < chatHistory.size(); i++) {
                const auto& message = chatHistory[i];
                if (message["content"].as_string().value_or("").find("<ov_genai_image_") != std::string::npos) {
                    return absl::InvalidArgumentError("Message contains restricted <ov_genai_image> tag");
                }
            }
        }

        if (executionContext->endpoint == Endpoint::RESPONSES) {
            size_t imageIndex = 0;
            std::unordered_map<size_t, std::string> imageTags;
            for (const auto& image : imageHistory) {
                const auto& [chatTurnIndex, imageTensor] = image;
                std::string imageTag = "<ov_genai_image_" + std::to_string(imageIndex++) + ">\n";
                imageTags[chatTurnIndex] = imageTags[chatTurnIndex] + imageTag;
                vlmExecutionContext->inputImages.push_back(imageTensor);
            }
            for (const auto& [chatTurnIndex, imageTagString] : imageTags) {
                std::string messageContent = chatHistory[chatTurnIndex]["content"].as_string().value_or("");
                chatHistory[chatTurnIndex]["content"] = imageTagString + messageContent;
            }
        } else {
            for (const auto& image : imageHistory) {
                const auto& [chatTurnIndex, imageTensor] = image;
                (void)chatTurnIndex;
                vlmExecutionContext->inputImages.push_back(imageTensor);
            }
        }

#if (PYTHON_DISABLE == 0)
        if (getProperties()->chatTemplateMode == ChatTemplateMode::JINJA) {
            std::string jsonForTemplate;
            if (vlmExecutionContext->apiHandler->getProcessedJson().size() > 0) {
                jsonForTemplate = vlmExecutionContext->apiHandler->getProcessedJson();
            } else {
                jsonForTemplate = vlmExecutionContext->payload.body;
            }
            // Inject image tags into the JSON messages for Python Jinja template processing
            if (!imageTags.empty()) {
                rapidjson::Document jsonDoc;
                jsonDoc.Parse(jsonForTemplate.c_str());
                if (!jsonDoc.HasParseError() && jsonDoc.IsObject() && jsonDoc.HasMember("messages") && jsonDoc["messages"].IsArray()) {
                    auto& messages = jsonDoc["messages"];
                    for (const auto& [chatTurnIndex, imageTagString] : imageTags) {
                        if (chatTurnIndex < messages.Size()) {
                            auto& msg = messages[chatTurnIndex];
                            if (msg.IsObject() && msg.HasMember("content") && msg["content"].IsString()) {
                                std::string newContent = imageTagString + msg["content"].GetString();
                                msg["content"].SetString(newContent.c_str(), newContent.length(), jsonDoc.GetAllocator());
                            }
                        }
                    }
                    rapidjson::StringBuffer buffer;
                    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                    jsonDoc.Accept(writer);
                    jsonForTemplate = buffer.GetString();
                }
            }
            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "VLM CB: Applying chat template using Python Jinja processor");
            bool success = PyJinjaTemplateProcessor::applyChatTemplate(getProperties()->templateProcessor, getProperties()->modelsPath, jsonForTemplate, vlmExecutionContext->inputText);
            if (!success) {
                return absl::Status(absl::StatusCode::kInvalidArgument, vlmExecutionContext->inputText);
            }
        } else  // NOLINT(readability/braces)
#endif
        {
            constexpr bool addGenerationPrompt = true;  // confirm it should be hardcoded
            auto toolParsingResult = vlmExecutionContext->apiHandler->parseToolsToJsonContainer();
            if (!toolParsingResult.ok()) {
                return toolParsingResult.status();
            }
            const auto& tools = toolParsingResult.value();
            auto chatTemplateKwargsParsingResult = vlmExecutionContext->apiHandler->parseChatTemplateKwargsToJsonContainer();
            if (!chatTemplateKwargsParsingResult.ok()) {
                return chatTemplateKwargsParsingResult.status();
            }
            const auto& chatTemplateKwargs = chatTemplateKwargsParsingResult.value();
            if (llm_calculator_logger->should_log(spdlog::level::trace)) {
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "VLM chatHistory messages: {}", chatHistory.get_messages().to_json_string());
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "VLM chatHistory.get_tools(): {}", chatHistory.get_tools().to_json_string());
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "VLM chatHistory.get_extra_context(): {}", chatHistory.get_extra_context().to_json_string());
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "VLM tools: {}", tools.has_value() ? tools->to_json_string() : std::string("<none>"));
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "VLM chatTemplateKwargs: {}", chatTemplateKwargs.has_value() ? chatTemplateKwargs->to_json_string() : std::string("<none>"));
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "VLM addGenerationPrompt: {}", addGenerationPrompt);
            }
            try {
                vlmExecutionContext->inputText = properties->tokenizer.apply_chat_template(chatHistory, addGenerationPrompt, {}, tools, chatTemplateKwargs);
            } catch (const std::exception& e) {
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Failed to apply chat template: {}", e.what());
                return absl::Status(absl::StatusCode::kInvalidArgument, "Failed to apply chat template. The model either does not have chat template or has an invalid one.");
            }
        }
        if (vlmExecutionContext->inputText.empty()) {
            return absl::Status(absl::StatusCode::kInvalidArgument, "Final prompt after applying chat template is empty");
        }
        if (vlmExecutionContext->apiHandler->getOutputParser() != nullptr) {
            vlmExecutionContext->apiHandler->getOutputParser()->detectAndSetImplicitReasoningStart(vlmExecutionContext->inputText);
        }
    } else {
        return absl::InvalidArgumentError("Unsupported endpoint");
    }

    if (Config::instance().getServerSettings().verboseResponse) {
        vlmExecutionContext->apiHandler->enableVerboseResponse(vlmExecutionContext->inputText);
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

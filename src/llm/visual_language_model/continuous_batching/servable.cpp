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

#include "../../../logging.hpp"
#include "../../text_utils.hpp"
#include "../../../tokenize/tokenize_parser.hpp"

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

        for (size_t i = 0; i < chatHistory.size(); i++) {
            const auto& message = chatHistory[i];
            const auto& contentField = message["content"];
            if (contentField.is_array()) {
                for (size_t j = 0; j < contentField.size(); j++) {
                    const auto& item = contentField[j];
                    if (item["type"].as_string().value_or("") == "text" &&
                        item["text"].as_string().value_or("").find("<ov_genai_image_") != std::string::npos) {
                        return absl::InvalidArgumentError("Message contains restricted <ov_genai_image> tag");
                    }
                }
            } else if (contentField.as_string().value_or("").find("<ov_genai_image_") != std::string::npos) {
                return absl::InvalidArgumentError("Message contains restricted <ov_genai_image> tag");
            }
        }

        // imageHistory is a flat ordered list of tensors matching the {type:image} items in
        // chatHistory. Pass them directly to add_request; the chat template applied below will
        // emit the model-specific image tokens at the correct positions.
        vlmExecutionContext->inputImages = vlmExecutionContext->apiHandler->getImageHistory();

#if (PYTHON_DISABLE == 0)
        bool success;
        if (vlmExecutionContext->apiHandler->getProcessedJson().size() > 0) {
            success = PyJinjaTemplateProcessor::applyChatTemplate(getProperties()->templateProcessor, getProperties()->modelsPath, vlmExecutionContext->apiHandler->getProcessedJson(), vlmExecutionContext->inputText);
        } else {
            success = PyJinjaTemplateProcessor::applyChatTemplate(getProperties()->templateProcessor, getProperties()->modelsPath, vlmExecutionContext->payload.body, vlmExecutionContext->inputText);
        }
        if (!success) {
            return absl::Status(absl::StatusCode::kInvalidArgument, vlmExecutionContext->inputText);
        }
#else
        constexpr bool addGenerationPrompt = true;  // confirm it should be hardcoded
        auto toolsStatus = vlmExecutionContext->apiHandler->parseToolsToJsonContainer();
        if (!toolsStatus.ok()) {
            return toolsStatus.status();
        }
        const auto& tools = toolsStatus.value();
        auto chatTemplateKwargsStatus = vlmExecutionContext->apiHandler->parseChatTemplateKwargsToJsonContainer();
        if (!chatTemplateKwargsStatus.ok()) {
            return chatTemplateKwargsStatus.status();
        }
        const auto& chatTemplateKwargs = chatTemplateKwargsStatus.value();
        vlmExecutionContext->inputText = properties->tokenizer.apply_chat_template(chatHistory, addGenerationPrompt, {}, tools, chatTemplateKwargs);
#endif
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

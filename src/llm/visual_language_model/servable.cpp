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

#include "../../logging.hpp"

namespace ovms {

absl::Status VisualLanguageModelServable::addRequestToPipeline(std::shared_ptr<ContinuousBatchingServableExecutionContext>& executionContext) {
    auto vlmExecutionContext = std::static_pointer_cast<VisualLanguageModelServableExecutionContext>(executionContext);
    vlmExecutionContext->generationHandle = properties->pipeline->add_request(currentRequestId++,  // to be removed from API?
        vlmExecutionContext->inputText, vlmExecutionContext->inputImages,
        vlmExecutionContext->apiHandler->createGenerationConfig());
    return absl::OkStatus();
}

std::shared_ptr<GenAiServableExecutionContext> VisualLanguageModelServable::createExecutionContext() {
    return std::make_shared<VisualLanguageModelServableExecutionContext>();
}

std::shared_ptr<GenAiServableProperties> VisualLanguageModelServable::getProperties() {
    return properties;
}

bool VisualLanguageModelServable::supportsSpeculativeDecoding() const {
    return false;
}

absl::Status VisualLanguageModelServable::prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext) {
    auto vlmExecutionContext = std::static_pointer_cast<VisualLanguageModelServableExecutionContext>(executionContext);
    if (vlmExecutionContext->apiHandler == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "API handler is not initialized");
    }
    if (executionContext->endpoint == Endpoint::CHAT_COMPLETIONS) {
        ov::genai::ChatHistory& chatHistory = vlmExecutionContext->apiHandler->getChatHistory();
        ImageHistory& imageHistory = vlmExecutionContext->apiHandler->getImageHistory();
        size_t imageIndex = 0;
        for (auto& image : imageHistory) {
            auto& [chatTurnIndex, imageTensor] = image;
            std::string imageTag = "<ov_genai_image_" + std::to_string(imageIndex++) + ">\n";
            if (chatHistory[chatTurnIndex].find("content") != chatHistory[chatTurnIndex].end()) {
                chatHistory[chatTurnIndex]["content"] = imageTag + chatHistory[chatTurnIndex]["content"];
            } else {
                chatHistory[chatTurnIndex]["content"] = imageTag;
            }
            vlmExecutionContext->inputImages.push_back(imageTensor);
        }
        constexpr bool add_generation_prompt = true;  // confirm it should be hardcoded
        vlmExecutionContext->inputText = properties->tokenizer.apply_chat_template(chatHistory, add_generation_prompt);
    } else if (executionContext->endpoint == Endpoint::COMPLETIONS) {
        vlmExecutionContext->inputText = vlmExecutionContext->apiHandler->getPrompt().value();
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

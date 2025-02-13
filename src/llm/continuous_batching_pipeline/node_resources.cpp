//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "node_resources.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include "../../logging.hpp"
#include "../../status.hpp"

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../mediapipe_internal/mediapipe_utils.hpp"
#include "llm_executor.hpp"
#include "../../http_payload.hpp"
#include "../apis/openai_completions.hpp"
#include "../text_processor.hpp"

namespace ovms {

// TODO: Find better place for this function. It will need to be moved when other pipelines are introduced.
static std::string wrapTextInServerSideEventMessage(const std::string& text) {
    std::stringstream ss;
    ss << "data: " << text << "\n\n";
    return ss.str();
}

void ContinuousBatchingNodeResources::notifyExecutorThread() {
    auto properties = std::static_pointer_cast<ContinuousBatchingNodeProperties>(this->properties);
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Notifying executor thread");
    if (properties->llmExecutorWrapper == nullptr) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "LLMExecutorWrapper is not initialized");
        return;
    }
    properties->llmExecutorWrapper->notifyNewRequestArrived();
}

// Node resources interface start

ovms::Status ContinuousBatchingNodeResources::initialize() {
    auto properties = std::static_pointer_cast<ContinuousBatchingNodeProperties>(this->properties);
    if (properties->pipeline == nullptr) {
        throw std::logic_error("Cannot initiate generation with uninitialized pipeline");
    }
    properties->llmExecutorWrapper = std::make_shared<LLMExecutorWrapper>(properties->pipeline);
    return ovms::StatusCode::OK;
}

std::shared_ptr<BasicExecutionContext> ContinuousBatchingNodeResources::createExecutionContext() {
    return std::make_shared<ContinuousBatchingExecutionContext>();
}

absl::Status ContinuousBatchingNodeResources::createApiHandler(std::shared_ptr<BasicExecutionContext>& executionContext) {
    auto properties = std::static_pointer_cast<ContinuousBatchingNodeProperties>(this->properties);
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingExecutionContext>(executionContext);
    if (properties->pipeline == nullptr) {
        throw std::logic_error("Cannot initiate generation with uninitialized pipeline");
    }
    cbExecutionContext->apiHandler = std::make_shared<OpenAIChatCompletionsHandler>(*cbExecutionContext->payload.parsedJson, 
                                                                                  cbExecutionContext->endpoint, 
                                                                                  std::chrono::system_clock::now(), 
                                                                                  properties->pipeline->get_tokenizer());
    return absl::OkStatus();
}

absl::Status ContinuousBatchingNodeResources::parseRequest(std::shared_ptr<BasicExecutionContext>& executionContext) {
    auto properties = std::static_pointer_cast<ContinuousBatchingNodeProperties>(this->properties);
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingExecutionContext>(executionContext);
    if (cbExecutionContext->apiHandler == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "API handler is not initialized");
    }
    return cbExecutionContext->apiHandler->parseRequest(properties->maxTokensLimit, properties->bestOfLimit, properties->isSpeculativePipeline);
}

absl::Status ContinuousBatchingNodeResources::preparePipelineInput(std::shared_ptr<BasicExecutionContext>& executionContext) {
    auto properties = std::static_pointer_cast<ContinuousBatchingNodeProperties>(this->properties);
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingExecutionContext>(executionContext);
    if (cbExecutionContext->apiHandler == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "API handler is not initialized");
    }

    cbExecutionContext->inputText = std::string();

    switch (cbExecutionContext->endpoint) {
        case Endpoint::CHAT_COMPLETIONS: {
            bool success;
            if (cbExecutionContext->apiHandler->getProcessedJson().size() > 0) {
                // TODO: Extract to node resources interface
                success = TextProcessor::applyChatTemplate(properties->textProcessor, properties->modelsPath, cbExecutionContext->apiHandler->getProcessedJson(), cbExecutionContext->inputText);
            } else {
                // TODO: Extract to node resources interface
                success = TextProcessor::applyChatTemplate(properties->textProcessor, properties->modelsPath, cbExecutionContext->payload.body, cbExecutionContext->inputText);
            }
            if (!success) {
                return absl::Status(absl::StatusCode::kInvalidArgument, cbExecutionContext->inputText);
            }
            if (cbExecutionContext->inputText.size() == 0) {
                return absl::Status(absl::StatusCode::kInvalidArgument, "Final prompt after applying chat template is empty");
            }
            break;
        } 
        case Endpoint::COMPLETIONS: {
            cbExecutionContext->inputText = cbExecutionContext->apiHandler->getPrompt().value();
        }
    }
    return absl::OkStatus();
}

absl::Status ContinuousBatchingNodeResources::schedulePipelineExecution(std::shared_ptr<BasicExecutionContext>& executionContext) {
    auto properties = std::static_pointer_cast<ContinuousBatchingNodeProperties>(this->properties);
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingExecutionContext>(executionContext);
    if (cbExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    if (properties->pipeline == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "Pipeline is not initialized");
    }
    bool encodeAddSpecialTokens = (cbExecutionContext->endpoint == Endpoint::COMPLETIONS);

    ov::Tensor inputIds = properties->pipeline->get_tokenizer().encode(cbExecutionContext->inputText, ov::genai::add_special_tokens(encodeAddSpecialTokens)).input_ids;
    cbExecutionContext->apiHandler->setPromptTokensUsage(inputIds.get_size());
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "{}", getPromptTokensString(inputIds));

    cbExecutionContext->generationHandle = properties->pipeline->add_request(currentRequestId++, // to be removed from API?
                                                                            inputIds,
                                                                            cbExecutionContext->apiHandler->createGenerationConfig());


    cbExecutionContext->payload.client->registerDisconnectionCallback([genHandle = cbExecutionContext->generationHandle]() {
        genHandle->drop();
    });
    
    cbExecutionContext->lastStreamerCallbackOutput = ""; // initialize with empty string
    auto callback = [&lastStreamerCallbackOutput = cbExecutionContext->lastStreamerCallbackOutput](std::string text) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Streamer callback executed with text: [{}]", text);
        lastStreamerCallbackOutput = text;
        return false;
    };

    cbExecutionContext->textStreamer = std::make_shared<ov::genai::TextCallbackStreamer>(properties->pipeline->get_tokenizer(), callback);
    notifyExecutorThread();

    return absl::OkStatus();
}

absl::Status ContinuousBatchingNodeResources::readCompleteExecutionResults(std::shared_ptr<BasicExecutionContext>& executionContext) {
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingExecutionContext>(executionContext);
    if (cbExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }

    std::vector<ov::genai::GenerationOutput> generationOutputs = cbExecutionContext->generationHandle->read_all();
    if (cbExecutionContext->generationHandle->get_status() == ov::genai::GenerationStatus::DROPPED_BY_HANDLE) {
        return absl::CancelledError();
    }
    RET_CHECK(generationOutputs.size() >= 1);
    cbExecutionContext->response = cbExecutionContext->apiHandler->serializeUnaryResponse(generationOutputs);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", cbExecutionContext->response);
    return absl::OkStatus();
}

absl::Status ContinuousBatchingNodeResources::readPartialExecutionResults(std::shared_ptr<BasicExecutionContext>& executionContext) {
    auto cbExecutionContext = std::static_pointer_cast<ContinuousBatchingExecutionContext>(executionContext);
    if (cbExecutionContext->payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Relevant properties read");
    // Streaming scenario
    // Each iteration is single execution of Process() method in the calculator
    if (cbExecutionContext->generationHandle->get_status() == ov::genai::GenerationStatus::DROPPED_BY_HANDLE) {
        return absl::CancelledError();
    }

    if (cbExecutionContext->generationHandle->get_status() == ov::genai::GenerationStatus::RUNNING || cbExecutionContext->generationHandle->can_read()) {
        // Subsequent iteration
        OVMS_PROFILE_SCOPE("Generation of subsequent streaming response");
        ov::genai::GenerationOutputs generationOutputs = cbExecutionContext->generationHandle->read();
        RET_CHECK(generationOutputs.size() == 1);  // TODO: Support multiple generations
        cbExecutionContext->apiHandler->incrementProcessedTokens(generationOutputs.begin()->second.generated_ids.size());
        auto generationOutput = generationOutputs.begin()->second;
        // This loop could be handled in the streamer, but for now we want to keep it identical with GenAI
        // so such change should be done in GenAI first
        std::stringstream ss;
        for (const auto& token : generationOutput.generated_ids) {
            cbExecutionContext->textStreamer->put(token);
            ss << cbExecutionContext->lastStreamerCallbackOutput;
        }
        std::string lastTextChunk = ss.str();
        ov::genai::GenerationFinishReason finishReason = generationOutputs.begin()->second.finish_reason;
        if (finishReason == ov::genai::GenerationFinishReason::NONE) {  // continue
            if (lastTextChunk.size() > 0) {
                cbExecutionContext->response = wrapTextInServerSideEventMessage(cbExecutionContext->apiHandler->serializeStreamingChunk(lastTextChunk, finishReason));
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", cbExecutionContext->response);
            }
            cbExecutionContext->sendLoopbackSignal = true;
        } else {  // finish generation
            OVMS_PROFILE_SCOPE("Generation of last streaming response");
            cbExecutionContext->textStreamer->end();
            // if streamer::put returned a value, streamer::end() result will not contain it, so we add it manually
            if (!cbExecutionContext->lastStreamerCallbackOutput.empty())
                lastTextChunk = lastTextChunk + cbExecutionContext->lastStreamerCallbackOutput;
            cbExecutionContext->response = wrapTextInServerSideEventMessage(cbExecutionContext->apiHandler->serializeStreamingChunk(lastTextChunk, finishReason));
            if (cbExecutionContext->apiHandler->getStreamOptions().includeUsage)
                cbExecutionContext->response += wrapTextInServerSideEventMessage(cbExecutionContext->apiHandler->serializeStreamingUsageChunk());

            cbExecutionContext->response += wrapTextInServerSideEventMessage("[DONE]");

            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", cbExecutionContext->response);
            cbExecutionContext->sendLoopbackSignal = false;
        }
    }
    return absl::OkStatus();
}

}  // namespace ovms

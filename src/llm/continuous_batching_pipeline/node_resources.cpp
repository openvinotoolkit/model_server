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
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Notifying executor thread");
    auto llmExecutorWrapper = getProperty<std::shared_ptr<LLMExecutorWrapper>>("llm_executor_wrapper");
    if (llmExecutorWrapper == nullptr) {
        SPDLOG_LOGGER_ERROR(llm_calculator_logger, "LLMExecutorWrapper is not initialized");
        return;
    }
    llmExecutorWrapper->notifyNewRequestArrived();
}

// Node resources interface start

ovms::Status ContinuousBatchingNodeResources::initialize() {
    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipeline = getProperty<std::shared_ptr<ov::genai::ContinuousBatchingPipeline>>("pipeline");
    if (pipeline == nullptr) {
        throw std::logic_error("Cannot initiate generation with uninitialized pipeline");
    }
    std::shared_ptr<LLMExecutorWrapper> llmExecutorWrapper = std::make_shared<LLMExecutorWrapper>(pipeline);
    setProperty("llm_executor_wrapper", std::move(llmExecutorWrapper));
    return ovms::StatusCode::OK;
}

absl::Status ContinuousBatchingNodeResources::createApiHandler(ov::AnyMap& executionContext) {
    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipeline = getProperty<std::shared_ptr<ov::genai::ContinuousBatchingPipeline>>("pipeline");
    if (pipeline == nullptr) {
        throw std::logic_error("Cannot initiate generation with uninitialized pipeline");
    }
    executionContext["api_handler"] = std::make_shared<OpenAIChatCompletionsHandler>(*executionContext.at("payload").as<ovms::HttpPayload>().parsedJson, 
                                                                                     executionContext.at("endpoint").as<ovms::Endpoint>(), 
                                                                                     std::chrono::system_clock::now(), 
                                                                                     pipeline->get_tokenizer());
    return absl::OkStatus();
}

absl::Status ContinuousBatchingNodeResources::parseRequest(ov::AnyMap& executionContext) {
    std::shared_ptr<OpenAIChatCompletionsHandler>& apiHandler = executionContext.at("api_handler").as<std::shared_ptr<OpenAIChatCompletionsHandler>>();
    if (apiHandler == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "API handler is not initialized");
    }
    uint32_t maxTokensLimit = getProperty<uint32_t>("max_tokens_limit");
    uint32_t bestOfLimit = getProperty<uint32_t>("best_of_limit");
    bool isSpeculativePipeline = hasProperty("draft_model_config");
    return apiHandler->parseRequest(maxTokensLimit, bestOfLimit, isSpeculativePipeline);
}

absl::Status ContinuousBatchingNodeResources::preparePipelineInput(ov::AnyMap& executionContext) {
    std::shared_ptr<OpenAIChatCompletionsHandler>& apiHandler = executionContext.at("api_handler").as<std::shared_ptr<OpenAIChatCompletionsHandler>>();
    if (apiHandler == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "API handler is not initialized");
    }
    ovms::HttpPayload& payload = executionContext.at("payload").as<ovms::HttpPayload>();
    ovms::Endpoint endpoint = executionContext.at("endpoint").as<ovms::Endpoint>();

    std::shared_ptr<TextProcessor> textProcessor = getProperty<std::shared_ptr<TextProcessor>>("text_processor");
    std::string modelsPath = getProperty<std::string>("models_path");

    executionContext["input_text"] = std::string();
    std::string& finalPrompt = executionContext.at("final_prompt").as<std::string>();

    switch (endpoint) {
        case Endpoint::CHAT_COMPLETIONS: {
            bool success;
            if (apiHandler->getProcessedJson().size() > 0) {
                // TODO: Extract to node resources interface
                success = TextProcessor::applyChatTemplate(*textProcessor, modelsPath, apiHandler->getProcessedJson(), finalPrompt);
            } else {
                // TODO: Extract to node resources interface
                success = TextProcessor::applyChatTemplate(*textProcessor, modelsPath, payload.body, finalPrompt);
            }
            if (!success) {
                return absl::Status(absl::StatusCode::kInvalidArgument, finalPrompt);
            }
            if (finalPrompt.size() == 0) {
                return absl::Status(absl::StatusCode::kInvalidArgument, "Final prompt after applying chat template is empty");
            }
            break;
        }
        case Endpoint::COMPLETIONS: {
            finalPrompt = apiHandler->getPrompt().value();
        }
        }
    return absl::OkStatus();
}

absl::Status ContinuousBatchingNodeResources::schedulePipelineExecution(ov::AnyMap& executionContext) {
    ovms::HttpPayload& payload = executionContext.at("payload").as<ovms::HttpPayload>();
    if (payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipeline = getProperty<std::shared_ptr<ov::genai::ContinuousBatchingPipeline>>("pipeline");
    if (pipeline == nullptr) {
        return absl::Status(absl::StatusCode::kInvalidArgument, "Pipeline is not initialized");
    }
    std::shared_ptr<OpenAIChatCompletionsHandler>& apiHandler = executionContext.at("api_handler").as<std::shared_ptr<OpenAIChatCompletionsHandler>>();

    std::string& inputText = executionContext.at("input_text").as<std::string>();
    Endpoint endpoint = executionContext.at("endpoint").as<ovms::Endpoint>();
    bool encodeAddSpecialTokens = (endpoint == Endpoint::COMPLETIONS);

    ov::Tensor inputIds = pipeline->get_tokenizer().encode(inputText, ov::genai::add_special_tokens(encodeAddSpecialTokens)).input_ids;
    apiHandler->setPromptTokensUsage(inputIds.get_size());
    SPDLOG_LOGGER_TRACE(llm_calculator_logger, "{}", getPromptTokensString(inputIds));

    executionContext["generation_handle"] = pipeline->add_request(currentRequestId++, // to be removed from API?
                                                                  inputIds,
                                                                  apiHandler->createGenerationConfig());

    ov::genai::GenerationHandle& generationHandle = executionContext.at("generation_handle").as<ov::genai::GenerationHandle>();

    payload.client->registerDisconnectionCallback([genHandle = generationHandle]() {
        genHandle->drop();
    });
    
    auto callback = [&executionContext](std::string text) {
        SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Streamer callback executed with text: [{}]", text);
        executionContext["last_streamer_callback_output"] = text;
        return false;
    };

    executionContext["text_streamer"] = std::make_shared<ov::genai::TextCallbackStreamer>(pipeline->get_tokenizer(), callback);
    notifyExecutorThread();

    return absl::OkStatus();
}

absl::Status ContinuousBatchingNodeResources::readCompleteExecutionResults(ov::AnyMap& executionContext) {
    ovms::HttpPayload& payload = executionContext.at("payload").as<ovms::HttpPayload>();
    if (payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    std::shared_ptr<OpenAIChatCompletionsHandler>& apiHandler = executionContext.at("api_handler").as<std::shared_ptr<OpenAIChatCompletionsHandler>>();
    ov::genai::GenerationHandle& generationHandle = executionContext.at("generation_handle").as<ov::genai::GenerationHandle>();

    std::vector<ov::genai::GenerationOutput> generationOutputs = generationHandle->read_all();
    if (generationHandle->get_status() == ov::genai::GenerationStatus::DROPPED_BY_HANDLE) {
        return absl::CancelledError();
    }
    RET_CHECK(generationOutputs.size() >= 1);
    executionContext["response"] = apiHandler->serializeUnaryResponse(generationOutputs);
    SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Complete unary response: {}", executionContext.at("response").as<std::string>());
    return absl::OkStatus();
}

absl::Status ContinuousBatchingNodeResources::readPartialExecutionResults(ov::AnyMap& executionContext) {
    ovms::HttpPayload& payload = executionContext.at("payload").as<ovms::HttpPayload>();
    if (payload.client->isDisconnected()) {
        return absl::CancelledError();
    }
    std::shared_ptr<OpenAIChatCompletionsHandler>& apiHandler = executionContext.at("api_handler").as<std::shared_ptr<OpenAIChatCompletionsHandler>>();
    ov::genai::GenerationHandle& generationHandle = executionContext.at("generation_handle").as<ov::genai::GenerationHandle>();
    std::shared_ptr<ov::genai::TextCallbackStreamer>& streamer = executionContext.at("text_streamer").as<std::shared_ptr<ov::genai::TextCallbackStreamer>>();

    // Streaming scenario
    // Each iteration is single execution of Process() method in the calculator
    if (generationHandle->get_status() == ov::genai::GenerationStatus::DROPPED_BY_HANDLE) {
        return absl::CancelledError();
    }

    if (generationHandle->get_status() == ov::genai::GenerationStatus::RUNNING || generationHandle->can_read()) {
        // Subsequent iteration
        OVMS_PROFILE_SCOPE("Generation of subsequent streaming response");
        ov::genai::GenerationOutputs generationOutputs = generationHandle->read();
        RET_CHECK(generationOutputs.size() == 1);  // TODO: Support multiple generations
        apiHandler->incrementProcessedTokens(generationOutputs.begin()->second.generated_ids.size());
        auto generationOutput = generationOutputs.begin()->second;
        // This loop could be handled in the streamer, but for now we want to keep it identical with GenAI
        // so such change should be done in GenAI first
        std::string& lastStreamerCallbackOutput = executionContext.at("last_streamer_callback_output").as<std::string>();
        std::stringstream ss;
        for (const auto& token : generationOutput.generated_ids) {
            streamer->put(token);
            ss << lastStreamerCallbackOutput;
        }
        std::string lastTextChunk = ss.str();
        ov::genai::GenerationFinishReason finishReason = generationOutputs.begin()->second.finish_reason;
        if (finishReason == ov::genai::GenerationFinishReason::NONE) {  // continue
            if (lastTextChunk.size() > 0) {
                executionContext["response"] = wrapTextInServerSideEventMessage(apiHandler->serializeStreamingChunk(lastTextChunk, finishReason));
                SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated subsequent streaming response: {}", executionContext.at("response").as<std::string>());
            }
            executionContext["send_loopback_signal"] = true;
        } else {  // finish generation
            OVMS_PROFILE_SCOPE("Generation of last streaming response");
            streamer->end();
            // if streamer::put returned a value, streamer::end() result will not contain it, so we add it manually
            if (!lastStreamerCallbackOutput.empty())
                lastTextChunk = lastTextChunk + lastStreamerCallbackOutput;
            executionContext["response"] = wrapTextInServerSideEventMessage(apiHandler->serializeStreamingChunk(lastTextChunk, finishReason));
            std::string& response = executionContext.at("response").as<std::string>();
            if (apiHandler->getStreamOptions().includeUsage)
                response += wrapTextInServerSideEventMessage(apiHandler->serializeStreamingUsageChunk());

            response += wrapTextInServerSideEventMessage("[DONE]");

            SPDLOG_LOGGER_DEBUG(llm_calculator_logger, "Generated complete streaming response: {}", response);
            executionContext["send_loopback_signal"] = false;
        }
    }
    return absl::OkStatus();
}

}  // namespace ovms

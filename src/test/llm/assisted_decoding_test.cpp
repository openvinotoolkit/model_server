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
#include <atomic>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#pragma warning(pop)

#include "../../http_rest_api_handler.hpp"
#include "../../http_status_code.hpp"
#include "../../json_parser.hpp"
#include "../../llm/apis/openai_completions.hpp"
#include "../../llm/language_model/continuous_batching/servable.hpp"
#include "../../llm/language_model/continuous_batching/llm_executor.hpp"
#include "../../server.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "../test_http_utils.hpp"
#include "../test_utils.hpp"

using namespace ovms;

class AssistedDecodingPipelinesHttpTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;
    const std::string endpointChatCompletions = "/v3/chat/completions";
    const std::string endpointCompletions = "/v3/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::string response;
    rapidjson::Document parsedResponse;
    ovms::HttpResponseComponents responseComponents;
    static std::shared_ptr<ov::genai::ContinuousBatchingPipeline> cbPipe;
    static std::shared_ptr<LLMExecutorWrapper> llmExecutorWrapper;
    ov::genai::GenerationConfig config;
    std::vector<std::string> expectedMessages;

    static void SetUpTestSuite() {
        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, getGenericFullPathForSrcTest("/ovms/src/test/llm/assisted_decoding_config.json").c_str(), 60);

        try {
            plugin_config_t tokenizerPluginConfig = {};
            std::string device = "CPU";
            ov::genai::SchedulerConfig schedulerConfig;
            schedulerConfig.max_num_batched_tokens = 256;
            schedulerConfig.cache_size = 1;
            schedulerConfig.dynamic_split_fuse = true;
            schedulerConfig.max_num_seqs = 256;
            plugin_config_t pluginConfig;
            // Setting precision to f32 fails on SPR hosts - to be investigated??
            JsonParser::parsePluginConfig("{\"INFERENCE_PRECISION_HINT\":\"f32\"}", pluginConfig);
            cbPipe = std::make_shared<ov::genai::ContinuousBatchingPipeline>(getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m"), schedulerConfig, device, pluginConfig, tokenizerPluginConfig);
            llmExecutorWrapper = std::make_shared<LLMExecutorWrapper>(cbPipe);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Error during llm node initialization for models_path exception: {}", e.what());
        } catch (...) {
            SPDLOG_ERROR("Error during llm node initialization for models_path");
        }
    }

    int generateExpectedText(std::string prompt, bool addSpecialTokens = true) {
        try {
            ov::Tensor promptIds = cbPipe->get_tokenizer().encode(prompt, ov::genai::add_special_tokens(addSpecialTokens)).input_ids;
            std::cout << "Generated prompt ids: " << getPromptTokensString(promptIds) << std::endl;
            auto generationHandle = cbPipe->add_request(
                currentRequestId++,
                promptIds,
                config);
            if (generationHandle == nullptr) {
                return -1;
            }
            llmExecutorWrapper->notifyNewRequestArrived();
            std::vector<ov::genai::GenerationOutput> generationOutput = generationHandle->read_all();
            std::sort(generationOutput.begin(), generationOutput.end(), [=](ov::genai::GenerationOutput& r1, ov::genai::GenerationOutput& r2) {
                return r1.score > r2.score;
            });
            size_t i = 0;
            std::shared_ptr<ov::genai::Tokenizer> tokenizer = std::make_shared<ov::genai::Tokenizer>(cbPipe->get_tokenizer());
            for (ov::genai::GenerationOutput& out : generationOutput) {
                if (i >= config.num_return_sequences)
                    break;
                i++;
                std::vector<int64_t> tokens = out.generated_ids;
                SPDLOG_LOGGER_TRACE(llm_calculator_logger, "Generated tokens: {}", tokens);
                std::string completion = tokenizer->decode(tokens);
                expectedMessages.emplace_back(completion);
            }
        } catch (ov::AssertFailure& e) {
            return -1;
        } catch (...) {
            return -1;
        }
        return 0;
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));  // make the streaming flow sequential
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointChatCompletions, headers), ovms::StatusCode::OK);
    }

    static void TearDownTestSuite() {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }

    void TearDown() {
        handler.reset();
    }
};

std::shared_ptr<ov::genai::ContinuousBatchingPipeline> AssistedDecodingPipelinesHttpTest::cbPipe;
std::shared_ptr<LLMExecutorWrapper> AssistedDecodingPipelinesHttpTest::llmExecutorWrapper;
std::unique_ptr<std::thread> AssistedDecodingPipelinesHttpTest::t;

// Speculative decoding

TEST_F(AssistedDecodingPipelinesHttpTest, unaryCompletionsJsonSpeculativeDecoding) {
    // Generate reference from the base model (unassisted generation)
    config.max_new_tokens = 10;
    config.temperature = 0;
    ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
    ASSERT_EQ(config.num_return_sequences, expectedMessages.size());

    // Static number of candidates
    std::string requestBody = R"(
        {
            "model": "lm_cb_speculative",
            "stream": false,
            "temperature" : 0,
            "max_tokens": 10,
            "prompt": "What is OpenVINO?",
            "num_assistant_tokens": 3
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    auto& choice = parsedResponse["choices"].GetArray()[0];
    ASSERT_TRUE(choice["text"].IsString());
    EXPECT_STREQ(choice["text"].GetString(), expectedMessages[0].c_str());

    // Dynamic number of candidates
    requestBody = R"(
        {
            "model": "lm_cb_speculative",
            "stream": false,
            "temperature": 0,
            "max_tokens": 10,
            "prompt": "What is OpenVINO?",
            "assistant_confidence_threshold": 0.4
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    choice = parsedResponse["choices"].GetArray()[0];
    ASSERT_TRUE(choice["text"].IsString());
    EXPECT_STREQ(choice["text"].GetString(), expectedMessages[0].c_str());
}

TEST_F(AssistedDecodingPipelinesHttpTest, unaryChatCompletionsJsonSpeculativeDecoding) {
    // Generate reference from the base model (unassisted generation)
    config.max_new_tokens = 8;
    config.temperature = 0;
    ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
    ASSERT_EQ(config.num_return_sequences, expectedMessages.size());

    // Static number of candidates
    std::string requestBody = R"(
        {
            "model": "lm_cb_speculative",
            "stream": false,
            "temperature": 0,
            "max_tokens": 8,
            "num_assistant_tokens": 3,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    auto& choice = parsedResponse["choices"].GetArray()[0];
    ASSERT_TRUE(choice["message"].IsObject());
    ASSERT_TRUE(choice["message"]["content"].IsString());
    ASSERT_TRUE(choice["finish_reason"].IsString());
    ASSERT_FALSE(choice["logprobs"].IsObject());
    ASSERT_EQ(choice["message"]["content"].GetString(), expectedMessages[0]);

    // Dynamic number of candidates
    requestBody = R"(
        {
            "model": "lm_cb_speculative",
            "stream": false,
            "temperature": 0,
            "max_tokens": 8,
            "assistant_confidence_threshold": 0.4,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    choice = parsedResponse["choices"].GetArray()[0];
    ASSERT_TRUE(choice["message"].IsObject());
    ASSERT_TRUE(choice["message"]["content"].IsString());
    ASSERT_TRUE(choice["finish_reason"].IsString());
    ASSERT_FALSE(choice["logprobs"].IsObject());
    ASSERT_EQ(choice["message"]["content"].GetString(), expectedMessages[0]);
}

TEST_F(AssistedDecodingPipelinesHttpTest, speculativeDecodingExclusiveParametersProvided) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_speculative",
            "prompt": "hello",
            "num_assistant_tokens": 5,
            "assistant_confidence_threshold": 0.5
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(AssistedDecodingPipelinesHttpTest, speculativeDecodingExclusiveParametersProvidedChat) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_speculative",
            "messages": [{"content": "def"}],
            "num_assistant_tokens": 5,
            "assistant_confidence_threshold": 0.5
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

// Prompt lookup decoding

TEST_F(AssistedDecodingPipelinesHttpTest, unaryCompletionsJsonPromptLookupDecoding) {
    // Generate reference from the base model (unassisted generation)
    config.max_new_tokens = 10;
    config.temperature = 0;
    ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
    ASSERT_EQ(config.num_return_sequences, expectedMessages.size());

    std::string requestBody = R"(
        {
            "model": "lm_cb_prompt_lookup",
            "stream": false,
            "temperature" : 0,
            "max_tokens": 10,
            "prompt": "What is OpenVINO?",
            "num_assistant_tokens": 5,
            "max_ngram_size": 3
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    auto& choice = parsedResponse["choices"].GetArray()[0];
    ASSERT_TRUE(choice["text"].IsString());
    EXPECT_STREQ(choice["text"].GetString(), expectedMessages[0].c_str());
}

TEST_F(AssistedDecodingPipelinesHttpTest, unaryChatCompletionsJsonPromptLookupDecoding) {
    // Generate reference from the base model (unassisted generation)
    config.max_new_tokens = 10;
    config.temperature = 0;
    ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
    ASSERT_EQ(config.num_return_sequences, expectedMessages.size());

    auto requestBody = R"(
        {
            "model": "lm_cb_speculative",
            "stream": false,
            "temperature": 0,
            "max_tokens": 10,
            "num_assistant_tokens": 5,
            "max_ngram_size": 3,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    auto& choice = parsedResponse["choices"].GetArray()[0];
    ASSERT_TRUE(choice["message"].IsObject());
    ASSERT_TRUE(choice["message"]["content"].IsString());
    ASSERT_TRUE(choice["finish_reason"].IsString());
    ASSERT_FALSE(choice["logprobs"].IsObject());
    ASSERT_EQ(choice["message"]["content"].GetString(), expectedMessages[0]);
}

// Consider parametrization of negative tests with request body and endpoint as parameters
TEST_F(AssistedDecodingPipelinesHttpTest, promptLookupDecodingMissingParameterCompletions) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_prompt_lookup",
            "prompt": "def",
            "num_assistant_tokens": 5
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);

    requestBody = R"(
        {
            "model": "lm_cb_prompt_lookup",
            "prompt": "def",
            "max_ngram_size": 3
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(AssistedDecodingPipelinesHttpTest, promptLookupDecodingMissingParameterChatCompletions) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_prompt_lookup",
            "messages": [{"content": "def"}],
            "num_assistant_tokens": 5
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);

    requestBody = R"(
        {
            "model": "lm_cb_prompt_lookup",
            "messages": [{"content": "def"}],
            "max_ngram_size": 3
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

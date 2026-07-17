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
#include <atomic>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>

#include <fmt/ranges.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>
#if (PYTHON_DISABLE == 0)
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#pragma warning(pop)
#endif

#include "../../http_rest_api_handler.hpp"
#include "../../config.hpp"
#include "../../http_status_code.hpp"
#include "../../json_parser.hpp"
#include "../../llm/apis/openai_completions.hpp"
#include "../../llm/io_processing/base_generation_config_builder.hpp"
#include "../../llm/language_model/continuous_batching/llm_executor.hpp"
#include "../../llm/language_model/continuous_batching/servable.hpp"
#include "../../llm/servable.hpp"
#include "../../llm/servable_initializer.hpp"
#include "../../llm/text_utils.hpp"
#include "../../mediapipe_internal/mediapipefactory.hpp"
#include "../../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../../ov_utils.hpp"
#include "../../server.hpp"
#include "src/graph_export/graph_export.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "../constructor_enabled_model_manager.hpp"
#include "../platform_utils.hpp"
#include "../test_http_utils.hpp"
#include "../test_utils.hpp"
#include "src/test/environment.hpp"

using namespace ovms;

struct TestParameters {
    std::string modelName;
    bool generateExpectedOutput;
    bool checkLogprobs;
    bool checkFinishReason;
    bool testSpeculativeDecoding;
    bool checkHandshakeChunk;
};

class LLMFlowHttpTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpointChatCompletions = "/v1/chat/completions";
    const std::string endpointCompletions = "/v1/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
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
        ::SetUpServer(t, server, port, getGenericFullPathForSrcTest("/ovms/src/test/llm/config.json").c_str(), 60);

        try {
            plugin_config_t tokenizerPluginConfig = {};
            std::string device = "CPU";
            ov::genai::SchedulerConfig schedulerConfig;
            schedulerConfig.max_num_batched_tokens = 256;
            schedulerConfig.cache_size = 1;
            schedulerConfig.dynamic_split_fuse = true;
            schedulerConfig.max_num_seqs = 256;
            plugin_config_t pluginConfig;
            // Setting precision to f32 fails on SPR hosts - to be investigated
            // JsonParser::parsePluginConfig("{\"INFERENCE_PRECISION_HINT\":\"f32\"}", pluginConfig);
            cbPipe = std::make_shared<ov::genai::ContinuousBatchingPipeline>(getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/HuggingFaceTB/SmolLM2-360M-Instruct"), schedulerConfig, device, pluginConfig, tokenizerPluginConfig);
            llmExecutorWrapper = std::make_shared<LLMExecutorWrapper>(cbPipe);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Error during llm node initialization for models_path exception: {}", e.what());
        } catch (...) {
            SPDLOG_ERROR("Error during llm node initialization for models_path");
        }
    }

    int generateExpectedText(std::string prompt, bool addSpecialTokens = true, bool applyChatTemplate = false) {
        try {
            if (applyChatTemplate) {
                ov::genai::ChatHistory chatHistory({{{"role", "user"}, {"content", prompt}}});
                prompt = cbPipe->get_tokenizer().apply_chat_template(chatHistory, true);
            }
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
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));  // make the streaming flow sequential
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointChatCompletions, headers), ovms::StatusCode::OK);
    }

    static void TearDownTestSuite() {
        llmExecutorWrapper.reset();
        cbPipe.reset();
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }

    void TearDown() {
        handler.reset();
    }
};
std::shared_ptr<ov::genai::ContinuousBatchingPipeline> LLMFlowHttpTest::cbPipe;
std::shared_ptr<LLMExecutorWrapper> LLMFlowHttpTest::llmExecutorWrapper;
std::unique_ptr<std::thread> LLMFlowHttpTest::t;

class LLMFlowHttpQueueGraphTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;
    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpointChatCompletions = "/v1/chat/completions";
    const std::string endpointCompletions = "/v1/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    rapidjson::Document parsedResponse;
    ovms::HttpResponseComponents responseComponents;

    static void SetUpTestSuite() {
        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, getGenericFullPathForSrcTest("/ovms/src/test/llm/config_queue.json").c_str(), 60);
    }

    static void TearDownTestSuite() {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointCompletions, headers), ovms::StatusCode::OK);
    }

    void TearDown() {
        handler.reset();
    }
};

std::unique_ptr<std::thread> LLMFlowHttpQueueGraphTest::t;

// --------------------------------------- OVMS LLM nodes tests

/* 
// TODO: Move this test to OpenAiJsonResponse tests
TEST(OpenAiApiHandlerTest, writeLogprobs) {
    // TODO: remove that skip
    GTEST_SKIP();
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    std::vector<float> inputs{-0.5, -100, 0, 5};
    std::vector<std::string> expected{"-0.5", "-100.0", "0.0", "null"};
    for (size_t i = 0; i < inputs.size(); i++) {
        OpenAIChatCompletionsHandler::writeLogprob(writer, inputs[i]);
        EXPECT_EQ(buffer.GetString(), expected[i]);
        buffer.Clear();
    }
}
*/

// Reusable helper: asserts that a streaming chat completion chunk is the initial
// initial empty message with role:assistant and content:null.
inline void assertInitialStreamChatCompletionChunk(const std::string& response, const std::string& expectedModel) {
    const std::string dataPrefix = "data:";
    ASSERT_GE(response.size(), dataPrefix.size());
    ASSERT_EQ(response.substr(0, dataPrefix.size()), dataPrefix);
    size_t pos = response.find("\n");
    ASSERT_NE(pos, std::string::npos);
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.substr(dataPrefix.size(), pos - dataPrefix.size()).c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_TRUE(d.HasMember("choices"));
    ASSERT_TRUE(d["choices"].IsArray());
    ASSERT_EQ(d["choices"].Size(), 1);
    const auto& choice = d["choices"][0];
    ASSERT_EQ(choice["index"].GetInt(), 0);
    ASSERT_TRUE(choice["finish_reason"].IsNull());
    ASSERT_TRUE(choice["delta"].IsObject());
    EXPECT_STREQ(choice["delta"]["role"].GetString(), "assistant");
    ASSERT_TRUE(choice["delta"]["content"].IsNull());
    ASSERT_TRUE(d.HasMember("created"));
    ASSERT_TRUE(d["created"].IsInt());
    EXPECT_STREQ(d["model"].GetString(), expectedModel.c_str());
    EXPECT_STREQ(d["object"].GetString(), "chat.completion.chunk");
}

class LLMFlowHttpTestParameterized : public LLMFlowHttpTest, public ::testing::WithParamInterface<TestParameters> {};

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJson) {
    auto params = GetParam();
    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.temperature = 0;
    if (params.generateExpectedOutput) {
        ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
        ASSERT_EQ(config.num_return_sequences, expectedMessages.size());
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "temperature": 0,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of such if...else...
    if (params.modelName.find("vlm") == std::string::npos) {
        ASSERT_EQ(
            handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
            ovms::StatusCode::OK);
        parsedResponse.Parse(response.c_str());
        ASSERT_TRUE(parsedResponse["choices"].IsArray());
        ASSERT_EQ(parsedResponse["choices"].Size(), 1);
        int i = 0;
        for (auto& choice : parsedResponse["choices"].GetArray()) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
            ASSERT_FALSE(choice["logprobs"].IsObject());
            ASSERT_TRUE(choice["text"].IsString());
            if (params.generateExpectedOutput) {
                EXPECT_STREQ(choice["text"].GetString(), expectedMessages[i].c_str());
            }
            ASSERT_EQ(choice["index"], i++);
        }

        ASSERT_TRUE(parsedResponse["usage"].IsObject());
        ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
        ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
        ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
        ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
        EXPECT_STREQ(parsedResponse["model"].GetString(), params.modelName.c_str());
        EXPECT_STREQ(parsedResponse["object"].GetString(), "text_completion");
    } else {  // Completions endpoint not supported for VLM servable
        ASSERT_EQ(
            handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
            ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    }
}

TEST_F(LLMFlowHttpQueueGraphTest, unaryCompletionsJsonQueueGraph) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular_queue",
            "stream": false,
            "seed" : 1,
            "best_of": 16,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["text"].IsString());
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5);
    EXPECT_STREQ(parsedResponse["model"].GetString(), "lm_cb_regular_queue");
    EXPECT_STREQ(parsedResponse["object"].GetString(), "text_completion");
}

TEST_F(LLMFlowHttpQueueGraphTest, unaryChatCompletionsJsonQueueGraph) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular_queue",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5);
    EXPECT_STREQ(parsedResponse["model"].GetString(), "lm_cb_regular_queue");
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

TEST_F(LLMFlowHttpQueueGraphTest, streamChatCompletionsQueueGraph) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular_queue",
            "stream": true,
            "seed" : 1,
            "max_tokens": 5,
            "ignore_eos": true,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";
    ON_CALL(*writer, PartialReply).WillByDefault([this](std::string response) {
        rapidjson::Document d;
        std::string dataPrefix = "data:";
        ASSERT_STREQ(response.substr(0, dataPrefix.size()).c_str(), dataPrefix.c_str());
        size_t pos = response.find("\n");
        ASSERT_NE(pos, response.npos);
        rapidjson::ParseResult parsingSucceeded = d.Parse(response.substr(dataPrefix.size(), (pos - dataPrefix.size())).c_str());
        ASSERT_EQ(parsingSucceeded.Code(), 0);
        ASSERT_TRUE(d["choices"].IsArray());
        ASSERT_EQ(d["choices"].Size(), 1);
        int i = 0;
        for (auto& choice : d["choices"].GetArray()) {
            if (choice["finish_reason"].IsString()) {
                EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
            } else {
                ASSERT_TRUE(choice["finish_reason"].IsNull());
            }
            ASSERT_EQ(choice["index"], i++);
            ASSERT_TRUE(choice["delta"].IsObject());
            // First chunk may have null content (role announcement)
            ASSERT_TRUE(choice["delta"]["content"].IsString() || choice["delta"]["content"].IsNull());
        }
        EXPECT_STREQ(d["model"].GetString(), "lm_cb_regular_queue");
        EXPECT_STREQ(d["object"].GetString(), "chat.completion.chunk");
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

// Test that verifies graph reuse works correctly with queue size 1
// Sends 2 sequential requests to ensure the same graph instance is reused
TEST_F(LLMFlowHttpQueueGraphTest, queueGraphReuseTwoRequests) {
    std::string requestBody = R"(
        {
            "model": "lm_cb_regular_queue",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    // First request
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    ASSERT_TRUE(parsedResponse["choices"].GetArray()[0]["text"].IsString());

    // Second request - reuses the same graph from the queue
    // This validates that timestamp increment works for graph reuse
    response.clear();
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    ASSERT_TRUE(parsedResponse["choices"].GetArray()[0]["text"].IsString());
    // Note: Responses may differ due to KV cache state despite same seed
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJsonEchoWithCompletion) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos || params.modelName.find("legacy") != std::string::npos) {
        // VLM does not support completions endpoint and legacy servables do not support echo
        GTEST_SKIP();
    }

    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.temperature = 0;
    config.echo = true;
    if (params.generateExpectedOutput) {
        ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
        ASSERT_EQ(config.num_return_sequences, expectedMessages.size());
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "temperature": 0,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?",
            "echo": true
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (params.checkFinishReason) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
        }
        if (params.checkLogprobs) {
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_TRUE(choice["text"].IsString());
        if (params.generateExpectedOutput) {
            EXPECT_STREQ(choice["text"].GetString(), expectedMessages[i].c_str());
        }
        EXPECT_TRUE(std::string(choice["text"].GetString()).find("What is OpenVINO?") != std::string::npos);
        EXPECT_EQ(std::string(choice["text"].GetString()).rfind("What is OpenVINO?", 0), 0);  // Check if prompt is at the beginning
        ASSERT_EQ(choice["index"], i++);
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), params.modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "text_completion");
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsEchoWithCompletion) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos || params.modelName.find("legacy") != std::string::npos) {
        // VLM does not support completions endpoint and legacy servables do not support echo
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "seed" : 1,
            "max_tokens": 10,
            "echo": true,
            "prompt": "What is OpenVINO?"
        }
    )";
    std::vector<std::string> chunks;
    ON_CALL(*writer, PartialReply).WillByDefault([this, &chunks, &params](std::string response) {
        // A single PartialReply may contain multiple SSE events (e.g. all echo
        // tokens in the first call).  Iterate every event and collect text chunks.
        const std::string eventSep = "\n\n";
        const std::string dataPrefix = "data:";
        ASSERT_STREQ(response.substr(0, dataPrefix.size()).c_str(), dataPrefix.c_str());
        size_t start = 0;
        while (start < response.size()) {
            const size_t eventEnd = response.find(eventSep, start);
            if (eventEnd == std::string::npos)
                break;
            const std::string event = response.substr(start, eventEnd - start);
            start = eventEnd + eventSep.size();
            if (event.size() < dataPrefix.size())
                continue;
            const std::string body = event.substr(dataPrefix.size());
            if (body.find("[DONE]") != std::string::npos)
                break;
            rapidjson::Document d;
            rapidjson::ParseResult pr = d.Parse(body.c_str());
            ASSERT_EQ(pr.Code(), 0);
            ASSERT_TRUE(d["choices"].IsArray());
            ASSERT_EQ(d["choices"].Size(), 1);
            int i = 0;
            for (auto& choice : d["choices"].GetArray()) {
                ASSERT_EQ(choice["index"], i++);
                if (params.checkLogprobs) {
                    ASSERT_FALSE(choice["logprobs"].IsObject());
                }
                ASSERT_TRUE(choice["text"].IsString());
                chunks.push_back(std::string(choice["text"].GetString()));
            }
            EXPECT_STREQ(d["model"].GetString(), params.modelName.c_str());
            EXPECT_STREQ(d["object"].GetString(), "text_completion.chunk");
        }
    });

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_GT(chunks.size(), 1);
    std::string combined;
    for (const auto& chunk : chunks)
        combined += chunk;
    EXPECT_EQ(combined.rfind("What is OpenVINO?", 0), 0) << "Expected output to start with echoed prompt, got: " << combined;
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJsonEchoOnly) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos || params.modelName.find("legacy") != std::string::npos) {
        // VLM does not support completions endpoint and legacy servables do not support echo
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 0,
            "prompt": "What is OpenVINO?",
            "echo": true,
            "logprobs": 1
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (params.checkFinishReason) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
        }

        if (params.checkLogprobs) {
            ASSERT_TRUE(choice["logprobs"].IsObject());
            ASSERT_TRUE(choice["logprobs"].GetObject()["token_logprobs"].IsArray());
            for (size_t i = 0; i < choice["logprobs"].GetObject()["token_logprobs"].Size(); ++i) {
                auto& logprob = choice["logprobs"].GetObject()["token_logprobs"][i];
                if (i == 0) {
                    ASSERT_TRUE(logprob.IsNull());
                } else {
                    ASSERT_TRUE(logprob.IsFloat());
                    ASSERT_LT(logprob.GetFloat(), 0);
                }
            }
        }

        ASSERT_TRUE(choice["text"].IsString());
        EXPECT_STREQ(choice["text"].GetString(), "What is OpenVINO?");
        ASSERT_EQ(choice["index"], i++);
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 0 /* max_tokens */);
    if (params.checkLogprobs) {
        ASSERT_EQ(parsedResponse["usage"].GetObject()["prompt_tokens"].GetInt(), parsedResponse["choices"].GetArray()[0]["logprobs"].GetObject()["token_logprobs"].Size());
    }
    EXPECT_STREQ(parsedResponse["model"].GetString(), params.modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "text_completion");
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsEchoOnly) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos || params.modelName.find("legacy") != std::string::npos) {
        // VLM does not support completions endpoint and legacy servables do not support echo
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "seed" : 1,
            "max_tokens": 0,
            "echo": true,
            "prompt": "What is OpenVINO?"
        }
    )";

    if (params.modelName.find("legacy") == std::string::npos) {
        // Echo tokens are streamed one SSE event per token (through the normal
        // delay-buffer path), so a single PartialReply may contain multiple events.
        // Accumulate all text chunks and verify their concatenation equals the prompt.
        std::string echoText;
        std::string lastFinishReason;
        EXPECT_CALL(*writer, PartialReply(::testing::_)).WillOnce([this, &params, &echoText, &lastFinishReason](std::string response) {
            const std::string eventSep = "\n\n";
            const std::string dataPrefix = "data:";
            ASSERT_STREQ(response.substr(0, dataPrefix.size()).c_str(), dataPrefix.c_str());
            size_t start = 0;
            while (start < response.size()) {
                const size_t eventEnd = response.find(eventSep, start);
                if (eventEnd == std::string::npos)
                    break;
                const std::string event = response.substr(start, eventEnd - start);
                start = eventEnd + eventSep.size();
                if (event.size() < dataPrefix.size())
                    continue;
                const std::string body = event.substr(dataPrefix.size());
                if (body.find("[DONE]") != std::string::npos)
                    break;
                rapidjson::Document d;
                rapidjson::ParseResult pr = d.Parse(body.c_str());
                ASSERT_EQ(pr.Code(), 0);
                ASSERT_TRUE(d["choices"].IsArray());
                ASSERT_EQ(d["choices"].Size(), 1);
                for (auto& choice : d["choices"].GetArray()) {
                    if (params.checkLogprobs) {
                        ASSERT_FALSE(choice["logprobs"].IsObject());
                    }
                    ASSERT_TRUE(choice["text"].IsString());
                    echoText += choice["text"].GetString();
                    if (choice.HasMember("finish_reason") && choice["finish_reason"].IsString()) {
                        lastFinishReason = choice["finish_reason"].GetString();
                    }
                }
                EXPECT_STREQ(d["model"].GetString(), params.modelName.c_str());
                EXPECT_STREQ(d["object"].GetString(), "text_completion.chunk");
            }
        });
        ASSERT_EQ(
            handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
            ovms::StatusCode::PARTIAL_END);
        if (params.checkFinishReason) {
            EXPECT_STREQ(lastFinishReason.c_str(), "length");
        }
        EXPECT_EQ(echoText, "What is OpenVINO?");
    } else {
        // In legacy servable streaming with echo, prompt can be sent back in multiple chunks
        std::vector<std::string> responses;
        EXPECT_CALL(*writer, PartialReply(::testing::_))
            .WillRepeatedly([this, &responses](std::string response) {
                responses.push_back(response);
            });
        EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
        ASSERT_EQ(
            handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
            ovms::StatusCode::PARTIAL_END);
        std::string mergedContent;
        for (const auto& response : responses) {
            std::regex content_regex("\"text\":\"(.*?)\"");
            std::smatch match;
            if (std::regex_search(response, match, content_regex)) {
                mergedContent += match[1].str();
            }
        }
        EXPECT_EQ(mergedContent, "What is OpenVINO?");
    }
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJsonFinishReasonLength) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "ignore_eos": true,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        if (params.checkFinishReason) {
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
        }
        ASSERT_EQ(choice["index"], i++);
        if (params.checkLogprobs) {
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_TRUE(choice["text"].IsString());
    }
    ASSERT_EQ(parsedResponse["model"], params.modelName.c_str());
    ASSERT_EQ(parsedResponse["object"], "text_completion");
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJsonSingleStopString) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "temperature": 0,
            "ignore_eos": false,
            "max_tokens": 1000,
            "stop": ".",
            "include_stop_str_in_output": true,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        if (params.checkFinishReason) {
            EXPECT_STREQ(choice["finish_reason"].GetString(), "stop");
        }
        ASSERT_EQ(choice["index"], i++);
        if (params.checkLogprobs) {
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_TRUE(choice["text"].IsString());
        auto text_size = std::string(choice["text"].GetString()).size();
        ASSERT_EQ(choice["text"].GetString()[text_size - 1], '.');
    }
    ASSERT_EQ(parsedResponse["model"], params.modelName.c_str());
    ASSERT_EQ(parsedResponse["object"], "text_completion");
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJsonSpaceStopString) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "ignore_eos": false,
            "max_tokens": 1000,
            "temperature": 0,
            "stop": " ",
            "include_stop_str_in_output": true,
            "prompt": "                                   |                                |                             |  "
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse.HasMember("choices"));
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    ASSERT_TRUE(parsedResponse["choices"].GetArray()[0].HasMember("text"));
    ASSERT_TRUE(parsedResponse["choices"].GetArray()[0]["text"].IsString());
    ASSERT_EQ(parsedResponse["choices"].GetArray()[0]["text"].GetString(), std::string{""});
}

TEST_P(LLMFlowHttpTestParameterized, defaultRoutingInvalidJson) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            INVALID JSON
        }
    )";

    const std::string uriThatMatchesGraphName = std::string("/v1/") + params.modelName;

    headers.clear();  // no sign of application/json
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", uriThatMatchesGraphName, headers), ovms::StatusCode::OK);
    ASSERT_EQ(
        handler->dispatchToProcessor(uriThatMatchesGraphName, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJsonNFail) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "temperature": 0,
            "max_tokens": -5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJsonN) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.temperature = 0;
    config.echo = false;
    if (params.generateExpectedOutput) {
        ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
        ASSERT_EQ(config.num_return_sequences, expectedMessages.size());
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "temperature": 0,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        if (params.checkFinishReason) {
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
        }
        if (params.checkLogprobs) {
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_TRUE(choice["text"].IsString());
        if (params.generateExpectedOutput) {
            EXPECT_STREQ(choice["text"].GetString(), expectedMessages[i].c_str());
        }
        ASSERT_EQ(choice["index"], i++);
    }
    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), params.modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "text_completion");
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsJsonNFail) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "temperature": 0,
            "max_tokens": -5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsJsonN) {
    GTEST_SKIP();  // TODO: Temporary skip to synchronize CI workers
    auto params = GetParam();
    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.temperature = 0;
    config.echo = false;
    if (params.generateExpectedOutput) {
        ASSERT_EQ(generateExpectedText("What is OpenVINO?", false, true), 0);
        ASSERT_EQ(config.num_return_sequences, expectedMessages.size());
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "temperature": 0,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        if (params.checkFinishReason) {
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
        }
        if (params.checkLogprobs) {
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        if (params.generateExpectedOutput) {
            ASSERT_EQ(choice["message"]["content"].GetString(), expectedMessages[i]);
        }
        ASSERT_EQ(choice["index"], i++);
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), params.modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

TEST_P(LLMFlowHttpTestParameterized, KFSApiRequestToChatCompletionsGraph) {
    auto params = GetParam();
    std::string requestBody = R"({
    "inputs" : [
        {
        "name" : "input",
        "shape" : [ 2, 2 ],
        "datatype" : "UINT32",
        "data" : [ 1, 2, 3, 4 ]
        }
    ]
    })";
    std::unordered_map<std::string, std::string> headers;
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", "/v2/models/" + params.modelName + "/versions/1/infer", headers), ovms::StatusCode::OK);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsJson) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "temperature": 0,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (params.checkFinishReason) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
        }
        if (params.checkLogprobs) {
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_EQ(choice["index"], i++);
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), params.modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsSkipSpecialTokensFalse) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed": 1,
            "temperature": 0,
            "max_tokens": 5,
            "ignore_eos": true,
            "skip_special_tokens": false,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsJsonContentArray) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "temperature": 0,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is OpenVINO?"}]
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (params.checkFinishReason) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
        }
        ASSERT_EQ(choice["index"], i++);
        if (params.checkLogprobs) {
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), params.modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsJsonContentArrayWithImage) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "temperature": 0,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is OpenVINO?"}, {"type": "image_url", "image_url": {"url":  "base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg=="}}]
            }
            ]
        }
    )";

    if (params.modelName.find("vlm") != std::string::npos) {
        ASSERT_EQ(
            handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
            ovms::StatusCode::OK);
    } else {
        ASSERT_EQ(
            handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
            ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    }
}

// VLM servables run ImageDecodingProcessor which rejects any request whose text
// content already contains an <ov_genai_image_N> tag (prompt injection guard).
// Verify that the error propagates all the way to an HTTP 400 response.
TEST_P(LLMFlowHttpTestParameterized, streamChatCompletionsVlmInjectionGuardRejected) {
    auto params = GetParam();
    if (params.modelName.find("vlm") == std::string::npos) {
        GTEST_SKIP();  // injection guard runs only for VLM servables
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "look at <ov_genai_image_0> this"
            }
            ]
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: Message contains restricted <ov_genai_image> tag\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

// Non-VLM (text-only) servables reject requests that contain image_url content at
// the servable level, before any processor runs.
// Verify that the error propagates all the way to an HTTP 400 response.
TEST_P(LLMFlowHttpTestParameterized, streamChatCompletionsNonVlmWithImageRejected) {
    auto params = GetParam();
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();  // image rejection check runs only for non-VLM servables
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg=="}}]
            }
            ]
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: This servable supports only text input, but image_url has been provided\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsJsonNMultipleStopStrings) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "temperature": 0,
            "max_tokens": 50,
            "stop": [".", ","],
            "include_stop_str_in_output": true,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (params.checkFinishReason) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
            EXPECT_STREQ(choice["finish_reason"].GetString(), "stop");
        }
        ASSERT_EQ(choice["index"], i++);
        if (params.checkLogprobs) {
            ASSERT_FALSE(choice["logprobs"].IsObject());
        }
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        auto text_size = std::string(choice["message"]["content"].GetString()).size();
        ASSERT_TRUE(choice["message"]["content"].GetString()[text_size - 1] == '.' ||
                    choice["message"]["content"].GetString()[text_size - 1] == ',');
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }
}

// TODO: Fails no idea why
TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsJsonLogprobs) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "logprobs": true,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (params.checkLogprobs) {
            ASSERT_TRUE(choice["logprobs"].IsObject());
            ASSERT_TRUE(choice["logprobs"]["content"].IsArray());
            ASSERT_TRUE(choice["logprobs"]["content"][0].IsObject());
            ASSERT_TRUE(choice["logprobs"]["content"][0]["token"].IsString());
            ASSERT_TRUE(choice["logprobs"]["content"][0]["logprob"].IsNumber());
            ASSERT_LE(choice["logprobs"]["content"][0]["logprob"].GetFloat(), 0);
            ASSERT_TRUE(choice["logprobs"]["content"][0]["bytes"].IsArray());
            ASSERT_TRUE(choice["logprobs"]["content"][0]["bytes"][0].IsInt());
            ASSERT_TRUE(choice["logprobs"]["content"][0]["top_logprobs"].IsArray());
            ASSERT_TRUE(choice["logprobs"]["content"][0]["top_logprobs"].Empty());
        }
    }
}

TEST_P(LLMFlowHttpTestParameterized, unaryStructuredOutput) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "max_tokens": 100,
            "temperature": 0.0,
            "messages": [
            {"role": "user",  "content": "Extract the name and age of the person from the text and structure the output in JSON format. Margaret is 20 years old."}
            ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {
                        "type": "object",
                        "properties": {
                            "name": {
                            "type": "string"
                            },
                            "age": {
                            "type": "integer"
                            }
                        },
                        "required": ["name", "age"]
                        }
                    }
                }
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["message"]["content"].IsString());
        rapidjson::Document parsedContent;
        parsedContent.Parse(choice["message"]["content"].GetString());
        ASSERT_TRUE(parsedContent.IsObject());
        ASSERT_TRUE(parsedContent.HasMember("name"));
        ASSERT_TRUE(parsedContent["name"].IsString());
        ASSERT_TRUE(parsedContent.HasMember("age"));
        ASSERT_TRUE(parsedContent["age"].IsInt());
    }
}

TEST_P(LLMFlowHttpTestParameterized, unaryStructuredOutputBadSchema) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
            {"role": "user",  "content": "Extract the name and age of the person from the text and structure the output in JSON format. Margaret is 20 years old."}
            ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {
                        "type": "object",
                        "properties": {
                            "name": {
                            "type": "string"
                            },
                            "age": {
                            "type": "my_integer"
                            }
                        },
                        "required": ["name", "age"]
                        }
                    }
                }
        }
    )";

    // Request should be processed correctly with guided generation implicitly disabled due to bad schema
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMFlowHttpTestParameterized, unaryStructuredOutputNonOpenAI) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "max_tokens": 150,
            "temperature": 0.0,
            "messages": [
            {"role": "user",  "content": "Extract the name and age of the person from the text and structure the output in JSON format. Margaret is 20 years old."}
            ],
                "response_format": {
                    "type": "sequence",
                    "elements": [
                        {
                            "type": "json_schema",
                            "json_schema": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                    "type": "string"
                                    },
                                    "age": {
                                    "type": "integer"
                                    }
                                },
                                "required": ["name", "age"]
                            }
                        },
                        {
                            "type": "const_string",
                            "value": "\n\nYou're welcome!"
                        }
                    ]
                }
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["message"]["content"].IsString());
        std::string content = choice["message"]["content"].GetString();
        // Check if content ends with "\n\nYou're welcome!" and extract JSON part
        const std::string suffix = "\n\nYou're welcome!";
        ASSERT_TRUE(content.size() >= suffix.size());
        ASSERT_EQ(content.compare(content.size() - suffix.size(), suffix.size(), suffix), 0);
        std::string jsonPart = content.substr(0, content.size() - suffix.size());
        rapidjson::Document parsedContent;
        parsedContent.Parse(jsonPart.c_str());
        ASSERT_TRUE(parsedContent.IsObject());
        ASSERT_TRUE(parsedContent.HasMember("name"));
        ASSERT_TRUE(parsedContent["name"].IsString());
        ASSERT_TRUE(parsedContent.HasMember("age"));
        ASSERT_TRUE(parsedContent["age"].IsInt());
    }
}

TEST_P(LLMFlowHttpTestParameterized, unaryToolBadSchema) {
    std::string requestBody = R"(
    {
        "model": "lm_cb_with_tool_parser",
        "stream": false,
        "temperature": 0,
        "max_tokens": 5,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_horoscope",
                    "description": "Get today's horoscope for an astrological sign.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sign": {
                                "type": "string_with_a_suffix",
                                "description": "An astrological sign like Taurus or Aquarius"
                            }
                        },
                        "required": [
                            "sign"
                        ]
                    }
                }
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "What is my horoscope? I am an Aquarius."
            }
        ]
    }
    )";
    // Request should be processed correctly with guided generation implicitly disabled due to bad schema
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJsonLogprobs) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "logprobs": 1,
            "prompt":  "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (params.checkLogprobs) {
            ASSERT_TRUE(choice["logprobs"].IsObject());
            ASSERT_TRUE(choice["logprobs"]["text_offset"].IsArray());
            ASSERT_TRUE(choice["logprobs"]["text_offset"][0].IsInt());
            ASSERT_TRUE(choice["logprobs"]["token_logprobs"].IsArray());
            ASSERT_TRUE(choice["logprobs"]["token_logprobs"][0].IsNumber());
            ASSERT_LE(choice["logprobs"]["token_logprobs"][0].GetFloat(), 0);
            ASSERT_TRUE(choice["logprobs"]["tokens"].IsArray());
            ASSERT_TRUE(choice["logprobs"]["tokens"][0].IsString());
            ASSERT_TRUE(choice["logprobs"]["top_logprobs"].IsArray());
        }
    }
}

TEST_P(LLMFlowHttpTestParameterized, ChatCompletionsJsonLogprobsStream) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "logprobs": true,
            "seed" : 1,
            "max_tokens": 1,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, CompletionsJsonLogprobsStream) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "logprobs": 2,
            "seed" : 1,
            "max_tokens": 1,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsStopStringBadType) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "stop": {},
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsIncludeStopStringInOutputBadType) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "stop": "\n\n",
            "include_stop_str_in_output": "yes",
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsStopStringElementBadType) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "stop": [".", "OpenVINO", 1.92],
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsStopStringExceedingSize) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "stop": ["a", "b", "c", "d", "1", "2", "3", "4", "x", "y", "z", "w", "9", "8", "7", "6", "exceeded"],
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsPromptTokensWithMaxTokensExceedsMaxModelLength) {
    auto params = GetParam();
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string prompt;
    // creating prompt that will be tokenized to 8189 tokens when model max length is 8192; 29 are tokens from chat template,
    // and 3 tokens are reserved (e.g., for special/assistant tokens or safety margin).
    for (int i = 0; i < 8192 - 29 - 3; i++) {
        prompt += "hello ";
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "max_tokens" : 10,
            "messages": [
            {
                "role": "user",
                "content": ")" +
                              prompt + R"("
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsPromptTokensWithMaxCompletionTokensExceedsMaxModelLength) {
    auto params = GetParam();
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string prompt;
    // creating prompt that will be tokenized to 8189 tokens when model max length is 8192; 25 are tokens from chat template.
    for (int i = 0; i < 8191 - 25 - 3; i++) {  // 3 extra tokens are reserved for special tokens added by the tokenizer
        prompt += "hello ";
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "max_completion_tokens": 10,
            "messages": [
            {
                "role": "user",
                "content": ")" +
                              prompt + R"("
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsPromptTokensEqualToMaxModelLength) {
    auto params = GetParam();
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string prompt;
    // creating prompt that will be tokenized to  tokens when model max length is 8192; 32 are tokens from chat template.
    for (int i = 0; i < 8192 - 32 + 1; i++) {
        prompt += "hello ";
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "messages": [
            {
                "role": "user",
                "content": ")" +
                              prompt + R"("
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsStoppedByMaxModelLength) {
    auto params = GetParam();
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string prompt;
    // creating prompt that will be tokenized to 2044 tokens when model max length is 2048
    for (int i = 0; i < 2044; i++) {
        prompt += "hello ";
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "messages": [
            {
                "role": "user",
                "content": ")" +
                              prompt + R"("
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    // parsedResponse.Parse(response.c_str());
    // ASSERT_TRUE(parsedResponse["usage"].IsObject());
    // ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    // EXPECT_EQ(parsedResponse["usage"].GetObject()["prompt_tokens"].GetInt(), 2047);
    // ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    // EXPECT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 1); // TODO check why those check are failing sporadically
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsStopStringEmpty) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "stop": [],
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMFlowHttpTestParameterized, streamBeamSearchCompletionsFail) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "best_of": 2,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, streamBeamSearchChatCompletionsFail) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "best_of": 2,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, inferCompletionsStream) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "seed" : 1,
            "max_tokens": 5,
            "ignore_eos": true,
            "prompt": "What is OpenVINO?"
        }
    )";
    bool firstChunk = true;
    ON_CALL(*writer, PartialReply).WillByDefault([this, &params, &firstChunk](std::string response) {
        rapidjson::Document d;
        std::string dataPrefix = "data:";
        ASSERT_STREQ(response.substr(0, dataPrefix.size()).c_str(), dataPrefix.c_str());
        size_t pos = response.find("\n");
        ASSERT_NE(pos, response.npos);
        rapidjson::ParseResult parsingSucceeded = d.Parse(response.substr(dataPrefix.size(), (pos - dataPrefix.size())).c_str());
        ASSERT_EQ(parsingSucceeded.Code(), 0);
        ASSERT_TRUE(d["choices"].IsArray());
        ASSERT_EQ(d["choices"].Size(), 1);
        int i = 0;
        for (auto& choice : d["choices"].GetArray()) {
            if (params.checkFinishReason) {
                if (choice["finish_reason"].IsString()) {
                    EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
                } else {
                    ASSERT_TRUE(choice["finish_reason"].IsNull());
                }
            }
            ASSERT_EQ(choice["index"], i++);
            if (params.checkLogprobs) {
                ASSERT_FALSE(choice["logprobs"].IsObject());
            }
            if (firstChunk && params.checkHandshakeChunk) {
                ASSERT_TRUE(choice["text"].IsNull() || choice["text"].IsString());
            } else {
                ASSERT_TRUE(choice["text"].IsString());
            }
            firstChunk = false;
        }
        EXPECT_STREQ(d["model"].GetString(), params.modelName.c_str());
        EXPECT_STREQ(d["object"].GetString(), "text_completion.chunk");
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, inferChatCompletionsStream) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "seed" : 1,
            "max_tokens": 5,
            "ignore_eos": true,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";
    int replyCounter = 0;
    ON_CALL(*writer, PartialReply).WillByDefault([this, &params, &replyCounter](std::string response) {
        if (replyCounter == 0 && params.checkHandshakeChunk) {
            replyCounter++;
            assertInitialStreamChatCompletionChunk(response, params.modelName);
            return;
        }
        rapidjson::Document d;
        std::string dataPrefix = "data:";
        ASSERT_STREQ(response.substr(0, dataPrefix.size()).c_str(), dataPrefix.c_str());
        size_t pos = response.find("\n");
        ASSERT_NE(pos, response.npos);
        rapidjson::ParseResult parsingSucceeded = d.Parse(response.substr(dataPrefix.size(), (pos - dataPrefix.size())).c_str());
        ASSERT_EQ(parsingSucceeded.Code(), 0);
        ASSERT_TRUE(d["choices"].IsArray());
        ASSERT_EQ(d["choices"].Size(), 1);
        int i = 0;
        for (auto& choice : d["choices"].GetArray()) {
            if (params.checkFinishReason) {
                if (choice["finish_reason"].IsString()) {
                    EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
                } else {
                    ASSERT_TRUE(choice["finish_reason"].IsNull());
                }
            }
            ASSERT_EQ(choice["index"], i++);
            if (params.checkLogprobs) {
                ASSERT_FALSE(choice["logprobs"].IsObject());
            }
            if (choice.HasMember("delta")) {
                ASSERT_TRUE(choice["delta"].IsObject());
                ASSERT_TRUE(choice["delta"]["content"].IsString());
            }
        }
        EXPECT_STREQ(d["model"].GetString(), params.modelName.c_str());
        EXPECT_STREQ(d["object"].GetString(), "chat.completion.chunk");
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, inferChatCompletionsStreamSkipSpecialTokensFalse) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "seed": 1,
            "max_tokens": 5,
            "ignore_eos": true,
            "skip_special_tokens": false,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";
    int replyCounter = 0;
    ON_CALL(*writer, PartialReply).WillByDefault([this, &params, &replyCounter](std::string response) {
        if (replyCounter == 0 && params.checkHandshakeChunk) {
            replyCounter++;
            assertInitialStreamChatCompletionChunk(response, params.modelName);
            return;
        }
        rapidjson::Document d;
        std::string dataPrefix = "data:";
        ASSERT_STREQ(response.substr(0, dataPrefix.size()).c_str(), dataPrefix.c_str());
        size_t pos = response.find("\n");
        ASSERT_NE(pos, response.npos);
        rapidjson::ParseResult parsingSucceeded = d.Parse(response.substr(dataPrefix.size(), (pos - dataPrefix.size())).c_str());
        ASSERT_EQ(parsingSucceeded.Code(), 0);
        ASSERT_TRUE(d["choices"].IsArray());
        ASSERT_EQ(d["choices"].Size(), 1);
        for (auto& choice : d["choices"].GetArray()) {
            if (choice.HasMember("delta")) {
                ASSERT_TRUE(choice["delta"].IsObject());
                ASSERT_TRUE(choice["delta"]["content"].IsString());
            }
        }
        EXPECT_STREQ(d["object"].GetString(), "chat.completion.chunk");
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsStreamOptionsSetFail) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "stream_options": { "include_usage": true },
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsStreamOptionsSetFail) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "stream_options": { "include_usage": true },
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, streamChatCompletionsFinishReasonLength) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
    }
}

TEST_P(LLMFlowHttpTestParameterized, streamChatCompletionsSingleStopString) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "temperature" : 0,
            "ignore_eos": false,
            "max_tokens": 1000,
            "stop": ".",
            "include_stop_str_in_output": true,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO? Give one sentence answer."
            }
            ]
        }
    )";

    std::vector<std::string> responses;

    // Gather responses
    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });

    // dispatchToProcessor is blocking because of mocked PartialReplyBegin in fixture:
    // ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    SPDLOG_TRACE("Will dispatch");
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    SPDLOG_TRACE("After dispatch");

    if (params.checkHandshakeChunk) {
        // Check if there is more than 1 partial response - initial and at least one real response with stop string
        ASSERT_GT(responses.size(), 1);

        // Assert initial message with empty content
        assertInitialStreamChatCompletionChunk(responses[0], params.modelName);
    } else {
        // For legacy there is no initial empty message
        ASSERT_GT(responses.size(), 0);
    }

    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"stop\"") != std::string::npos);
    }

    // In legacy streaming we don't know if the callback is the last one, so we rely on entire generation call finish.
    // Because of that, we might get additional response with empty content at the end of the stream.
    const size_t numberOfLastResponsesToCheckForStopString = std::min(
        params.modelName.find("legacy") != std::string::npos ? size_t{2} : size_t{1},
        responses.size());

    // The stop string (.) does not need to be at the end of the message.
    // There are cases when the last generation contains dot and a new lines, or generated token is "e.g",
    // or simply any token (or group of tokens) that has dot in a middle.

    const std::string eventSep = "\n\n";
    const std::string dataPrefix = "data:";

    // Check for no existence of a dot:
    for (size_t i = params.checkHandshakeChunk ? 1 : 0; i < responses.size() - numberOfLastResponsesToCheckForStopString; ++i) {
        size_t start = 0;
        while (start < responses[i].size()) {
            const size_t eventEnd = responses[i].find(eventSep, start);
            if (eventEnd == std::string::npos)
                break;
            const std::string event = responses[i].substr(start, eventEnd - start);
            start = eventEnd + eventSep.size();
            if (event.size() < dataPrefix.size())
                continue;
            const std::string body = event.substr(dataPrefix.size());
            if (body.find("[DONE]") != std::string::npos)
                break;
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(body.c_str());
            ASSERT_EQ(ok.Code(), 0) << d.GetParseError() << "\n"
                                    << body;
            ASSERT_TRUE(d["choices"].IsArray());
            ASSERT_EQ(d["choices"].Size(), 1);
            ASSERT_TRUE(d["choices"][0].IsObject());
            if (!d["choices"][0].HasMember("delta"))
                continue;
            ASSERT_TRUE(d["choices"][0]["delta"].IsObject());
            if (!d["choices"][0]["delta"].HasMember("content") || !d["choices"][0]["delta"]["content"].IsString())
                continue;
            std::string content = d["choices"][0]["delta"]["content"].GetString();
            ASSERT_EQ(content.find('.'), std::string::npos) << "found dot in response: " << responses[i] << " at index: " << i << " out of: " << responses.size();
        }
    }

    bool foundDotInLastResponse = false;
    // Check for existence of a dot:
    for (size_t i = responses.size() - numberOfLastResponsesToCheckForStopString; i < responses.size(); ++i) {
        size_t start = 0;
        while (start < responses[i].size()) {
            const size_t eventEnd = responses[i].find(eventSep, start);
            if (eventEnd == std::string::npos)
                break;
            const std::string event = responses[i].substr(start, eventEnd - start);
            start = eventEnd + eventSep.size();
            if (event.size() < dataPrefix.size())
                continue;
            const std::string body = event.substr(dataPrefix.size());
            if (body.find("[DONE]") != std::string::npos)
                break;
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(body.c_str());
            ASSERT_EQ(ok.Code(), 0) << d.GetParseError() << "\n"
                                    << body;
            ASSERT_TRUE(d["choices"].IsArray());
            ASSERT_EQ(d["choices"].Size(), 1);
            ASSERT_TRUE(d["choices"][0].IsObject());
            if (!d["choices"][0].HasMember("delta"))
                continue;
            ASSERT_TRUE(d["choices"][0]["delta"].IsObject());
            if (!d["choices"][0]["delta"].HasMember("content") || !d["choices"][0]["delta"]["content"].IsString())
                continue;
            std::string content = d["choices"][0]["delta"]["content"].GetString();
            if (content.find('.') != std::string::npos) {
                foundDotInLastResponse = true;
            }
        }
    }
    ASSERT_TRUE(foundDotInLastResponse) << "cannot find dot last responses";
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsFinishReasonLength) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
    }
}

// Potential sporadic - move to functional if problematic
TEST_P(LLMFlowHttpTestParameterized, streamCompletionsSingleStopString) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "seed" : 1,
            "ignore_eos": false,
            "max_tokens": 1000,
            "stop": ".",
            "temperature":0,
            "include_stop_str_in_output": true,
            "prompt": "What is OpenVINO?"
        }
    )";

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    SPDLOG_DEBUG("Test middle");
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"stop\"") != std::string::npos);
    }
    std::regex content_regex("\"text\":\".*\\.[ ]{0,1}\"");
    if (params.modelName.find("legacy") != std::string::npos) {
        // In legacy streaming we don't know if the callback is the last one, so we rely on entire generation call finish.
        // Because of that, we might get additional response with empty content at the end of the stream.
        // Guard against responses.size() < 2 (can happen when all deltas arrive in a single drain).
        if (responses.size() >= 2) {
            ASSERT_TRUE(std::regex_search(responses[responses.size() - 2], content_regex) || std::regex_search(responses.back(), content_regex));
        } else {
            ASSERT_GE(responses.size(), 1u);
            ASSERT_TRUE(std::regex_search(responses.back(), content_regex));
        }
    } else {
        ASSERT_TRUE(std::regex_search(responses.back(), content_regex));
    }
    SPDLOG_DEBUG("Test end");
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsSpaceStopString) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "seed" : 1,
            "ignore_eos": false,
            "max_tokens": 1000,
            "stop": " ",
            "temperature":0,
            "include_stop_str_in_output": true,
            "prompt": "                 |                  |                   |  "
        }
    )";

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_GE(responses.size(), 1);
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"stop\"") != std::string::npos);
    }
    ASSERT_TRUE(responses.back().find("\"text\":\"\"") != std::string::npos);
}

TEST_P(LLMFlowHttpTestParameterized, streamChatCompletionsUsage) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stream_options": { "include_usage": true },
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_GT(responses.size(), 0);
    ASSERT_TRUE(responses.back().find("\"completion_tokens\":5") != std::string::npos) << responses.back();  // ensure 5 - reaching max_tokens
    ASSERT_TRUE(responses.back().find("\"prompt_tokens\"") != std::string::npos) << responses.back();        // this is always present and > 0, depends on pipeline type and underlying model
    ASSERT_TRUE(responses.back().find("\"total_tokens\"") != std::string::npos) << responses.back();         // this is always present and > 0, depends on pipeline type and underlying model
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos) << responses.back();
    }
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsUsage) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stream_options": { "include_usage": true },
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_TRUE(responses.back().find("\"completion_tokens\":5") != std::string::npos) << responses.back();  // ensure 5 - reaching max_tokens
    ASSERT_TRUE(responses.back().find("\"prompt_tokens\"") != std::string::npos) << responses.back();        // this is always present and > 0, depends on pipeline type and underlying model
    ASSERT_TRUE(responses.back().find("\"total_tokens\"") != std::string::npos) << responses.back();         // this is always present and > 0, depends on pipeline type and underlying model
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos) << responses.back();
    }
}

TEST_P(LLMFlowHttpTestParameterized, streamChatCompletionsBadStopStringType) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stop": {},
            "include_stop_str_in_output": true,
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: stop is not a string or array of strings\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsBadStopStringElementType) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stop": ["abc", "def", []],
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: stop array contains non string element\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsIncludeStopStrInOutputFalse) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stop": ".",
            "include_stop_str_in_output": false,
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: include_stop_str_in_output cannot be set to false if streaming is used\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsBadIncludeStopStrInOutputType) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stop": ["abc", "def"],
            "include_stop_str_in_output": 1.9,
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: include_stop_str_in_output accepts values true or false\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, streamChatCompletionsBadStreamOptionsBadType) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stream_options": ["include_usage"],
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: stream_options is not an object\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsStreamOptionsBadType) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stream_options": ["include_usage"],
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: stream_options is not an object\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, streamChatCompletionsStreamOptionsBadContent) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stream_options": { "option": "A" },
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: Found unexpected stream options. Properties accepted in stream_options: include_usage\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsStreamOptionsBadContent) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stream_options": { "include_usage": true, "option": "A" },
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: Found unexpected stream options. Properties accepted in stream_options: include_usage\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, streamChatCompletionsBadIncludeUsage) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stream_options": { "include_usage": 123 },
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: stream_options.include_usage is not a boolean\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

TEST_P(LLMFlowHttpTestParameterized, streamCompletionsBadIncludeUsage) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "stream_options": { "include_usage": 123 },
            "ignore_eos": true,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([this](std::string response, ovms::HTTPStatusCode code) {
            ASSERT_EQ(response, "{\"error\":\"Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: stream_options.include_usage is not a boolean\"}");
            rapidjson::Document d;
            rapidjson::ParseResult ok = d.Parse(response.c_str());
            ASSERT_EQ(ok.Code(), 0);
            ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
}

// /v3/chat/completions endpoint
// unary, gready search
// Correct payload, however disconnection immediately
TEST_P(LLMFlowHttpTestParameterized, inferChatCompletionsUnaryClientDisconnectedImmediately) {
    auto params = GetParam();
    if (params.modelName.find("legacy") != std::string::npos) {
        // TODO: Disconnection logic should probably be adjusted for legacy servable streaming
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
                {
                    "role": "user",
                    "content": "What is OpenVINO?"
                }
            ]
        }
    )";

    ON_CALL(*writer, IsDisconnected()).WillByDefault(::testing::Return(true));
    ON_CALL(*writer, RegisterDisconnectionCallback(::testing::_)).WillByDefault([](std::function<void()> fn) {
        fn();  // disconnect immediately, even before read_all is called
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

// /v3/chat/completions endpoint
// streaming
// Correct payload, however disconnection immediately
TEST_P(LLMFlowHttpTestParameterized, inferChatCompletionsStreamClientDisconnectedImmediately) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "seed" : 1,
            "max_tokens": 5,
            "messages": [
                {
                    "role": "user",
                    "content": "What is OpenVINO?"
                }
            ]
        }
    )";

    EXPECT_CALL(*writer, IsDisconnected())
        .WillOnce(::testing::Return(true));

    std::atomic<int> i = 0;
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_)).WillOnce([this, &i](std::string partialResponse, ovms::HTTPStatusCode code) {
        i++;
        ASSERT_EQ(partialResponse, "{\"error\":\"Mediapipe execution failed. MP status - CANCELLED: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: \"}");
        rapidjson::Document d;
        rapidjson::ParseResult ok = d.Parse(partialResponse.c_str());
        ASSERT_EQ(ok.Code(), 0);
        ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
    });  // no results

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_EQ(i, 1);
    ASSERT_EQ(response, "");
}

// /v3/completions endpoint
// streaming
// Correct payload, however disconnection immediately
TEST_P(LLMFlowHttpTestParameterized, inferCompletionsStreamClientDisconnectedImmediately) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": true,
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    EXPECT_CALL(*writer, IsDisconnected())
        .WillOnce(::testing::Return(true));

    std::atomic<int> i = 0;
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_)).WillOnce([this, &i](std::string partialResponse, ovms::HTTPStatusCode code) {
        i++;
        ASSERT_EQ(partialResponse, "{\"error\":\"Mediapipe execution failed. MP status - CANCELLED: CalculatorGraph::Run() failed: \\nCalculator::Process() for node \\\"llmNode1\\\" failed: \"}");
        rapidjson::Document d;
        rapidjson::ParseResult ok = d.Parse(partialResponse.c_str());
        ASSERT_EQ(ok.Code(), 0);
        ASSERT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
    });  // no results

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_EQ(i, 1);
    ASSERT_EQ(response, "");
}

INSTANTIATE_TEST_SUITE_P(
    LLMFlowHttpTestInstances,
    LLMFlowHttpTestParameterized,
    ::testing::Values(
        // params:     model name, generate expected output, check logprobs, check finish reason, test speculative decoding, supports empty handshake msg
        TestParameters{"lm_cb_regular", true, true, true, false, true},
        TestParameters{"lm_legacy_regular", false, false, true, false, false},
        TestParameters{"vlm_cb_regular", false, true, true, false, true},
        TestParameters{"vlm_legacy_regular", false, false, true, false, false}));

const std::string validRequestBodyWithParameter(const std::string& modelName, const std::string& parameter, const std::string& value) {
    std::string requestBody = R"(
        {
            "model": ")" + modelName +
                              R"(",
            "max_tokens": 1,
            ")" + parameter + R"(": )" +
                              value + R"(,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    return requestBody;
}

class LLMHttpParametersValidationTest : public LLMFlowHttpTest, public ::testing::WithParamInterface<TestParameters> {};

TEST_P(LLMHttpParametersValidationTest, maxTokensInvalid) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": "INVALID",
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, maxTokensExceedsUint32Size) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 4294967296,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, maxCompletionsTokensInvalid) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_completion_tokens": "INVALID",
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, maxCompletionsTokensExceedsUint32Size) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_completion_tokens": 4294967296,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, streamInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "stream", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::JSON_INVALID);
}

TEST_P(LLMHttpParametersValidationTest, messagesInvalid) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 1,
            "messages": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, messagesMissing) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 1
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, messageNotAnObject) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 1,
            "messages": [
                "What is OpenVINO?"
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, contentNotValid) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": "user",
                "content": [1,2,3]
            }
            ]
        }
    )";

    ovms::Status status = handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    ASSERT_NE(status.string().find("Invalid message structure - content array should contain objects"), std::string::npos);
}

TEST_P(LLMHttpParametersValidationTest, additionalArrayTypeElementInMessage) {
    // Note that tool calls are not visible in non-Python build
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": "assistant",
                "content": "Some content",
                "tool_calls": [{"type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"}}]
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, missingContentInMessage) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"}}]
            }
            ]
        }
    )";

    ovms::Status status = handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status.getCode(), ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, roleNotAString) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": false,
                "content": "What is OpenVino?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, promptInvalid) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 1,
            "prompt": 5
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, promptMissing) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "max_tokens": 1
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, modelMissing) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::JSON_INVALID);
}

TEST_P(LLMHttpParametersValidationTest, modelInvalid) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": 0,
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::JSON_INVALID);
}

TEST_P(LLMHttpParametersValidationTest, ignoreEosValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "ignore_eos", "false");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, ignoreEosInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "ignore_eos", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, repetitionPenaltyValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "repetition_penalty", "2.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "repetition_penalty", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, repetitionPenaltyInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "repetition_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, lengthPenaltyValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "length_penalty", "2.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "length_penalty", "2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, lengthPenaltyInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "length_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, temperatureValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "temperature", "1.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "temperature", "0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "temperature", "2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, temperatureInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "temperature", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, temperatureOutOfRange) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "temperature", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, frequencyPenaltyValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "frequency_penalty", "1.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "frequency_penalty", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, frequencyPenaltyInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "frequency_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, frequencyPenaltyOutOfRange) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "frequency_penalty", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, presencePenaltyValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "presence_penalty", "1.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "presence_penalty", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, presencePenaltyInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "presence_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, presencePenaltyOutOfRange) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "presence_penalty", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, topPValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_p", "0.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "top_p", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, topPInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_p", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, topPOutOfRange) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_p", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, topKValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_k", "2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, topKInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_k", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, topKMinuOneValid) {
    auto params = GetParam();
    // -1 is the sentinel for "consider all tokens"
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_k", "-1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, topKZeroInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_k", "0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, topKNegativeInvalid) {
    auto params = GetParam();
    // Only -1 is a valid negative value; other negatives must be rejected
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_k", "-2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, minPValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "min_p", "0.05");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "min_p", "0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, minPInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "min_p", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, minPOutOfRange) {
    auto params = GetParam();
    // min_p must be in [0.0, 1.0) — value of 1.0 is out of range
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "min_p", "1.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, minPNegative) {
    auto params = GetParam();
    // min_p must be in [0.0, 1.0) — negative value is out of range
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "min_p", "-0.1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, seedValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "seed", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, seedInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "seed", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, seedBoundaryZero) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "seed", "0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, seedBoundaryMax) {
    auto params = GetParam();
    // Maximum valid seed: 2^32 - 1 = 4294967295
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "seed", "4294967295");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, seedOutOfRangeNegative) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "seed", "-1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, seedOutOfRangeOverflow) {
    auto params = GetParam();
    // 2^32 = 4294967296 is one past the maximum valid seed
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "seed", "4294967296");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, bestOfValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "best_of", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, bestOfInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "best_of", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, bestOfNegative) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "best_of", "-1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, bestOfExceedsLimit) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "best_of", "40");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, nValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "n", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, nInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "n", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, nNegative) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "n", "-1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, nGreaterThanBestOf) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "best_of" : 1,
            "n" : 2,
            "max_tokens": 1,
            "messages": [
            {
                "role": "user",
                "content": "What is OpenVINO?"
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, MessagesEmpty) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "max_tokens": 1,
            "messages": []
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, MessagesWithEmptyObject) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "messages": [{}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, EmptyPrompt) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "prompt": ""
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, MessagesWithOnlyRole) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "messages": [{"role": "user"}],
            "max_tokens": 10
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);  // GenAI supports such messages
}

TEST_P(LLMHttpParametersValidationTest, SpeculativeDecodingNoSDSpecificParametersProvided) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    if (!params.testSpeculativeDecoding) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "prompt": "hello"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, SpeculativeDecodingNoSDSpecificParametersProvidedChat) {
    auto params = GetParam();
    if (!params.testSpeculativeDecoding) {
        GTEST_SKIP();
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "messages": [{"content": "def"}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, MessagesWithOnlyContent) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "max_tokens": 1,
            "messages": [{"content": "def"}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, MessagesWithMoreMessageFields) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "def", "unexpected": "123"}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
}

INSTANTIATE_TEST_SUITE_P(
    LLMHttpParametersValidationTestInstances,
    LLMHttpParametersValidationTest,
    ::testing::Values(
        // params:     model name, generate expected output, check logprobs, check finish reason, test speculative decoding, supports empty handshake msg
        TestParameters{"lm_cb_regular", true, true, true, false, true},
        TestParameters{"lm_legacy_regular", false, false, false, false, false},
        TestParameters{"vlm_cb_regular", false, true, true, false, true},
        TestParameters{"vlm_legacy_regular", false, false, true, false, false}));

// Common tests for all pipeline types (testing logic executed prior pipeline type selection)
class LLMConfigHttpTest : public ::testing::Test {};

TEST_F(LLMConfigHttpTest, LLMNodeNameMissing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "./"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_MISSING_NAME);
}

TEST_F(LLMConfigHttpTest, LLMNodeOptionsMissing) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "LLMExecutor"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_MISSING_OPTIONS);
}

TEST_F(LLMConfigHttpTest, LLMNodeNameExists) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD_1:input"
        input_stream: "HTTP_REQUEST_PAYLOAD_2:input2"
        output_stream: "HTTP_RESPONSE_PAYLOAD_1:output"
        output_stream: "HTTP_RESPONSE_PAYLOAD_2:output2"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback2"
        input_stream: "HTTP_REQUEST_PAYLOAD:input2"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback2"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output2"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/ovms/src/test/llm_testing/facebook/opt-125m"
                cache_size: 1
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    auto& m = mediapipeDummy.getGenAiServableMap();
    m.insert(std::pair<std::string, std::shared_ptr<GenAiServable>>("llmNode", nullptr));
    ASSERT_EQ(mediapipeDummy.validateForConfigFileExistence(), StatusCode::OK);
    ASSERT_EQ(mediapipeDummy.validateForConfigLoadablenessPublic(), StatusCode::OK);
    ASSERT_EQ(mediapipeDummy.initializeNodes(), StatusCode::LLM_NODE_NAME_ALREADY_EXISTS);
}

TEST_F(LLMConfigHttpTest, LLMNodeNonExistantModelsPath) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/models_path"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    auto status = mediapipeDummy.validate(manager);
    ASSERT_EQ(status, StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST) << status.string();
}

TEST_F(LLMConfigHttpTest, LLMNodeBadWorkspacePathEmpty) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ""
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    auto status = mediapipeDummy.validate(manager);
    ASSERT_EQ(status, StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST) << status.string();
}

TEST_F(LLMConfigHttpTest, LLMNodeWorkspacePathToFileNotDir) {
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/ovms/src/test/llm_testing/facebook/opt-125m/config.json"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    auto status = mediapipeDummy.validate(manager);
    ASSERT_EQ(status, StatusCode::LLM_NODE_PATH_DOES_NOT_EXIST_AND_NOT_GGUFFILE) << status.string();
}

class LLMConfigHttpTestParameterized : public ::testing::Test, public ::testing::WithParamInterface<std::tuple<std::string, ovms::StatusCode>> {};

TEST_P(LLMConfigHttpTestParameterized, LLMNodeResourceInitFailed) {
    auto [pipelineType, expectedStatusCode] = GetParam();
    ConstructorEnabledModelManager manager;
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/",
                pipeline_type: )" +
                            pipelineType + R"(
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";

    ovms::MediapipeGraphConfig mgc{"mediaDummy", "", ""};
    DummyMediapipeGraphDefinition mediapipeDummy("mediaDummy", mgc, testPbtxt, nullptr);
    mediapipeDummy.inputConfig = testPbtxt;
    auto status = mediapipeDummy.validate(manager);
    ASSERT_EQ(status, expectedStatusCode);
    ASSERT_EQ(mediapipeDummy.getGenAiServable("llmNode"), nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    LLMConfigHttpTestInstances,
    LLMConfigHttpTestParameterized,
    // For VLM, directory contents are checked in pipeline selection logic,
    // before pipeline initialization, hence INTERNAL_ERROR not LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED
    // We might want to consider unification of error codes in the future
    ::testing::Values(
        std::make_tuple("LM_CB", ovms::StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED),
        std::make_tuple("LM", ovms::StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED),  // TODO unstable
        std::make_tuple("VLM_CB", ovms::StatusCode::INTERNAL_ERROR),
        std::make_tuple("VLM", ovms::StatusCode::INTERNAL_ERROR)));

// Those tests are working on Continuous Batching path, since most of the node options are scheduler parameters that are not used in non-CB servables
// We could consider adding tests for non-CB path in the future in the separate test suite
class LLMOptionsHttpTestPython : public ::testing::Test {};

class LLMOptionsHttpTest : public LLMOptionsHttpTestPython {
public:
    std::string modelsPath;
    void SetUp() { modelsPath = "/ovms/src/test/llm_testing/facebook/opt-125m"; }
};

class LLMVLMOptionsHttpTest : public LLMOptionsHttpTestPython {
public:
    std::string modelsPath;
    void SetUp() { modelsPath = "/ovms/src/test/llm_testing/OpenVINO/InternVL2-1B-int4-ov"; }
};

void TestLLMNodeOptionsCheckDefault(std::string& modelsPath) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ")" +
                            modelsPath + R"("
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
    auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());
    ASSERT_EQ(properties->schedulerConfig.max_num_batched_tokens, 256);
    ASSERT_EQ(properties->schedulerConfig.cache_size, 0);
    ASSERT_EQ(properties->schedulerConfig.dynamic_split_fuse, true);
    ASSERT_EQ(properties->schedulerConfig.max_num_seqs, 256);
    ASSERT_EQ(properties->schedulerConfig.enable_prefix_caching, false);
    ASSERT_EQ(properties->device, "CPU");
    // CPU default properties (inference_num_threads, enable_cpu_pinning) are automatically
    // added to pluginConfig for CPU device; verify no user-specified entries are present.
    ASSERT_EQ(properties->pluginConfig.count("PERFORMANCE_HINT"), 0);
    ASSERT_EQ(properties->pluginConfig.count("NUM_STREAMS"), 0);
    ASSERT_EQ(properties->pluginConfig.count("KV_CACHE_PRECISION"), 0);
}
TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckDefault) {
    TestLLMNodeOptionsCheckDefault(modelsPath);
}
TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsCheckDefault) {
    TestLLMNodeOptionsCheckDefault(modelsPath);
}

void LLMNodeOptionsCheckHalfDefault(std::string& modelsPath) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ")" +
                            modelsPath + R"("
                max_num_batched_tokens: 98
                cache_size: 1
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
    auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());

    ASSERT_EQ(properties->schedulerConfig.max_num_batched_tokens, 98);
    ASSERT_EQ(properties->schedulerConfig.cache_size, 1);
    ASSERT_EQ(properties->schedulerConfig.dynamic_split_fuse, true);
    ASSERT_EQ(properties->schedulerConfig.max_num_seqs, 256);
}
TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckHalfDefault) {
    LLMNodeOptionsCheckHalfDefault(modelsPath);
}
TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsCheckHalfDefault) {
    LLMNodeOptionsCheckHalfDefault(modelsPath);
}

void LLMNodeOptionsWrongPluginFormat(std::string& modelsPath) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ")" +
                            modelsPath + R"("
                cache_size: 1
                plugin_config: "[PERF_COUNT=TRUE]"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::PLUGIN_CONFIG_WRONG_FORMAT);
}
TEST_F(LLMOptionsHttpTest, LLMNodeOptionsWrongPluginFormat) {
    LLMNodeOptionsWrongPluginFormat(modelsPath);
}
TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsWrongPluginFormat) {
    LLMNodeOptionsWrongPluginFormat(modelsPath);
}

void LLMNodeOptionsCheckPluginConfig(std::string& modelsPath) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ")" +
                            modelsPath + R"("
                plugin_config: '{"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}'
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
    auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());

    // CPU default properties are added automatically; check only the user-specified entries.
    ASSERT_EQ(properties->pluginConfig.count("PERFORMANCE_HINT"), 1);
    ASSERT_EQ(properties->pluginConfig.count("NUM_STREAMS"), 1);
    ASSERT_EQ(properties->pluginConfig["PERFORMANCE_HINT"], "LATENCY");
    ASSERT_EQ(properties->pluginConfig["NUM_STREAMS"], "1");
}
TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckPluginConfig) {
    LLMNodeOptionsCheckPluginConfig(modelsPath);
}
TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsCheckPluginConfig) {
    LLMNodeOptionsCheckPluginConfig(modelsPath);
}

// RAII guard that restores the global Config singleton (and optionally removes a temporary
// cache directory) on scope exit. The cache_dir tests below mutate the process-wide Config
// singleton; without this guard a failed ASSERT_* mid-test (which returns early) would leak
// the modified --cache_dir into subsequent tests in this suite.
struct GlobalCacheDirGuard {
    ovms::ServerSettingsImpl savedServerSettings;
    ovms::ModelsSettingsImpl savedModelsSettings;
    std::string cacheDirToRemove;

    explicit GlobalCacheDirGuard(std::string cacheDirToRemove = "") :
        savedServerSettings(ovms::Config::instance().getServerSettings()),
        savedModelsSettings(ovms::Config::instance().getModelSettings()),
        cacheDirToRemove(std::move(cacheDirToRemove)) {}

    ~GlobalCacheDirGuard() {
        ovms::Config::instance().parse(&savedServerSettings, &savedModelsSettings);
        if (!cacheDirToRemove.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(cacheDirToRemove, ec);
        }
    }
};

// Verifies that the global --cache_dir (ServerSettings) is propagated into the
// continuous batching pipeline plugin config, and that an explicit CACHE_DIR in
// the node's plugin_config takes precedence over the global value.
// Regression test for openvinotoolkit/model_server#4230.
void LLMNodeOptionsCacheDirPropagation(std::string& modelsPath) {
    // Restore the global cache_dir on scope exit even if an ASSERT below fails early.
    GlobalCacheDirGuard cacheDirGuard;
    // Seed the global cache_dir via the CLI parser (same path used in production).
    const std::string globalCacheDir = (std::filesystem::temp_directory_path() / "ovms_global_cache").string();
    char* n_argv[] = {(char*)"ovms", (char*)"--model_path", (char*)"/path/to/model", (char*)"--model_name", (char*)"some_name", (char*)"--rest_port", (char*)"8080", (char*)"--cache_dir", (char*)globalCacheDir.c_str()};
    int arg_count = 9;
    ovms::Config::instance().parse(arg_count, n_argv);
    ASSERT_EQ(ovms::Config::instance().cacheDir(), globalCacheDir);

    // Case 1: no CACHE_DIR in node plugin_config -> global value is applied.
    {
        std::string testPbtxt = R"(
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"

            node: {
            name: "llmNode"
            calculator: "HttpLLMCalculator"
            input_stream: "LOOPBACK:loopback"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "LLM_NODE_RESOURCES:llm"
            output_stream: "LOOPBACK:loopback"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            input_stream_info: {
                tag_index: 'LOOPBACK:0',
                back_edge: true
            }
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    models_path: ")" +
                                modelsPath + R"("
                }
            }
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                    sync_set {
                    tag_index: "LOOPBACK:0"
                    }
                }
                }
            }
            }
        )";
        adjustConfigForTargetPlatform(testPbtxt);
        ::mediapipe::CalculatorGraphConfig config;
        ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
        std::shared_ptr<GenAiServable> servable;
        ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
        auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());
        ASSERT_EQ(properties->pluginConfig.count("CACHE_DIR"), 1);
        ASSERT_EQ(properties->pluginConfig["CACHE_DIR"].as<std::string>(), globalCacheDir);
    }

    // Case 2: explicit CACHE_DIR in node plugin_config wins over the global value.
    {
        std::string testPbtxt = R"(
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"

            node: {
            name: "llmNode"
            calculator: "HttpLLMCalculator"
            input_stream: "LOOPBACK:loopback"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "LLM_NODE_RESOURCES:llm"
            output_stream: "LOOPBACK:loopback"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            input_stream_info: {
                tag_index: 'LOOPBACK:0',
                back_edge: true
            }
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    models_path: ")" +
                                modelsPath + R"("
                    plugin_config: '{"CACHE_DIR": "/tmp/ovms_node_cache"}'
                }
            }
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                    sync_set {
                    tag_index: "LOOPBACK:0"
                    }
                }
                }
            }
            }
        )";
        adjustConfigForTargetPlatform(testPbtxt);
        ::mediapipe::CalculatorGraphConfig config;
        ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
        std::shared_ptr<GenAiServable> servable;
        ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
        auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());
        ASSERT_EQ(properties->pluginConfig.count("CACHE_DIR"), 1);
        // The test harness may rewrite the path for the target platform, so match
        // on substrings: the explicit node value must win over the global one.
        std::string nodeCacheDir = properties->pluginConfig["CACHE_DIR"].as<std::string>();
        ASSERT_NE(nodeCacheDir.find("ovms_node_cache"), std::string::npos) << "Explicit node CACHE_DIR should be used, got: " << nodeCacheDir;
        ASSERT_EQ(nodeCacheDir.find("ovms_global_cache"), std::string::npos) << "Global cache_dir must not override explicit node CACHE_DIR, got: " << nodeCacheDir;
    }
    // GlobalCacheDirGuard restores the global cache_dir on scope exit.
}
TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCacheDirPropagation) {
    LLMNodeOptionsCacheDirPropagation(modelsPath);
}
TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsCacheDirPropagation) {
    LLMNodeOptionsCacheDirPropagation(modelsPath);
}

// End-to-end regression test for #4230: LLMNodeOptionsCacheDirPropagation above only
// verifies that --cache_dir lands in properties->pluginConfig; it does not prove that
// OpenVINO Core actually persists compiled-model cache artifacts, which was the crux of
// the original bug report (log said "cache enabled", nothing was ever written on disk).
// This test constructs a real ContinuousBatchingPipeline against --cache_dir and asserts
// that compiled-model cache artifacts actually land under it.
TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCacheDirWritesCacheArtifacts) {
    std::string cacheDir = std::filesystem::temp_directory_path().string() +
                           "/LLMNodeOptionsCacheDirWritesCacheArtifacts_" +
                           ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::remove_all(cacheDir);
    std::filesystem::create_directories(cacheDir);
    // Restore the global cache_dir and remove the temp cache dir on scope exit, even if an
    // ASSERT below fails early.
    GlobalCacheDirGuard cacheDirGuard(cacheDir);

    // Seed the global cache_dir via the CLI parser (same path used in production).
    char* n_argv[] = {(char*)"ovms", (char*)"--model_path", (char*)"/path/to/model", (char*)"--model_name", (char*)"some_name", (char*)"--rest_port", (char*)"8080", (char*)"--cache_dir", (char*)cacheDir.c_str()};
    int arg_count = 9;
    ovms::Config::instance().parse(arg_count, n_argv);
    ASSERT_EQ(ovms::Config::instance().cacheDir(), cacheDir);

    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ")" +
                            modelsPath + R"("
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);

    bool foundCacheArtifact = false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(cacheDir)) {
        if (entry.is_regular_file()) {
            foundCacheArtifact = true;
            break;
        }
    }
    EXPECT_TRUE(foundCacheArtifact)
        << "Expected compiled-model cache artifacts under --cache_dir after constructing the "
        << "continuous batching pipeline, found none in: " << cacheDir;
    // GlobalCacheDirGuard restores the global cache_dir and removes cacheDir on scope exit.
}

// Verifies that when multiple LLM nodes are defined in a single graph with mixed cache_dir
// configuration (one with explicit CACHE_DIR in plugin_config, one without), each node
// receives the correct cache_dir: explicit node uses its own value, the other uses the
// global --cache_dir from CLI. Regression test for openvinotoolkit/model_server#4230.
void LLMNodeOptionsMultipleNodesCacheDirPrecedence(std::string& modelsPath) {
    // Restore the global cache_dir on scope exit even if an ASSERT below fails early.
    GlobalCacheDirGuard cacheDirGuard;
    // Seed the global cache_dir via the CLI parser.
    const std::string globalCacheDir = (std::filesystem::temp_directory_path() / "ovms_global_cache_multi").string();
    const std::string nodeCacheDir = (std::filesystem::temp_directory_path() / "ovms_node_cache_multi").string();
    char* n_argv[] = {(char*)"ovms", (char*)"--model_path", (char*)"/path/to/model", (char*)"--model_name", (char*)"some_name", (char*)"--rest_port", (char*)"8080", (char*)"--cache_dir", (char*)globalCacheDir.c_str()};
    int arg_count = 9;
    ovms::Config::instance().parse(arg_count, n_argv);
    ASSERT_EQ(ovms::Config::instance().cacheDir(), globalCacheDir);

    // Create a graph with two LLM nodes:
    // - node1 has explicit CACHE_DIR in plugin_config
    // - node2 has no CACHE_DIR in plugin_config (should use global --cache_dir)
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        # First node: explicit CACHE_DIR in plugin_config
        node: {
            name: "llmNode1"
            calculator: "HttpLLMCalculator"
            input_stream: "LOOPBACK:loopback"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "LLM_NODE_RESOURCES:llm"
            output_stream: "LOOPBACK:loopback"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            input_stream_info: {
                tag_index: 'LOOPBACK:0',
                back_edge: true
            }
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    models_path: ")" +
                            modelsPath + R"("
                    plugin_config: '{"CACHE_DIR": ")" +
                            nodeCacheDir + R"("}'
                }
            }
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                        sync_set {
                            tag_index: "LOOPBACK:0"
                        }
                    }
                }
            }
        }

        # Second node: no CACHE_DIR in plugin_config (should use global --cache_dir)
        node: {
            name: "llmNode2"
            calculator: "HttpLLMCalculator"
            input_stream: "LOOPBACK:loopback"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "LLM_NODE_RESOURCES:llm"
            output_stream: "LOOPBACK:loopback"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            input_stream_info: {
                tag_index: 'LOOPBACK:0',
                back_edge: true
            }
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    models_path: ")" +
                            modelsPath + R"("
                }
            }
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                        sync_set {
                            tag_index: "LOOPBACK:0"
                        }
                    }
                }
            }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));

    // Verify node 1 (with explicit CACHE_DIR) uses the node-level value
    {
        std::shared_ptr<GenAiServable> servable1;
        ASSERT_EQ(initializeGenAiServable(servable1, config.node(0), ""), StatusCode::OK);
        auto properties1 = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable1->getProperties());
        ASSERT_EQ(properties1->pluginConfig.count("CACHE_DIR"), 1);
        std::string node1CacheDir = properties1->pluginConfig["CACHE_DIR"].as<std::string>();
        ASSERT_NE(node1CacheDir.find("ovms_node_cache_multi"), std::string::npos)
            << "Node 1 should have explicit CACHE_DIR, got: " << node1CacheDir;
        ASSERT_EQ(node1CacheDir.find("ovms_global_cache_multi"), std::string::npos)
            << "Node 1 should NOT use global cache_dir, got: " << node1CacheDir;
    }

    // Verify node 2 (without explicit CACHE_DIR) uses the global --cache_dir
    {
        std::shared_ptr<GenAiServable> servable2;
        ASSERT_EQ(initializeGenAiServable(servable2, config.node(1), ""), StatusCode::OK);
        auto properties2 = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable2->getProperties());
        ASSERT_EQ(properties2->pluginConfig.count("CACHE_DIR"), 1);
        std::string node2CacheDir = properties2->pluginConfig["CACHE_DIR"].as<std::string>();
        ASSERT_EQ(node2CacheDir, globalCacheDir)
            << "Node 2 should have global CACHE_DIR applied, got: " << node2CacheDir;
    }
    // GlobalCacheDirGuard restores the global cache_dir on scope exit.
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsMultipleNodesCacheDirPrecedence) {
    LLMNodeOptionsMultipleNodesCacheDirPrecedence(modelsPath);
}

TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsMultipleNodesCacheDirPrecedence) {
    LLMNodeOptionsMultipleNodesCacheDirPrecedence(modelsPath);
}

// Verifies that when multiple LLM models are loaded from config.json with CLI --cache_dir,
// each model receives the correct cache_dir: explicit CACHE_DIR in plugin_config takes
// precedence, otherwise the global --cache_dir is applied. Tests real-world scenario where
// users have separate LLM models defined in different directories in a single config.json.
// Regression test for openvinotoolkit/model_server#4230.
void LLMModelsFromConfigJsonMultipleCacheDirPrecedence(std::string& modelsPath) {
    const std::string resolvedModelsPath = getGenericFullPathForSrcTest(modelsPath);
    ASSERT_TRUE(std::filesystem::exists(resolvedModelsPath))
        << "LLM test models path does not exist: " << resolvedModelsPath;

    // Create temporary directories and graph.pbtxt files
    const std::string tmpDirLinux = "/tmp/LLMModelsFromConfigJson_" +
                                    std::to_string(std::time(nullptr)) + "_" +
                                    std::to_string(std::rand());
    const std::string tmpDir = getGenericFullPathForTmp(tmpDirLinux);

    // Restore global cache_dir and remove all temporary files/dirs on scope exit.
    GlobalCacheDirGuard cacheDirGuard(tmpDir);

    std::filesystem::create_directories(tmpDir);

    std::string model1Dir = tmpDir + "/model1";
    std::string model2Dir = tmpDir + "/model2";
    std::filesystem::create_directories(model1Dir);
    std::filesystem::create_directories(model2Dir);

    const std::string globalCacheDir = tmpDir + "/global_cache";
    const std::string nodeCacheDir = tmpDir + "/node1_cache";

    // Create graph.pbtxt for model1 with explicit CACHE_DIR
    std::string graph1Pbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
            name: "llmNode"
            calculator: "HttpLLMCalculator"
            input_stream: "LOOPBACK:loopback"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "LLM_NODE_RESOURCES:llm"
            output_stream: "LOOPBACK:loopback"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            input_stream_info: {
                tag_index: 'LOOPBACK:0',
                back_edge: true
            }
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    models_path: ")" +
                              resolvedModelsPath + R"("
                    pipeline_type: LM
                    plugin_config: '{"CACHE_DIR": ")" +
                              nodeCacheDir + R"("}'
                }
            }
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                        sync_set {
                            tag_index: "LOOPBACK:0"
                        }
                    }
                }
            }
        }
    )";
    adjustConfigForTargetPlatform(graph1Pbtxt);
    std::ofstream graph1File(model1Dir + "/graph.pbtxt");
    graph1File << graph1Pbtxt;
    graph1File.close();

    // Create graph.pbtxt for model2 WITHOUT explicit CACHE_DIR (should use global --cache_dir)
    std::string graph2Pbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
            name: "llmNode"
            calculator: "HttpLLMCalculator"
            input_stream: "LOOPBACK:loopback"
            input_stream: "HTTP_REQUEST_PAYLOAD:input"
            input_side_packet: "LLM_NODE_RESOURCES:llm"
            output_stream: "LOOPBACK:loopback"
            output_stream: "HTTP_RESPONSE_PAYLOAD:output"
            input_stream_info: {
                tag_index: 'LOOPBACK:0',
                back_edge: true
            }
            node_options: {
                [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                    models_path: ")" +
                              resolvedModelsPath + R"("
                    pipeline_type: LM
                }
            }
            input_stream_handler {
                input_stream_handler: "SyncSetInputStreamHandler",
                options {
                    [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                        sync_set {
                            tag_index: "LOOPBACK:0"
                        }
                    }
                }
            }
        }
    )";
    adjustConfigForTargetPlatform(graph2Pbtxt);
    std::ofstream graph2File(model2Dir + "/graph.pbtxt");
    graph2File << graph2Pbtxt;
    graph2File.close();

    // Create config.json with two graph definitions in separate directories.
    std::string configContent = R"JSON({
        "model_config_list": [
            {
                "config": {
                    "name": "model1_with_cache_dir",
                    "base_path": ")JSON" +
                                model1Dir + R"JSON(",
                    "graph_path": "graph.pbtxt"
                }
            },
            {
                "config": {
                    "name": "model2_without_cache_dir",
                    "base_path": ")JSON" +
                                model2Dir + R"JSON(",
                    "graph_path": "graph.pbtxt"
                }
            }
        ]
    })JSON";

    std::string configPath = tmpDir + "/config.json";
    std::ofstream configFile(configPath);
    configFile << configContent;
    configFile.close();

    // Set global cache_dir via CLI
    char* n_argv[] = {(char*)"ovms", (char*)"--config_path", (char*)configPath.c_str(), (char*)"--rest_port", (char*)"8080", (char*)"--cache_dir", (char*)globalCacheDir.c_str()};
    int arg_count = 7;
    ovms::Config::instance().parse(arg_count, n_argv);
    ASSERT_EQ(ovms::Config::instance().cacheDir(), globalCacheDir);

    ConstructorEnabledModelManager manager;
    auto configStatus = manager.loadConfig(configPath);
    ASSERT_TRUE(configStatus.ok()) << "Failed to load config.json: " << configStatus.string();

    // Get graph definitions created by ModelManager from config.json.
    auto* model1Def = manager.getMediapipeFactory().findDefinitionByName("model1_with_cache_dir");
    ASSERT_NE(model1Def, nullptr);
    auto* model2Def = manager.getMediapipeFactory().findDefinitionByName("model2_without_cache_dir");
    ASSERT_NE(model2Def, nullptr);

    auto readFileToString = [](const std::string& path) {
        std::ifstream input(path);
        std::stringstream ss;
        ss << input.rdbuf();
        return ss.str();
    };

    // Parse and validate model1 graph (with explicit CACHE_DIR in plugin_config)
    {
        ::mediapipe::CalculatorGraphConfig config1;
        std::string graphConfig1 = readFileToString(model1Def->getMediapipeGraphConfig().getGraphPath());
        ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(graphConfig1, &config1))
            << "Failed to parse model1 graph config";
        ASSERT_GT(config1.node_size(), 0) << "Model1 should have at least one node";

        // Initialize the servable and check its properties
        std::shared_ptr<GenAiServable> servable1;
        auto initStatus1 = initializeGenAiServable(servable1, config1.node(0), "model1_with_cache_dir");
        ASSERT_EQ(initStatus1, StatusCode::OK) << "Failed to initialize model1: " << initStatus1.string();

        auto properties1 = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable1->getProperties());
        ASSERT_EQ(properties1->pluginConfig.count("CACHE_DIR"), 1)
            << "Model1 should have CACHE_DIR in pluginConfig";
        std::string model1CacheDir = properties1->pluginConfig["CACHE_DIR"].as<std::string>();
        ASSERT_NE(model1CacheDir.find("node1_cache"), std::string::npos)
            << "Model1 should use explicit node CACHE_DIR, got: " << model1CacheDir;
        ASSERT_EQ(model1CacheDir.find("global_cache"), std::string::npos)
            << "Model1 should NOT use global cache_dir, got: " << model1CacheDir;
    }

    // Parse and validate model2 graph (without explicit CACHE_DIR, should use global)
    {
        ::mediapipe::CalculatorGraphConfig config2;
        std::string graphConfig2 = readFileToString(model2Def->getMediapipeGraphConfig().getGraphPath());
        ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(graphConfig2, &config2))
            << "Failed to parse model2 graph config";
        ASSERT_GT(config2.node_size(), 0) << "Model2 should have at least one node";

        // Initialize the servable and check its properties
        std::shared_ptr<GenAiServable> servable2;
        auto initStatus2 = initializeGenAiServable(servable2, config2.node(0), "model2_without_cache_dir");
        ASSERT_EQ(initStatus2, StatusCode::OK) << "Failed to initialize model2: " << initStatus2.string();

        auto properties2 = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable2->getProperties());
        ASSERT_EQ(properties2->pluginConfig.count("CACHE_DIR"), 1)
            << "Model2 should have CACHE_DIR in pluginConfig (applied from global --cache_dir)";
        std::string model2CacheDir = properties2->pluginConfig["CACHE_DIR"].as<std::string>();
        ASSERT_EQ(model2CacheDir, globalCacheDir)
            << "Model2 should use global cache_dir, got: " << model2CacheDir;
    }

    // GlobalCacheDirGuard restores global cache_dir and removes tmpDir on scope exit
}

TEST_F(LLMOptionsHttpTest, LLMModelsFromConfigJsonMultipleCacheDirPrecedence) {
    LLMModelsFromConfigJsonMultipleCacheDirPrecedence(modelsPath);
}

void LLMNodeOptionsCheckNonDefault(std::string& modelsPath) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ")" +
                            modelsPath + R"("
                max_num_batched_tokens: 1024
                cache_size: 1
                max_num_seqs: 95
                dynamic_split_fuse: false
                enable_prefix_caching: true
                max_tokens_limit: 700
                best_of_limit: 3
                cache_eviction_config: {start_size: 32, recent_size: 128, max_cache_size: 672, aggregation_mode: NORM_SUM, apply_rotation: true}
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
    auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());

    ASSERT_EQ(properties->schedulerConfig.max_num_batched_tokens, 1024);
    ASSERT_EQ(properties->schedulerConfig.cache_size, 1);
    ASSERT_EQ(properties->schedulerConfig.dynamic_split_fuse, false);
    ASSERT_EQ(properties->schedulerConfig.max_num_seqs, 95);
    ASSERT_EQ(properties->schedulerConfig.enable_prefix_caching, true);
    ASSERT_EQ(properties->maxTokensLimit, 700);
    ASSERT_EQ(properties->bestOfLimit, 3);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.get_start_size(), 32);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.get_recent_size(), 128);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.get_max_cache_size(), 672);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.aggregation_mode, ov::genai::AggregationMode::NORM_SUM);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.apply_rotation, true);
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckNonDefault) {
    LLMNodeOptionsCheckNonDefault(modelsPath);
}

TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsCheckNonDefault) {
    LLMNodeOptionsCheckNonDefault(modelsPath);
}

void LLMNodeOptionsCheckDefaultSparseAttentionConfig(std::string& modelsPath) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ")" +
                            modelsPath + R"("
               sparse_attention_config: {}
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
    auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());

    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.mode, ov::genai::SparseAttentionMode::TRISHAPE);
    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.num_last_dense_tokens_in_prefill, 100);
    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.num_retained_start_tokens_in_cache, 128);
    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.num_retained_recent_tokens_in_cache, 1920);
    ASSERT_NEAR(properties->schedulerConfig.sparse_attention_config.xattention_threshold, 0.8, 1e-6);
    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.xattention_block_size, 64);
    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.xattention_stride, 8);
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckDefaultSparseAttentionConfig) {
    LLMNodeOptionsCheckDefaultSparseAttentionConfig(modelsPath);
}

TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsCheckDefaultSparseAttentionConfig) {
    LLMNodeOptionsCheckDefaultSparseAttentionConfig(modelsPath);
}

void LLMNodeOptionsCheckNonDefaultSparseAttentionConfig(std::string& modelsPath) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ")" +
                            modelsPath + R"("
               sparse_attention_config: {
                   mode: XATTENTION
                   num_last_dense_tokens_in_prefill: 101
                   num_retained_start_tokens_in_cache: 129
                   num_retained_recent_tokens_in_cache: 1921
                   xattention_threshold: 0.9
                   xattention_block_size: 65
                   xattention_stride: 9
               }
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
    auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());

    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.mode, ov::genai::SparseAttentionMode::XATTENTION);
    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.num_last_dense_tokens_in_prefill, 101);
    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.num_retained_start_tokens_in_cache, 129);
    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.num_retained_recent_tokens_in_cache, 1921);
    ASSERT_NEAR(properties->schedulerConfig.sparse_attention_config.xattention_threshold, 0.9, 1e-6);
    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.xattention_block_size, 65);
    ASSERT_EQ(properties->schedulerConfig.sparse_attention_config.xattention_stride, 9);
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckNonDefaultSparseAttentionConfig) {
    LLMNodeOptionsCheckNonDefaultSparseAttentionConfig(modelsPath);
}

TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsCheckNonDefaultSparseAttentionConfig) {
    LLMNodeOptionsCheckNonDefaultSparseAttentionConfig(modelsPath);
}

void LLMNodeOptionsCheckDefaultCacheEvictionConfig(std::string& modelsPath) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ")" +
                            modelsPath + R"("
               cache_eviction_config: {
                   start_size: 1
                   recent_size: 2
                   max_cache_size: 4
               }
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
    auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());

    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.aggregation_mode, ov::genai::AggregationMode::NORM_SUM);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.get_start_size(), 1);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.get_recent_size(), 2);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.get_max_cache_size(), 4);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.apply_rotation, false);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.snapkv_window_size, 8);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.kvcrush_config.anchor_point_mode, ov::genai::KVCrushAnchorPointMode::RANDOM);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.kvcrush_config.budget, 0);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.kvcrush_config.rng_seed, 0);
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckDefaultCacheEvictionConfig) {
    LLMNodeOptionsCheckDefaultCacheEvictionConfig(modelsPath);
}

TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsCheckDefaultCacheEvictionConfig) {
    LLMNodeOptionsCheckDefaultCacheEvictionConfig(modelsPath);
}

void LLMNodeOptionsCheckNonDefaultCacheEvictionConfig(std::string& modelsPath) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: ")" +
                            modelsPath + R"("
                cache_eviction_config: {
                    start_size: 32
                    recent_size: 128
                    max_cache_size: 672
                    aggregation_mode: SUM
                    apply_rotation: true
                    snapkv_window_size: 16
                    kv_crush_config: {
                        anchor_point_mode: ONES
                        budget: 1
                        rng_seed: 42
                    }
                }
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
    auto properties = std::static_pointer_cast<ContinuousBatchingServableProperties>(servable->getProperties());

    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.aggregation_mode, ov::genai::AggregationMode::SUM);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.get_start_size(), 32);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.get_recent_size(), 128);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.get_max_cache_size(), 672);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.apply_rotation, true);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.snapkv_window_size, 16);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.kvcrush_config.anchor_point_mode, ov::genai::KVCrushAnchorPointMode::ONES);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.kvcrush_config.budget, 1);
    ASSERT_EQ(properties->schedulerConfig.cache_eviction_config.kvcrush_config.rng_seed, 42);
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckNonDefaultCacheEvictionConfig) {
    LLMNodeOptionsCheckNonDefaultCacheEvictionConfig(modelsPath);
}

TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsCheckNonDefaultCacheEvictionConfig) {
    LLMNodeOptionsCheckNonDefaultCacheEvictionConfig(modelsPath);
}

// Speculative decoding is not supported in VLM pipelines, currently not using parameters for this test
TEST_F(LLMOptionsHttpTest, LLMNodeOptionsSpeculativeDecodingSanityCheck) {
    std::string testPbtxt = R"(
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"

        node: {
        name: "llmNode"
        calculator: "HttpLLMCalculator"
        input_stream: "LOOPBACK:loopback"
        input_stream: "HTTP_REQUEST_PAYLOAD:input"
        input_side_packet: "LLM_NODE_RESOURCES:llm"
        output_stream: "LOOPBACK:loopback"
        output_stream: "HTTP_RESPONSE_PAYLOAD:output"
        input_stream_info: {
            tag_index: 'LOOPBACK:0',
            back_edge: true
        }
        node_options: {
            [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
                models_path: "/ovms/src/test/llm_testing/facebook/opt-125m"
                draft_models_path: "/ovms/src/test/llm_testing/facebook/opt-125m"
            }
        }
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
            [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                sync_set {
                tag_index: "LOOPBACK:0"
                }
            }
            }
        }
        }
    )";
    adjustConfigForTargetPlatform(testPbtxt);
    ::mediapipe::CalculatorGraphConfig config;
    ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
    std::shared_ptr<GenAiServable> servable;
    ASSERT_EQ(initializeGenAiServable(servable, config.node(0), ""), StatusCode::OK);
}

class GetPromptTokensString : public ::testing::Test {
public:
    std::string expectedTokensString;
    std::vector<std::vector<size_t>> shapes{{10}};
    void SetUp() {
        expectedTokensString = "prompt_token_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]";
    }
};

TEST_F(GetPromptTokensString, typesTestF32) {
    std::vector<ov::element::Type_t> precisions{ov::element::Type_t::f32};
    std::vector<float> tensorsDataF{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    for (auto precision : precisions) {
        std::stringstream ss;
        ss << "Testing precision: " << precision << std::endl;
        std::cout << ss.str();
        ov::Tensor tensor = createTensorWithNoDataOwnership(precision, shapes[0], tensorsDataF.data());
        ASSERT_EQ(expectedTokensString, getPromptTokensString(tensor));
    }
}

TEST_F(GetPromptTokensString, typesTestF64) {
    std::vector<ov::element::Type_t> precisions{ov::element::Type_t::f64};
    std::vector<double> tensorsDataD{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (auto precision : precisions) {
        std::stringstream ss;
        ss << "Testing precision: " << precision << std::endl;
        std::cout << ss.str();
        ov::Tensor tensor = createTensorWithNoDataOwnership(precision, shapes[0], tensorsDataD.data());
        ASSERT_EQ(expectedTokensString, getPromptTokensString(tensor));
    }
}

TEST_F(GetPromptTokensString, typesTestI32) {
    std::vector<ov::element::Type_t> precisions{ov::element::Type_t::i32};
    std::vector<int> tensorsDataI{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (auto precision : precisions) {
        std::stringstream ss;
        ss << "Testing precision: " << precision << std::endl;
        std::cout << ss.str();
        ov::Tensor tensor = createTensorWithNoDataOwnership(precision, shapes[0], tensorsDataI.data());
        ASSERT_EQ(expectedTokensString, getPromptTokensString(tensor));
    }
}

TEST_F(GetPromptTokensString, typesTestI64) {
    std::vector<ov::element::Type_t> precisions{ov::element::Type_t::i64};
    std::vector<int64_t> tensorsDataI64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (auto precision : precisions) {
        std::stringstream ss;
        ss << "Testing precision: " << precision << std::endl;
        std::cout << ss.str();
        ov::Tensor tensor = createTensorWithNoDataOwnership(precision, shapes[0], tensorsDataI64.data());
        ASSERT_EQ(expectedTokensString, getPromptTokensString(tensor));
    }
}

TEST_F(GetPromptTokensString, typesTestI16) {
    std::vector<ov::element::Type_t> precisions{ov::element::Type_t::i16};
    std::vector<int16_t> tensorsDataI16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (auto precision : precisions) {
        std::stringstream ss;
        ss << "Testing precision: " << precision << std::endl;
        std::cout << ss.str();
        ov::Tensor tensor = createTensorWithNoDataOwnership(precision, shapes[0], tensorsDataI16.data());
        ASSERT_EQ(expectedTokensString, getPromptTokensString(tensor));
    }
}

class GetPromptTokensStringNegative : public GetPromptTokensString {
public:
    void SetUp() {
        expectedTokensString = "Could not pack input tokens for element type: f16";
    }
};

TEST_F(GetPromptTokensStringNegative, unsupportedTypesTestF16) {
    std::vector<ov::element::Type_t> precisions{ov::element::Type_t::f16};
    std::vector<float> tensorsDataF{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (auto precision : precisions) {
        std::stringstream ss;
        ss << "Testing precision: " << precision << std::endl;
        std::cout << ss.str();
        ov::Tensor tensor = createTensorWithNoDataOwnership(precision, shapes[0], tensorsDataF.data());
        ASSERT_EQ(expectedTokensString, getPromptTokensString(tensor));
    }
}

TEST_F(GetPromptTokensStringNegative, unsupportedTypesTestBool) {
    std::vector<ov::element::Type_t> precisions{ov::element::Type_t::boolean};
    std::vector<float> tensorsDataF{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    expectedTokensString = "Could not pack input tokens for element type: boolean";

    for (auto precision : precisions) {
        std::stringstream ss;
        ss << "Testing precision: " << precision << std::endl;
        std::cout << ss.str();
        ov::Tensor tensor = createTensorWithNoDataOwnership(precision, shapes[0], tensorsDataF.data());
        ASSERT_EQ(expectedTokensString, getPromptTokensString(tensor));
    }
}

#include "../../llm/language_model/legacy/servable.hpp"

class MockLegacyServable : public ovms::LegacyServable {
public:
    absl::Status callValidateInputComplianceWithProperties(const ov::Tensor& inputIds) {
        return validateInputComplianceWithProperties(inputIds);
    }
};

class IsolatedServableTests : public ::testing::Test {
public:
    MockLegacyServable legacyServable;

protected:
    void SetUp() override {
        // Code here will be called immediately after the constructor (right before each test).
    }

    void TearDown() override {
        // Code here will be called immediately after each test (right before the destructor).
    }
};

TEST_F(IsolatedServableTests, PromtSizeExceedsDefaultMaxPromptLenNPU) {
    legacyServable.getProperties()->device = "NPU";  // Simulate NPU device
    ovms::LegacyServableExecutionContext executionContext;
    // Create an ov::Tensor object with random data
    size_t dataSize = 1025;
    std::vector<float> randomData(dataSize);
    std::fill(randomData.begin(), randomData.end(), 1.0f);
    ov::Tensor tensor(ov::element::f32, {1, dataSize}, randomData.data());
    executionContext.inputRequest.inputIds = tensor;
    auto status = legacyServable.callValidateInputComplianceWithProperties(executionContext.inputRequest.inputIds);
    ASSERT_EQ(status, absl::InvalidArgumentError("Input length exceeds the maximum allowed length"));
}

TEST_F(IsolatedServableTests, PromtSizeExceedsNonDefaultMaxPromptLenNPU) {
    legacyServable.getProperties()->device = "NPU";                                                              // Simulate NPU device
    std::static_pointer_cast<LegacyServableProperties>(legacyServable.getProperties())->maxPromptLength = 4096;  // Set max prompt length to 4096
    ovms::LegacyServableExecutionContext executionContext;
    // Create an ov::Tensor object with random data
    size_t dataSize = 5025;
    std::vector<float> randomData(dataSize);
    std::fill(randomData.begin(), randomData.end(), 1.0f);
    ov::Tensor tensor(ov::element::f32, {1, dataSize}, randomData.data());
    executionContext.inputRequest.inputIds = tensor;
    auto status = legacyServable.callValidateInputComplianceWithProperties(executionContext.inputRequest.inputIds);
    ASSERT_EQ(status, absl::InvalidArgumentError("Input length exceeds the maximum allowed length"));
}

TEST_F(IsolatedServableTests, PromtSizeBetweenDefaultAndNonDefaultMaxPromptLenNPU) {
    legacyServable.getProperties()->device = "NPU";                                                              // Simulate NPU device
    std::static_pointer_cast<LegacyServableProperties>(legacyServable.getProperties())->maxPromptLength = 4096;  // Set max prompt length to 4096
    ovms::LegacyServableExecutionContext executionContext;
    // Create an ov::Tensor object with random data
    size_t dataSize = 3025;
    std::vector<float> randomData(dataSize);
    std::fill(randomData.begin(), randomData.end(), 1.0f);
    ov::Tensor tensor(ov::element::f32, {1, dataSize}, randomData.data());
    executionContext.inputRequest.inputIds = tensor;
    auto status = legacyServable.callValidateInputComplianceWithProperties(executionContext.inputRequest.inputIds);
    ASSERT_EQ(status, absl::OkStatus());
}

// TODO: Add missing tests for reading max prompt len property from configuration

class LLMStartWithTaskParameter : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;
    std::string srcModelDir = getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/HuggingFaceTB/SmolLM2-360M-Instruct");
#ifdef __linux__
    std::string tempDir;
    std::string modelDir;
    std::string graphPath;
#else
    std::string modelDir = srcModelDir;
    std::string graphPath = modelDir + "/graph.pbtxt";
    std::string graphPathRenamed = modelDir + "/graph.pbtxt.bak";
#endif

    void SetUp() override {
        GraphExport::clearInMemoryGraphContent();
#ifdef __linux__
        tempDir = std::filesystem::temp_directory_path().string() + "/LLMStartWithTaskParameter_" + ::testing::UnitTest::GetInstance()->current_test_info()->name();
        std::filesystem::remove_all(tempDir);
        std::filesystem::copy(srcModelDir, tempDir, std::filesystem::copy_options::recursive);
        modelDir = tempDir;
        graphPath = modelDir + "/graph.pbtxt";
#endif
    }
    void TearDown() override {
        ovms::Server& server = ovms::Server::instance();
        server.setShutdownRequest(1);
        if (t && t->joinable())
            t->join();
        server.setShutdownRequest(0);
        GraphExport::clearInMemoryGraphContent();
#ifdef __linux__
        std::filesystem::remove_all(tempDir);
#else
        // Restore graph.pbtxt if it was renamed
        if (std::filesystem::exists(graphPathRenamed)) {
            if (std::filesystem::exists(graphPath)) {
                std::filesystem::remove(graphPath);
            }
            std::filesystem::rename(graphPathRenamed, graphPath);
        }
#endif
    }
};

std::unique_ptr<std::thread> LLMStartWithTaskParameter::t = nullptr;

TEST_F(LLMStartWithTaskParameter, StartWithModelPathAndTaskWithoutGraphFile) {
#ifdef __linux__
    // On Linux models are on readonly FS - we use a temp copy with graph.pbtxt removed
    std::filesystem::remove(graphPath);
#else
    // On Windows models are on RW FS - rename graph.pbtxt so we can check it's not recreated
    if (std::filesystem::exists(graphPath)) {
        std::filesystem::rename(graphPath, graphPathRenamed);
    }
#endif

    std::string port = "9173";
    ovms::Server& server = ovms::Server::instance();
    ::SetUpServer(t, server, port,
        modelDir.c_str(),
        "SmolLM2",
        60,
        "text_generation");
    ASSERT_EQ(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::INITIALIZED);
    ASSERT_FALSE(std::filesystem::exists(graphPath)) << "graph.pbtxt should not be created when using --task with --model_path";
}

TEST_F(LLMStartWithTaskParameter, StartWithModelPathAndTaskDoesNotModifyExistingGraph) {
    ASSERT_TRUE(std::filesystem::exists(graphPath)) << "graph.pbtxt must exist for this test";
    auto modTimeBefore = std::filesystem::last_write_time(graphPath);

    std::string port = "9174";
    ovms::Server& server = ovms::Server::instance();
    ::SetUpServer(t, server, port,
        modelDir.c_str(),
        "SmolLM2",
        60,
        "text_generation");
    ASSERT_EQ(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::INITIALIZED);

    auto modTimeAfter = std::filesystem::last_write_time(graphPath);
    ASSERT_EQ(modTimeBefore, modTimeAfter) << "graph.pbtxt should not be modified when using --task with --model_path";
}

TEST_F(LLMStartWithTaskParameter, StartWithModelPathAndTaskAndValidPipelineType) {
    std::string port = "9175";
    ovms::Server& server = ovms::Server::instance();
    server.setShutdownRequest(0);
    randomizeAndEnsureFree(port);
    std::string fullModelPath = getGenericFullPathForSrcTest(modelDir.c_str());
    char* argv[] = {(char*)"ovms",
        (char*)"--model_name", (char*)"SmolLM2",
        (char*)"--model_path", (char*)fullModelPath.c_str(),
        (char*)"--port", (char*)port.c_str(),
        (char*)"--task", (char*)"text_generation",
        (char*)"--pipeline_type", (char*)"LM_CB"};
    int argc = 11;
    t.reset(new std::thread([&argc, &argv, &server]() {
        EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
    }));
    EnsureServerStartedWithTimeout(server, 60);
    ASSERT_EQ(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::INITIALIZED);
}

TEST_F(LLMStartWithTaskParameter, StartWithModelPathAndTaskAndInvalidPipelineType) {
    std::string port = "9176";
    ovms::Server& server = ovms::Server::instance();
    server.setShutdownRequest(0);
    randomizeAndEnsureFree(port);
    std::string fullModelPath = getGenericFullPathForSrcTest(modelDir.c_str());
    char* argv[] = {(char*)"ovms",
        (char*)"--model_name", (char*)"SmolLM2",
        (char*)"--model_path", (char*)fullModelPath.c_str(),
        (char*)"--port", (char*)port.c_str(),
        (char*)"--task", (char*)"text_generation",
        (char*)"--pipeline_type", (char*)"invalid"};
    int argc = 11;
    t.reset(new std::thread([&argc, &argv, &server]() {
        EXPECT_NE(EXIT_SUCCESS, server.start(argc, argv));
    }));
    // Validation failure should complete quickly
    if (t && t->joinable())
        t->join();
    ASSERT_NE(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::INITIALIZED)
        << "Server should not start with invalid pipeline_type";
}

// Unit tests for BaseGenerationConfigBuilder multinomial sampling defaults

TEST(BaseGenerationConfigBuilderTest, TopKDefaultedTo40WhenSamplingEnabled) {
    ov::genai::GenerationConfig baseConfig;
    BaseGenerationConfigBuilder builder{baseConfig, /*enableToolGuidedGeneration=*/false, DecodingMethod::STANDARD};
    OpenAIRequest request;
    request.temperature = 1.0f;  // enables do_sample; topK not set
    builder.parseConfigFromRequest(request);
    EXPECT_EQ(builder.getConfig().top_k, 40u);
}

TEST(BaseGenerationConfigBuilderTest, TopKPreservedWhenExplicitlySet) {
    ov::genai::GenerationConfig baseConfig;
    BaseGenerationConfigBuilder builder{baseConfig, /*enableToolGuidedGeneration=*/false, DecodingMethod::STANDARD};
    OpenAIRequest request;
    request.temperature = 1.0f;
    request.topK = 10;
    builder.parseConfigFromRequest(request);
    EXPECT_EQ(builder.getConfig().top_k, 10u);
}

TEST(BaseGenerationConfigBuilderTest, TopKMinusOneMapsToInactive) {
    ov::genai::GenerationConfig baseConfig;
    BaseGenerationConfigBuilder builder{baseConfig, /*enableToolGuidedGeneration=*/false, DecodingMethod::STANDARD};
    OpenAIRequest request;
    request.temperature = 1.0f;
    request.topK = -1;  // sentinel: consider all tokens
    builder.parseConfigFromRequest(request);
    EXPECT_EQ(builder.getConfig().top_k, std::numeric_limits<size_t>::max());
}

TEST(BaseGenerationConfigBuilderTest, TopKNotChangedWhenSamplingDisabled) {
    ov::genai::GenerationConfig baseConfig;
    BaseGenerationConfigBuilder builder{baseConfig, /*enableToolGuidedGeneration=*/false, DecodingMethod::STANDARD};
    OpenAIRequest request;
    request.temperature = 0.0f;  // greedy decoding, do_sample = false
    builder.parseConfigFromRequest(request);
    EXPECT_EQ(builder.getConfig().top_k, std::numeric_limits<size_t>::max());
}

TEST(BaseGenerationConfigBuilderTest, SeedRandomizedWhenOmittedDuringSampling) {
    ov::genai::GenerationConfig baseConfig;
    OpenAIRequest request;
    request.temperature = 1.0f;  // enables do_sample; seed not set → must be randomized

    // Parse the same request twice — seeds must differ (non-deterministic per request)
    BaseGenerationConfigBuilder builder1{baseConfig, /*enableToolGuidedGeneration=*/false, DecodingMethod::STANDARD};
    builder1.parseConfigFromRequest(request);
    const size_t seed1 = builder1.getConfig().rng_seed;

    BaseGenerationConfigBuilder builder2{baseConfig, /*enableToolGuidedGeneration=*/false, DecodingMethod::STANDARD};
    builder2.parseConfigFromRequest(request);
    const size_t seed2 = builder2.getConfig().rng_seed;

    EXPECT_NE(seed1, 0u);
    EXPECT_NE(seed2, 0u);
    EXPECT_NE(seed1, seed2) << "Expected different seeds for successive omitted-seed requests";
}

TEST(BaseGenerationConfigBuilderTest, SeedPreservedWhenExplicitlySet) {
    ov::genai::GenerationConfig baseConfig;
    BaseGenerationConfigBuilder builder{baseConfig, /*enableToolGuidedGeneration=*/false, DecodingMethod::STANDARD};
    OpenAIRequest request;
    request.temperature = 1.0f;
    request.seed = 42u;
    builder.parseConfigFromRequest(request);
    EXPECT_EQ(builder.getConfig().rng_seed, 42u);
}

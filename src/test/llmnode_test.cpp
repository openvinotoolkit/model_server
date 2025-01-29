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

#include "../http_rest_api_handler.hpp"
#include "../http_status_code.hpp"
#include "../json_parser.hpp"
#include "../llm/apis/openai_completions.hpp"
#include "../llm/llm_executor.hpp"
#include "../llm/llmnoderesources.hpp"
#include "../ov_utils.hpp"
#include "../server.hpp"
#include "opencv2/opencv.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "test_http_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;

static std::atomic<uint64_t> currentRequestId = 0;

class LLMFlowHttpTest : public ::testing::Test {
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
        ::SetUpServer(t, server, port, getGenericFullPathForSrcTest("/ovms/src/test/llm/config_llm_dummy_kfs.json").c_str());
        auto start = std::chrono::high_resolution_clock::now();
        const int numberOfRetries = 5;
        while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < numberOfRetries)) {
        }

        try {
            plugin_config_t tokenizerPluginConfig = {};
            std::string device = "CPU";
            ov::genai::SchedulerConfig schedulerConfig;
            schedulerConfig.max_num_batched_tokens = 256;
            schedulerConfig.cache_size = 1;
            schedulerConfig.dynamic_split_fuse = true;
            schedulerConfig.max_num_seqs = 256;
            plugin_config_t pluginConfig;
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
std::shared_ptr<ov::genai::ContinuousBatchingPipeline> LLMFlowHttpTest::cbPipe;
std::shared_ptr<LLMExecutorWrapper> LLMFlowHttpTest::llmExecutorWrapper;
std::unique_ptr<std::thread> LLMFlowHttpTest::t;

// --------------------------------------- OVMS LLM nodes tests

TEST_F(LLMFlowHttpTest, writeLogprobs) {
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

TEST_F(LLMFlowHttpTest, unaryCompletionsJson) {
    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.num_beams = 16;
    ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
    ASSERT_EQ(config.num_return_sequences, expectedMessages.size());
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "best_of": 16,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["text"].IsString());
        EXPECT_STREQ(choice["text"].GetString(), expectedMessages[i].c_str());
        ASSERT_EQ(choice["index"], i++);
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), "llmDummyKFS");
    EXPECT_STREQ(parsedResponse["object"].GetString(), "text_completion");
}

TEST_F(LLMFlowHttpTest, unaryCompletionsJsonSpeculativeDecoding) {
    config.max_new_tokens = 10;
    config.temperature = 0;
    ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
    ASSERT_EQ(config.num_return_sequences, expectedMessages.size());
    // Static number of candidates
    std::string requestBody = R"(
        {
            "model": "llmDummySpeculativePipeline",
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
            "model": "llmDummySpeculativePipeline",
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

TEST_F(LLMFlowHttpTest, unaryCompletionsJsonEchoWithCompletion) {
    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.num_beams = 16;
    config.echo = true;
    ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
    ASSERT_EQ(config.num_return_sequences, expectedMessages.size());
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "best_of": 16,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?",
            "echo": true
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["text"].IsString());
        EXPECT_STREQ(choice["text"].GetString(), expectedMessages[i].c_str());
        EXPECT_TRUE(std::string(choice["text"].GetString()).find("What is OpenVINO?") != std::string::npos);
        EXPECT_EQ(std::string(choice["text"].GetString()).rfind("What is OpenVINO?", 0), 0);  // Check if prompt is at the beginning
        ASSERT_EQ(choice["index"], i++);
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), "llmDummyKFS");
    EXPECT_STREQ(parsedResponse["object"].GetString(), "text_completion");
}

TEST_F(LLMFlowHttpTest, streamCompletionsEchoWithCompletion) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": true,
            "seed" : 1,
            "max_tokens": 10,
            "echo": true,
            "prompt": "What is OpenVINO?"
        }
    )";
    std::vector<std::string> chunks;
    ON_CALL(*writer, PartialReply).WillByDefault([this, &chunks](std::string response) {
        rapidjson::Document d;
        std::string dataPrefix = "data:";
        ASSERT_STREQ(response.substr(0, dataPrefix.size()).c_str(), dataPrefix.c_str());
        size_t pos = response.find("\n");
        ASSERT_NE(pos, response.npos);
        rapidjson::ParseResult parsingSucceeded = d.Parse(response.substr(dataPrefix.size(), (pos - dataPrefix.size())).c_str());
        ASSERT_EQ(parsingSucceeded.Code(), 0);
        ASSERT_TRUE(d["choices"].IsArray());
        ASSERT_EQ(d["choices"].Capacity(), 1);
        int i = 0;
        for (auto& choice : d["choices"].GetArray()) {
            if (choice["finish_reason"].IsString()) {
                EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
            } else {
                ASSERT_TRUE(choice["finish_reason"].IsNull());
            }
            ASSERT_EQ(choice["index"], i++);
            ASSERT_FALSE(choice["logprobs"].IsObject());
            ASSERT_TRUE(choice["text"].IsString());
            chunks.push_back(std::string(choice["text"].GetString()));
        }
        EXPECT_STREQ(d["model"].GetString(), "llmDummyKFS");
        EXPECT_STREQ(d["object"].GetString(), "text_completion.chunk");
    });

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);

    // Since prompt is treated as a single entity and streamer returns chunk only after space or newline
    // we expect chunk with echoed prompt to contain space or new line at the end
    ASSERT_TRUE(chunks[0] == "What is OpenVINO?\n" || chunks[0] == "What is OpenVINO? ");
    ASSERT_GT(chunks.size(), 1);
}

TEST_F(LLMFlowHttpTest, unaryCompletionsJsonEchoOnly) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 0,
            "prompt": "What is OpenVINO?",
            "echo": true,
            "logprobs": 1
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        EXPECT_STREQ(choice["finish_reason"].GetString(), "length");

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

        ASSERT_TRUE(choice["text"].IsString());
        EXPECT_STREQ(choice["text"].GetString(), "What is OpenVINO?");
        ASSERT_EQ(choice["index"], i++);
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 0 /* max_tokens */);
    ASSERT_EQ(parsedResponse["usage"].GetObject()["prompt_tokens"].GetInt(), parsedResponse["choices"].GetArray()[0]["logprobs"].GetObject()["token_logprobs"].Size());
    EXPECT_STREQ(parsedResponse["model"].GetString(), "llmDummyKFS");
    EXPECT_STREQ(parsedResponse["object"].GetString(), "text_completion");
}

TEST_F(LLMFlowHttpTest, streamCompletionsEchoOnly) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": true,
            "seed" : 1,
            "max_tokens": 0,
            "echo": true,
            "prompt": "What is OpenVINO?"
        }
    )";
    EXPECT_CALL(*writer, PartialReply(::testing::_)).WillOnce([this](std::string response) {
        rapidjson::Document d;
        std::string dataPrefix = "data:";
        ASSERT_STREQ(response.substr(0, dataPrefix.size()).c_str(), dataPrefix.c_str());
        size_t pos = response.find("\n");
        ASSERT_NE(pos, response.npos);
        rapidjson::ParseResult parsingSucceeded = d.Parse(response.substr(dataPrefix.size(), (pos - dataPrefix.size())).c_str());
        ASSERT_EQ(parsingSucceeded.Code(), 0);
        ASSERT_TRUE(d["choices"].IsArray());
        ASSERT_EQ(d["choices"].Capacity(), 1);
        int i = 0;
        for (auto& choice : d["choices"].GetArray()) {
            if (choice["finish_reason"].IsString()) {
                EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
            } else {
                ASSERT_TRUE(choice["finish_reason"].IsNull());
            }
            ASSERT_EQ(choice["index"], i++);
            ASSERT_FALSE(choice["logprobs"].IsObject());
            ASSERT_TRUE(choice["text"].IsString());
            EXPECT_STREQ(choice["text"].GetString(), "What is OpenVINO?");
        }
        EXPECT_STREQ(d["model"].GetString(), "llmDummyKFS");
        EXPECT_STREQ(d["object"].GetString(), "text_completion.chunk");
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, unaryCompletionsJsonFinishReasonLength) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "ignore_eos": true,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
        ASSERT_EQ(choice["index"], i++);
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["text"].IsString());
    }
    ASSERT_EQ(parsedResponse["model"], "llmDummyKFS");
    ASSERT_EQ(parsedResponse["object"], "text_completion");
}

TEST_F(LLMFlowHttpTest, unaryCompletionsJsonSingleStopString) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "ignore_eos": false,
            "max_tokens": 1000,
            "stop": ".",
            "include_stop_str_in_output": true,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        EXPECT_STREQ(choice["finish_reason"].GetString(), "stop");
        ASSERT_EQ(choice["index"], i++);
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["text"].IsString());
        auto text_size = std::string(choice["text"].GetString()).size();
        ASSERT_EQ(choice["text"].GetString()[text_size - 1], '.');
    }
    ASSERT_EQ(parsedResponse["model"], "llmDummyKFS");
    ASSERT_EQ(parsedResponse["object"], "text_completion");
}

TEST_F(LLMFlowHttpTest, unaryCompletionsJsonNFail) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "best_of": 2,
            "n": 3,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}
TEST_F(LLMFlowHttpTest, unaryCompletionsJsonN) {
    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.num_beams = 16;
    config.num_return_sequences = 8;
    config.echo = false;
    ASSERT_EQ(generateExpectedText("What is OpenVINO?"), 0);
    ASSERT_EQ(config.num_return_sequences, expectedMessages.size());
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "best_of": 16,
            "n": 8,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 8);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["text"].IsString());
        EXPECT_STREQ(choice["text"].GetString(), expectedMessages[i].c_str());
        ASSERT_EQ(choice["index"], i++);
    }
    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 8 * 5 /* n * max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), "llmDummyKFS");
    EXPECT_STREQ(parsedResponse["object"].GetString(), "text_completion");
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsJsonNFail) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "best_of" : 2,
            "n" : 3,
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsJsonN) {
    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.num_beams = 16;
    config.num_return_sequences = 8;
    config.echo = false;
    ASSERT_EQ(generateExpectedText("What is OpenVINO?", false), 0);
    ASSERT_EQ(config.num_return_sequences, expectedMessages.size());
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "best_of" : 16,
            "n" : 8,
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 8);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        ASSERT_EQ(choice["message"]["content"].GetString(), expectedMessages[i]);
        ASSERT_EQ(choice["index"], i++);
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 8 * 5 /* n * max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), "llmDummyKFS");
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

TEST_F(LLMFlowHttpTest, KFSApiRequestToChatCompletionsGraph) {
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
    std::vector<std::pair<std::string, std::string>> headers;
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", "/v2/models/llmDummyKFS/versions/1/infer", headers), ovms::StatusCode::OK);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM);
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsJson) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "best_of" : 16,
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
        ASSERT_EQ(choice["index"], i++);
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), "llmDummyKFS");
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsJsonSpeculativeDecoding) {
    // Setting lower max new tokens for this test to pass.
    // For some reason in this test, during second request processing, BOS token is added to the input.
    // It wasn't reproduced in real life scenario and shall be investigated.
    config.max_new_tokens = 8;
    config.temperature = 0;
    ASSERT_EQ(generateExpectedText("What is OpenVINO?", false), 0);
    ASSERT_EQ(config.num_return_sequences, expectedMessages.size());

    // Static number of candidates
    std::string requestBody = R"(
        {
            "model": "llmDummySpeculativePipeline",
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
    ASSERT_EQ(choice["message"]["content"].GetString(), expectedMessages[0]);

    // Dynamic number of candidates
    requestBody = R"(
        {
            "model": "llmDummySpeculativePipeline",
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
    ASSERT_EQ(choice["message"]["content"].GetString(), expectedMessages[0]);
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsJsonContentArray) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "best_of" : 16,
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
        ASSERT_EQ(choice["index"], i++);
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), "llmDummyKFS");
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsJsonContentArrayWithImage) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "best_of" : 16,
            "max_tokens": 5,
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is OpenVINO?"}, {"type": "image_url", "image_url": {"url":  "base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGLK27oAEAAA//8DYAHGgEvy5AAAAABJRU5ErkJggg=="}}]
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
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
        ASSERT_EQ(choice["index"], i++);
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }

    ASSERT_TRUE(parsedResponse["usage"].IsObject());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["prompt_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["completion_tokens"].IsInt());
    ASSERT_TRUE(parsedResponse["usage"].GetObject()["total_tokens"].IsInt());
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 5 /* max_tokens */);
    EXPECT_STREQ(parsedResponse["model"].GetString(), "llmDummyKFS");
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsJsonNMultipleStopStrings) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "best_of" : 4,
            "n": 4,
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 4);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        ASSERT_TRUE(choice["finish_reason"].IsString());
        EXPECT_STREQ(choice["finish_reason"].GetString(), "stop");
        ASSERT_EQ(choice["index"], i++);
        ASSERT_FALSE(choice["logprobs"].IsObject());
        ASSERT_TRUE(choice["message"].IsObject());
        ASSERT_TRUE(choice["message"]["content"].IsString());
        auto text_size = std::string(choice["message"]["content"].GetString()).size();
        ASSERT_TRUE(choice["message"]["content"].GetString()[text_size - 1] == '.' ||
                    choice["message"]["content"].GetString()[text_size - 1] == ',');
        EXPECT_STREQ(choice["message"]["role"].GetString(), "assistant");
    }
}

// TODO: Fails no idea why
TEST_F(LLMFlowHttpTest, DISABLED_unaryChatCompletionsJsonLogprobs) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    for (auto& choice : parsedResponse["choices"].GetArray()) {
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

TEST_F(LLMFlowHttpTest, unaryCompletionsJsonLogprobs) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "seed" : 1,
            "max_tokens": 5,
            "logprobs": 1,
            "prompt":  "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    for (auto& choice : parsedResponse["choices"].GetArray()) {
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

TEST_F(LLMFlowHttpTest, ChatCompletionsJsonLogprobsStream) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, CompletionsJsonLogprobsStream) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "logprobs": 2,
            "seed" : 1,
            "max_tokens": 1,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsStopStringBadType) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsIncludeStopStringInOutputBadType) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMFlowHttpTest, unaryCompletionsStopStringElementBadType) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "stop": [".", "OpenVINO", 1.92],
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsStopStringExceedingSize) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "stop": ["a", "b", "c", "d", "e"],
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMFlowHttpTest, unaryCompletionsStopStringEmpty) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "stop": [],
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMFlowHttpTest, streamBeamSearchCompletionsFail) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": true,
            "best_of": 2,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, streamBeamSearchChatCompletionsFail) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": true,
            "best_of": 2,
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
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, inferCompletionsStream) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": true,
            "seed" : 1,
            "max_tokens": 5,
            "ignore_eos": true,
            "prompt": "What is OpenVINO?"
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
        ASSERT_EQ(d["choices"].Capacity(), 1);
        int i = 0;
        for (auto& choice : d["choices"].GetArray()) {
            if (choice["finish_reason"].IsString()) {
                EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
            } else {
                ASSERT_TRUE(choice["finish_reason"].IsNull());
            }
            ASSERT_EQ(choice["index"], i++);
            ASSERT_FALSE(choice["logprobs"].IsObject());
            ASSERT_TRUE(choice["text"].IsString());
        }
        EXPECT_STREQ(d["model"].GetString(), "llmDummyKFS");
        EXPECT_STREQ(d["object"].GetString(), "text_completion.chunk");
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, inferChatCompletionsStream) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        ASSERT_EQ(d["choices"].Capacity(), 1);
        int i = 0;
        for (auto& choice : d["choices"].GetArray()) {
            if (choice["finish_reason"].IsString()) {
                EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
            } else {
                ASSERT_TRUE(choice["finish_reason"].IsNull());
            }
            ASSERT_EQ(choice["index"], i++);
            ASSERT_FALSE(choice["logprobs"].IsObject());
            ASSERT_TRUE(choice["delta"].IsObject());
            ASSERT_TRUE(choice["delta"]["content"].IsString());
        }
        EXPECT_STREQ(d["model"].GetString(), "llmDummyKFS");
        EXPECT_STREQ(d["object"].GetString(), "chat.completion.chunk");
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, unaryChatCompletionsStreamOptionsSetFail) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMFlowHttpTest, unaryCompletionsStreamOptionsSetFail) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "stream_options": { "include_usage": true },
            "seed" : 1,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMFlowHttpTest, streamChatCompletionsFinishReasonLength) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
}

// Potential sporadic - move to functional if problematic
TEST_F(LLMFlowHttpTest, streamChatCompletionsSingleStopString) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": true,
            "seed" : 1,
            "ignore_eos": false,
            "max_tokens": 1000,
            "stop": ".",
            "include_stop_str_in_output": true,
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_TRUE(responses.back().find("\"finish_reason\":\"stop\"") != std::string::npos);
    std::regex content_regex("\"content\":\".*\\.[ ]{0,1}\"");
    ASSERT_TRUE(std::regex_search(responses.back(), content_regex));
}

TEST_F(LLMFlowHttpTest, streamCompletionsFinishReasonLength) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
}

// Potential sporadic - move to functional if problematic
TEST_F(LLMFlowHttpTest, streamCompletionsSingleStopString) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_TRUE(responses.back().find("\"finish_reason\":\"stop\"") != std::string::npos);
    std::regex content_regex("\"text\":\".*\\.[ ]{0,1}\"");
    ASSERT_TRUE(std::regex_search(responses.back(), content_regex)) << responses.back();
}

TEST_F(LLMFlowHttpTest, streamChatCompletionsUsage) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_TRUE(responses.back().find("\"completion_tokens\":5") != std::string::npos);
    ASSERT_TRUE(responses.back().find("\"prompt_tokens\"") != std::string::npos);
    ASSERT_TRUE(responses.back().find("\"total_tokens\"") != std::string::npos);
    ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
}

TEST_F(LLMFlowHttpTest, streamCompletionsUsage) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_TRUE(responses.back().find("\"completion_tokens\":5") != std::string::npos);
    ASSERT_TRUE(responses.back().find("\"prompt_tokens\"") != std::string::npos);
    ASSERT_TRUE(responses.back().find("\"total_tokens\"") != std::string::npos);
    ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
}

TEST_F(LLMFlowHttpTest, streamChatCompletionsBadStopStringType) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, streamCompletionsBadStopStringElementType) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, streamCompletionsIncludeStopStrInOutputFalse) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, streamCompletionsBadIncludeStopStrInOutputType) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, streamChatCompletionsBadStreamOptionsBadType) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, streamCompletionsStreamOptionsBadType) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, streamChatCompletionsStreamOptionsBadContent) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, streamCompletionsStreamOptionsBadContent) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, streamChatCompletionsBadIncludeUsage) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(LLMFlowHttpTest, streamCompletionsBadIncludeUsage) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
}

// /v3/chat/completions endpoint
// unary, gready search
// Correct payload, however disconnection immediately
TEST_F(LLMFlowHttpTest, inferChatCompletionsUnaryClientDisconnectedImmediately) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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

    EXPECT_CALL(*writer, RegisterDisconnectionCallback(::testing::_)).WillOnce([](std::function<void()> fn) {
        fn();  // disconnect immediately, even before read_all is called
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

// /v3/chat/completions endpoint
// streaming
// Correct payload, however disconnection immediately
TEST_F(LLMFlowHttpTest, inferChatCompletionsStreamClientDisconnectedImmediately) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_EQ(i, 1);
    ASSERT_EQ(response, "");
}

// /v3/completions endpoint
// streaming
// Correct payload, however disconnection immediately
TEST_F(LLMFlowHttpTest, inferCompletionsStreamClientDisconnectedImmediately) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_EQ(i, 1);
    ASSERT_EQ(response, "");
}

const std::string validRequestBodyWithParameter(const std::string& parameter, const std::string& value) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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

class LLMHttpParametersValidationTest : public LLMFlowHttpTest {};

TEST_F(LLMHttpParametersValidationTest, maxTokensInvalid) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, maxTokensExceedsUint32Size) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, maxTokensExceeds4000WhenIgnoreEosTrue) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "ignore_eos": true,
            "max_tokens": 4001,
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
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, maxCompletionsTokensInvalid) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, maxCompletionsTokensExceedsUint32Size) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, maxCompletionsTokensExceeds4000WhenIgnoreEosTrue) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "ignore_eos": true,
            "max_completion_tokens": 4001,
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
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, streamInvalid) {
    std::string requestBody = validRequestBodyWithParameter("stream", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::JSON_INVALID);
}

TEST_F(LLMHttpParametersValidationTest, messagesInvalid) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1,
            "messages": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, messagesMissing) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, messageNotAnObject) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1,
            "messages": [
                "What is OpenVINO?"
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, messageNotAString) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1,
            "messages": [
            {
                "role": "user",
                "content": 1
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, roleNotAString) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, promptInvalid) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1,
            "prompt": 5
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, promptMissing) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "stream": false,
            "max_tokens": 1
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, modelMissing) {
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::JSON_INVALID);
}

TEST_F(LLMHttpParametersValidationTest, modelInvalid) {
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::JSON_INVALID);
}

TEST_F(LLMHttpParametersValidationTest, ignoreEosValid) {
    std::string requestBody = validRequestBodyWithParameter("ignore_eos", "false");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, ignoreEosInvalid) {
    std::string requestBody = validRequestBodyWithParameter("ignore_eos", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, repetitionPenaltyValid) {
    std::string requestBody = validRequestBodyWithParameter("repetition_penalty", "2.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter("repetition_penalty", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, repetitionPenaltyInvalid) {
    std::string requestBody = validRequestBodyWithParameter("repetition_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, diversityPenaltyValid) {
    std::string requestBody = validRequestBodyWithParameter("diversity_penalty", "2.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, diversityPenaltyInvalid) {
    std::string requestBody = validRequestBodyWithParameter("diversity_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, lengthPenaltyValid) {
    std::string requestBody = validRequestBodyWithParameter("length_penalty", "2.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter("length_penalty", "2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, lengthPenaltyInvalid) {
    std::string requestBody = validRequestBodyWithParameter("length_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, temperatureValid) {
    std::string requestBody = validRequestBodyWithParameter("temperature", "1.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter("temperature", "0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter("temperature", "2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, temperatureInvalid) {
    std::string requestBody = validRequestBodyWithParameter("temperature", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, temperatureOutOfRange) {
    std::string requestBody = validRequestBodyWithParameter("temperature", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, frequencyPenaltyValid) {
    std::string requestBody = validRequestBodyWithParameter("frequency_penalty", "1.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter("frequency_penalty", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, frequencyPenaltyInvalid) {
    std::string requestBody = validRequestBodyWithParameter("frequency_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, frequencyPenaltyOutOfRange) {
    std::string requestBody = validRequestBodyWithParameter("frequency_penalty", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, presencePenaltyValid) {
    std::string requestBody = validRequestBodyWithParameter("presence_penalty", "1.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter("presence_penalty", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, presencePenaltyInvalid) {
    std::string requestBody = validRequestBodyWithParameter("presence_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, presencePenaltyOutOfRange) {
    std::string requestBody = validRequestBodyWithParameter("presence_penalty", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, topPValid) {
    std::string requestBody = validRequestBodyWithParameter("top_p", "0.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter("top_p", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, topPInvalid) {
    std::string requestBody = validRequestBodyWithParameter("top_p", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, topPOutOfRange) {
    std::string requestBody = validRequestBodyWithParameter("top_p", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, topKValid) {
    std::string requestBody = validRequestBodyWithParameter("top_k", "2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, topKInvalid) {
    std::string requestBody = validRequestBodyWithParameter("top_k", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, seedValid) {
    std::string requestBody = validRequestBodyWithParameter("seed", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, seedInvalid) {
    std::string requestBody = validRequestBodyWithParameter("seed", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, bestOfValid) {
    std::string requestBody = validRequestBodyWithParameter("best_of", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, bestOfInvalid) {
    std::string requestBody = validRequestBodyWithParameter("best_of", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, bestOfNegative) {
    std::string requestBody = validRequestBodyWithParameter("best_of", "-1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, bestOfExceedsLimit) {
    std::string requestBody = validRequestBodyWithParameter("best_of", "40");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, nValid) {
    std::string requestBody = validRequestBodyWithParameter("n", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, nInvalid) {
    std::string requestBody = validRequestBodyWithParameter("n", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, nNegative) {
    std::string requestBody = validRequestBodyWithParameter("best_of", "-1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, nGreaterThanBestOf) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, MessagesEmpty) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "max_tokens": 1,
            "messages": []
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, MessagesWithEmptyObject) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "messages": [{}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, EmptyPrompt) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "prompt": ""
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, MessagesWithOnlyRole) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "messages": [{"role": "abc"}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, SpeculativeDecodingExclusiveParametersProvided) {
    std::string requestBody = R"(
        {
            "model": "llmDummySpeculativePipeline",
            "prompt": "hello",
            "num_assistant_tokens": 5,
            "assistant_confidence_threshold": 0.5
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, SpeculativeDecodingExclusiveParametersProvidedChat) {
    std::string requestBody = R"(
        {
            "model": "llmDummySpeculativePipeline",
            "messages": [{"content": "def"}],
            "num_assistant_tokens": 5,
            "assistant_confidence_threshold": 0.5
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, SpeculativeDecodingNoSDSpecificParametersProvided) {
    std::string requestBody = R"(
        {
            "model": "llmDummySpeculativePipeline",
            "prompt": "hello"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, SpeculativeDecodingNoSDSpecificParametersProvidedChat) {
    std::string requestBody = R"(
        {
            "model": "llmDummySpeculativePipeline",
            "messages": [{"content": "def"}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(LLMHttpParametersValidationTest, MessagesWithOnlyContent) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "max_tokens": 1,
            "messages": [{"content": "def"}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_F(LLMHttpParametersValidationTest, MessagesWithMoreMessageFields) {
    std::string requestBody = R"(
        {
            "model": "llmDummyKFS",
            "max_tokens": 1,
            "messages": [{"role": "123", "content": "def", "unexpected": "123"}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

class LLMConfigHttpTest : public ::testing::Test {
public:
    void SetUp() { py::initialize_interpreter(); }
    void TearDown() { py::finalize_interpreter(); }
};

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
    auto& m = mediapipeDummy.getLLMNodeResourcesMap();
    m.insert(std::pair<std::string, std::shared_ptr<LLMNodeResources>>("llmNode", nullptr));
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
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST);
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
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST);
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
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_DIRECTORY_DOES_NOT_EXIST);
}

TEST_F(LLMConfigHttpTest, LLMNodeResourceInitFailed) {
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
                models_path: "/"
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
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED);
    ASSERT_EQ(mediapipeDummy.getLLMNodeResources("llmNode"), nullptr);
}

struct MockedLLMNodeResources : public LLMNodeResources {
public:
    void initializeContinuousBatchingPipeline(
        const std::string& basePath,
        const ov::genai::SchedulerConfig& schedulerConfig,
        const std::string& device,
        const plugin_config_t& pluginConfig,
        const plugin_config_t& tokenizerPluginConfig) override {
        // Do not initialize, it is not needed in a test
    }

    void initiateGeneration() {
        // Do not initiate, the cb lib is not initialized anyway
    }
};

class LLMOptionsHttpTest : public ::testing::Test {
public:
    void SetUp() { py::initialize_interpreter(); }
    void TearDown() { py::finalize_interpreter(); }
};

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckDefault) {
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
    MockedLLMNodeResources nodeResources;
    ASSERT_EQ(LLMNodeResources::initializeLLMNodeResources(nodeResources, config.node(0), ""), StatusCode::OK);
    ASSERT_EQ(nodeResources.schedulerConfig.max_num_batched_tokens, 256);
    ASSERT_EQ(nodeResources.schedulerConfig.cache_size, 8);
    ASSERT_EQ(nodeResources.schedulerConfig.dynamic_split_fuse, true);
    ASSERT_EQ(nodeResources.schedulerConfig.max_num_seqs, 256);
    ASSERT_EQ(nodeResources.schedulerConfig.enable_prefix_caching, false);
    ASSERT_EQ(nodeResources.device, "CPU");
    ASSERT_EQ(nodeResources.pluginConfig.size(), 0);
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckHalfDefault) {
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
    MockedLLMNodeResources nodeResources;
    ASSERT_EQ(LLMNodeResources::initializeLLMNodeResources(nodeResources, config.node(0), ""), StatusCode::OK);

    ASSERT_EQ(nodeResources.schedulerConfig.max_num_batched_tokens, 98);
    ASSERT_EQ(nodeResources.schedulerConfig.cache_size, 1);
    ASSERT_EQ(nodeResources.schedulerConfig.dynamic_split_fuse, true);
    ASSERT_EQ(nodeResources.schedulerConfig.max_num_seqs, 256);
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsWrongPluginFormat) {
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
    MockedLLMNodeResources nodeResources;
    ASSERT_EQ(LLMNodeResources::initializeLLMNodeResources(nodeResources, config.node(0), ""), StatusCode::PLUGIN_CONFIG_WRONG_FORMAT);
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckPluginConfig) {
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
                plugin_config: '{"A": "B", "C": "D"}'
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
    std::shared_ptr<LLMNodeResources> nodeResources = std::make_shared<MockedLLMNodeResources>();
    ASSERT_EQ(LLMNodeResources::initializeLLMNodeResources(*nodeResources, config.node(0), ""), StatusCode::OK);

    ASSERT_EQ(nodeResources->pluginConfig.size(), 2);
    ASSERT_EQ(nodeResources->pluginConfig.count("A"), 1);
    ASSERT_EQ(nodeResources->pluginConfig.count("C"), 1);
    ASSERT_EQ(nodeResources->pluginConfig["A"], "B");
    ASSERT_EQ(nodeResources->pluginConfig["C"], "D");
}

TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckNonDefault) {
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
                max_num_batched_tokens: 1024
                cache_size: 1
                max_num_seqs: 95
                dynamic_split_fuse: false
                enable_prefix_caching: true
                max_tokens_limit: 700
                best_of_limit: 3
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
    MockedLLMNodeResources nodeResources;
    ASSERT_EQ(LLMNodeResources::initializeLLMNodeResources(nodeResources, config.node(0), ""), StatusCode::OK);

    ASSERT_EQ(nodeResources.schedulerConfig.max_num_batched_tokens, 1024);
    ASSERT_EQ(nodeResources.schedulerConfig.cache_size, 1);
    ASSERT_EQ(nodeResources.schedulerConfig.dynamic_split_fuse, false);
    ASSERT_EQ(nodeResources.schedulerConfig.max_num_seqs, 95);
    ASSERT_EQ(nodeResources.schedulerConfig.enable_prefix_caching, true);
    ASSERT_EQ(nodeResources.maxTokensLimit, 700);
    ASSERT_EQ(nodeResources.bestOfLimit, 3);
}

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
    MockedLLMNodeResources nodeResources;
    ASSERT_EQ(LLMNodeResources::initializeLLMNodeResources(nodeResources, config.node(0), ""), StatusCode::OK);
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

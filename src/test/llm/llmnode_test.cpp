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

#include "../../http_rest_api_handler.hpp"
#include "../../http_status_code.hpp"
#include "../../json_parser.hpp"
#include "../../llm/apis/openai_completions.hpp"
#include "../../llm/language_model/continuous_batching/llm_executor.hpp"
#include "../../llm/language_model/continuous_batching/servable.hpp"
#include "../../llm/servable.hpp"
#include "../../llm/servable_initializer.hpp"
#include "../../llm/text_processor.hpp"
#include "../../ov_utils.hpp"
#include "../../server.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "../test_http_utils.hpp"
#include "../test_utils.hpp"

using namespace ovms;

struct TestParameters {
    std::string modelName;
    bool generateExpectedOutput;
    bool checkLogprobs;
    bool checkFinishReason;
    bool testSpeculativeDecoding;
};

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

class LLMFlowHttpTestParameterized : public LLMFlowHttpTest, public ::testing::WithParamInterface<TestParameters> {};

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJson) {
    auto params = GetParam();
    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.num_beams = 16;
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
            "best_of": 16,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of such if...else...
    if (params.modelName.find("vlm") == std::string::npos) {
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
            handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
            ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    }
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
    config.num_beams = 16;
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
            ASSERT_TRUE(choice["text"].IsString());
            chunks.push_back(std::string(choice["text"].GetString()));
        }
        EXPECT_STREQ(d["model"].GetString(), params.modelName.c_str());
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
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
        EXPECT_CALL(*writer, PartialReply(::testing::_)).WillOnce([this, &params](std::string response) {
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
                if (params.checkFinishReason) {
                    ASSERT_TRUE(choice["finish_reason"].IsString());
                    EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
                }
                ASSERT_EQ(choice["index"], i++);
                if (params.checkLogprobs) {
                    ASSERT_FALSE(choice["logprobs"].IsObject());
                }
                ASSERT_TRUE(choice["text"].IsString());
                EXPECT_STREQ(choice["text"].GetString(), "What is OpenVINO?");
            }
            EXPECT_STREQ(d["model"].GetString(), params.modelName.c_str());
            EXPECT_STREQ(d["object"].GetString(), "text_completion.chunk");
        });
        ASSERT_EQ(
            handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
            ovms::StatusCode::PARTIAL_END);
    } else {
        // In legacy servable streaming with echo, prompt can be sent back in multiple chunks
        std::vector<std::string> responses;
        EXPECT_CALL(*writer, PartialReply(::testing::_))
            .WillRepeatedly([this, &responses](std::string response) {
                responses.push_back(response);
            });
        EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
        ASSERT_EQ(
            handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
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
            "stop": " ",
            "include_stop_str_in_output": true,
            "prompt": "                                   |                                |                             |  "
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse.HasMember("choices"));
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Size(), 1);
    ASSERT_TRUE(parsedResponse["choices"].GetArray()[0].HasMember("text"));
    ASSERT_TRUE(parsedResponse["choices"].GetArray()[0]["text"].IsString());
    ASSERT_EQ(parsedResponse["choices"].GetArray()[0]["text"].GetString(), std::string{""});
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

TEST_P(LLMFlowHttpTestParameterized, unaryCompletionsJsonN) {
    auto params = GetParam();
    // TODO: In the next step we should break this suite into smaller ones, use proper configuration instead of skipping
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.num_beams = 16;
    config.num_return_sequences = 8;
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
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 8 * 5 /* n * max_tokens */);
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

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsJsonN) {
    auto params = GetParam();
    config.max_new_tokens = 5;
    config.rng_seed = 1;
    config.num_beams = 16;
    config.num_return_sequences = 8;
    config.echo = false;
    if (params.generateExpectedOutput) {
        ASSERT_EQ(generateExpectedText("What is OpenVINO?", false), 0);
        ASSERT_EQ(config.num_return_sequences, expectedMessages.size());
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
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
    ASSERT_EQ(parsedResponse["usage"].GetObject()["completion_tokens"].GetInt(), 8 * 5 /* n * max_tokens */);
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
    std::vector<std::pair<std::string, std::string>> headers;
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", "/v2/models/" + params.modelName + "/versions/1/infer", headers), ovms::StatusCode::OK);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsJsonContentArray) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
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

    if (params.modelName.find("vlm") != std::string::npos) {
        ASSERT_EQ(
            handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
            ovms::StatusCode::OK);
    } else {
        ASSERT_EQ(
            handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
            ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    }
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsJsonNMultipleStopStrings) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsPromptTokensWithMaxTokensExceedsMaxModelLength) {
    auto params = GetParam();
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string prompt;
    // creating prompt that will be tokenized to 2048 tokens when model max length is 2048
    for (int i = 0; i < 2044; i++) {
        prompt += "hello ";
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "max_tokens" : 5,
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsPromptTokensWithMaxCompletionTokensExceedsMaxModelLength) {
    auto params = GetParam();
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string prompt;
    // creating prompt that will be tokenized to 2048 tokens when model max length is 2048
    for (int i = 0; i < 2044; i++) {
        prompt += "hello ";
    }
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "stream": false,
            "seed" : 1,
            "max_completion_tokens": 5,
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMFlowHttpTestParameterized, unaryChatCompletionsPromptTokensEqualToMaxModelLength) {
    auto params = GetParam();
    if (params.modelName.find("vlm") != std::string::npos) {
        GTEST_SKIP();
    }
    std::string prompt;
    // creating prompt that will be tokenized to 2048 tokens when model max length is 2048
    for (int i = 0; i < 2048; i++) {
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
    ON_CALL(*writer, PartialReply).WillByDefault([this, &params](std::string response) {
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
            ASSERT_TRUE(choice["text"].IsString());
        }
        EXPECT_STREQ(d["model"].GetString(), params.modelName.c_str());
        EXPECT_STREQ(d["object"].GetString(), "text_completion.chunk");
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
    ON_CALL(*writer, PartialReply).WillByDefault([this, &params](std::string response) {
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
            ASSERT_TRUE(choice["delta"].IsObject());
            ASSERT_TRUE(choice["delta"]["content"].IsString());
        }
        EXPECT_STREQ(d["model"].GetString(), params.modelName.c_str());
        EXPECT_STREQ(d["object"].GetString(), "chat.completion.chunk");
    });
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
    }
}

// Potential sporadic - move to functional if problematic
TEST_P(LLMFlowHttpTestParameterized, streamChatCompletionsSingleStopString) {
    GTEST_SKIP() << "Real sporadic, either fix or move to functional";
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
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
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"stop\"") != std::string::npos);
    }
    std::regex content_regex("\"content\":\".*\\.[ ]{0,1}\"");
    if (params.modelName.find("legacy") != std::string::npos) {
        // In legacy streaming we don't know if the callback is the last one, so we rely on entire generation call finish.
        // Because of that, we might get additional response with empty content at the end of the stream.
        ASSERT_TRUE(std::regex_search(responses[responses.size() - 2], content_regex) || std::regex_search(responses.back(), content_regex));
    } else {
        ASSERT_TRUE(std::regex_search(responses.back(), content_regex));
    }
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"stop\"") != std::string::npos);
    }
    std::regex content_regex("\"text\":\".*\\.[ ]{0,1}\"");
    if (params.modelName.find("legacy") != std::string::npos) {
        // In legacy streaming we don't know if the callback is the last one, so we rely on entire generation call finish.
        // Because of that, we might get additional response with empty content at the end of the stream.
        ASSERT_TRUE(std::regex_search(responses[responses.size() - 2], content_regex) || std::regex_search(responses.back(), content_regex));
    } else {
        ASSERT_TRUE(std::regex_search(responses.back(), content_regex));
    }
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_TRUE(responses.back().find("\"completion_tokens\":5") != std::string::npos);
    ASSERT_TRUE(responses.back().find("\"prompt_tokens\"") != std::string::npos);
    ASSERT_TRUE(responses.back().find("\"total_tokens\"") != std::string::npos);
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_TRUE(responses.back().find("\"completion_tokens\":5") != std::string::npos);
    ASSERT_TRUE(responses.back().find("\"prompt_tokens\"") != std::string::npos);
    ASSERT_TRUE(responses.back().find("\"total_tokens\"") != std::string::npos);
    if (params.checkFinishReason) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    ASSERT_EQ(i, 1);
    ASSERT_EQ(response, "");
}

INSTANTIATE_TEST_SUITE_P(
    LLMFlowHttpTestInstances,
    LLMFlowHttpTestParameterized,
    ::testing::Values(
        // params:     model name, generate expected output, check logprobs, check finish reason, test speculative decoding
        TestParameters{"lm_cb_regular", true, true, true, false},
        TestParameters{"lm_legacy_regular", false, false, false, false},
        TestParameters{"vlm_cb_regular", false, true, true, false},
        TestParameters{"vlm_legacy_regular", false, false, false, false}));

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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, streamInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "stream", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, messageNotAString) {
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
                "content": 1
            }
            ]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::JSON_INVALID);
}

TEST_P(LLMHttpParametersValidationTest, ignoreEosValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "ignore_eos", "false");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, ignoreEosInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "ignore_eos", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, repetitionPenaltyValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "repetition_penalty", "2.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "repetition_penalty", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, repetitionPenaltyInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "repetition_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, lengthPenaltyValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "length_penalty", "2.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "length_penalty", "2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, lengthPenaltyInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "length_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, temperatureValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "temperature", "1.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "temperature", "0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "temperature", "2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, temperatureInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "temperature", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, temperatureOutOfRange) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "temperature", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, frequencyPenaltyValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "frequency_penalty", "1.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "frequency_penalty", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, frequencyPenaltyInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "frequency_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, frequencyPenaltyOutOfRange) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "frequency_penalty", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, presencePenaltyValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "presence_penalty", "1.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "presence_penalty", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, presencePenaltyInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "presence_penalty", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, presencePenaltyOutOfRange) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "presence_penalty", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, topPValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_p", "0.5");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    requestBody = validRequestBodyWithParameter(params.modelName, "top_p", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, topPInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_p", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, topPOutOfRange) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_p", "3.0");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, topKValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_k", "2");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, topKInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "top_k", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, seedValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "seed", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, seedInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "seed", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, bestOfValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "best_of", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, bestOfInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "best_of", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, bestOfNegative) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "best_of", "-1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, bestOfExceedsLimit) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "best_of", "40");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, nValid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "n", "1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

TEST_P(LLMHttpParametersValidationTest, nInvalid) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "n", "\"INVALID\"");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, nNegative) {
    auto params = GetParam();
    std::string requestBody = validRequestBodyWithParameter(params.modelName, "n", "-1");

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, MessagesWithOnlyRole) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "messages": [{"role": "abc"}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
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
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
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
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_P(LLMHttpParametersValidationTest, MessagesWithMoreMessageFields) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "max_tokens": 1,
            "messages": [{"role": "123", "content": "def", "unexpected": "123"}]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
}

INSTANTIATE_TEST_SUITE_P(
    LLMHttpParametersValidationTestInstances,
    LLMHttpParametersValidationTest,
    ::testing::Values(
        // params:     model name, generate expected output, check logprobs, check finish reason, test speculative decoding
        TestParameters{"lm_cb_regular", true, true, true, false},
        TestParameters{"lm_legacy_regular", false, false, false, false},
        TestParameters{"vlm_cb_regular", false, true, true, false},
        TestParameters{"vlm_legacy_regular", false, false, false, false}));

// Common tests for all pipeline types (testing logic executed prior pipeline type selection)
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

class LLMConfigHttpTestParameterized : public ::testing::Test, public ::testing::WithParamInterface<std::tuple<std::string, ovms::StatusCode>> {
public:
    void SetUp() { py::initialize_interpreter(); }
    void TearDown() { py::finalize_interpreter(); }
};

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
    ASSERT_EQ(mediapipeDummy.validate(manager), expectedStatusCode);
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
        std::make_tuple("LM", ovms::StatusCode::LLM_NODE_RESOURCE_STATE_INITIALIZATION_FAILED),
        std::make_tuple("VLM_CB", ovms::StatusCode::INTERNAL_ERROR),
        std::make_tuple("VLM", ovms::StatusCode::INTERNAL_ERROR)));

// Those tests are working on Continuous Batching path, since most of the node options are scheduler parameters that are not used in non-CB servables
// We could consider adding tests for non-CB path in the future in the separate test suite
class LLMOptionsHttpTestPython : public ::testing::Test {
public:
    static void SetUpTestSuite() { py::initialize_interpreter(); }
    static void TearDownTestSuite() { py::finalize_interpreter(); }
};

class LLMOptionsHttpTest : public LLMOptionsHttpTestPython {
public:
    std::string modelsPath;
    void SetUp() { modelsPath = "/ovms/src/test/llm_testing/facebook/opt-125m"; }
};

class LLMVLMOptionsHttpTest : public LLMOptionsHttpTestPython {
public:
    std::string modelsPath;
    void SetUp() { modelsPath = "/ovms/src/test/llm_testing/OpenGVLab/InternVL2-1B"; }
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
    ASSERT_EQ(properties->schedulerConfig.cache_size, 8);
    ASSERT_EQ(properties->schedulerConfig.dynamic_split_fuse, true);
    ASSERT_EQ(properties->schedulerConfig.max_num_seqs, 256);
    ASSERT_EQ(properties->schedulerConfig.enable_prefix_caching, false);
    ASSERT_EQ(properties->device, "CPU");
    ASSERT_EQ(properties->pluginConfig.size(), 0);
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

    ASSERT_EQ(properties->pluginConfig.size(), 2);
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
}
TEST_F(LLMOptionsHttpTest, LLMNodeOptionsCheckNonDefault) {
    LLMNodeOptionsCheckNonDefault(modelsPath);
}
TEST_F(LLMVLMOptionsHttpTest, LLMVLMNodeOptionsCheckNonDefault) {
    LLMNodeOptionsCheckNonDefault(modelsPath);
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
    executionContext.inputIds = tensor;
    auto status = legacyServable.callValidateInputComplianceWithProperties(executionContext.inputIds);
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
    executionContext.inputIds = tensor;
    auto status = legacyServable.callValidateInputComplianceWithProperties(executionContext.inputIds);
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
    executionContext.inputIds = tensor;
    auto status = legacyServable.callValidateInputComplianceWithProperties(executionContext.inputIds);
    ASSERT_EQ(status, absl::OkStatus());
}

// TODO: Add missing tests for reading max prompt len property from configuration

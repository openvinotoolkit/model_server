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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/openvino.hpp>

#include "../../http_rest_api_handler.hpp"
#include "../../http_status_code.hpp"
#include "../../json_parser.hpp"
#include "../../llm/apis/openai_completions.hpp"
#include "../../llm/language_model/continuous_batching/llm_executor.hpp"
#include "../../llm/language_model/continuous_batching/servable.hpp"
#include "../../llm/servable.hpp"
#include "../../llm/servable_initializer.hpp"
#include "../../llm/text_utils.hpp"
#include "../../ov_utils.hpp"
#include "../../server.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "../constructor_enabled_model_manager.hpp"
#include "../platform_utils.hpp"
#include "../test_http_utils.hpp"
#include "../test_utils.hpp"

using namespace ovms;

struct TokenizeTestParameters {
    std::string modelName;
    uint64_t paddingTokenId;
    std::vector<uint64_t> expectedTokens;
};

class LLMHttpTokenizeTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpointTokenize = "/v3/tokenize";
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
            cbPipe = std::make_shared<ov::genai::ContinuousBatchingPipeline>(getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m"), schedulerConfig, device, pluginConfig, tokenizerPluginConfig);
            llmExecutorWrapper = std::make_shared<LLMExecutorWrapper>(cbPipe);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Error during llm node initialization for models_path exception: {}", e.what());
        } catch (...) {
            SPDLOG_ERROR("Error during llm node initialization for models_path");
        }
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        ON_CALL(*writer, PartialReplyBegin(::testing::_)).WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));  // make the streaming flow sequential
        ovms::Server& server = ovms::Server::instance();
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointTokenize, headers), ovms::StatusCode::OK);
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

std::shared_ptr<ov::genai::ContinuousBatchingPipeline> LLMHttpTokenizeTest::cbPipe;
std::shared_ptr<LLMExecutorWrapper> LLMHttpTokenizeTest::llmExecutorWrapper;
std::unique_ptr<std::thread> LLMHttpTokenizeTest::t;

class LLMTokenizeTests : public LLMHttpTokenizeTest, public ::testing::WithParamInterface<TokenizeTestParameters> {};

TEST_P(LLMTokenizeTests, tokenizeString) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "text": "hello world",
            "add_special_tokens": false
        }
    )";

    std::cout << "Request body: " << requestBody << std::endl;
    auto expectedTokens = params.expectedTokens;

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    parsedResponse.Parse(response.c_str());

    ASSERT_EQ(parsedResponse.HasParseError(), false);
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tokens"));
    const auto& tokens = parsedResponse["tokens"];
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_EQ(tokens.Size(), expectedTokens.size());
    for (rapidjson::SizeType i = 0; i < tokens.Size(); ++i) {
        const auto& token = tokens[i];
        ASSERT_TRUE(token.IsInt());
        ASSERT_EQ(token.GetInt(), expectedTokens[i]);
    }
}

TEST_P(LLMTokenizeTests, tokenizeArrayOfStrings) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "text": ["Hello, how are you?", "What is the capital of France?"]
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    parsedResponse.Parse(response.c_str());

    ASSERT_EQ(parsedResponse.HasParseError(), false);
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tokens"));
    const auto& tokens = parsedResponse["tokens"];
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_EQ(tokens.Size(), 2);
    for (rapidjson::SizeType i = 0; i < tokens.Size(); ++i) {
        const auto& tokenArray = tokens[i];
        ASSERT_TRUE(tokenArray.IsArray());
        ASSERT_GT(tokenArray.Size(), 0);
    }
}

TEST_P(LLMTokenizeTests, tokenizeEmptyString) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "text": "",
            "add_special_tokens": false
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    parsedResponse.Parse(response.c_str());

    ASSERT_EQ(parsedResponse.HasParseError(), false);
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tokens"));
    const auto& tokens = parsedResponse["tokens"];
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_EQ(tokens.Size(), 0);
}

TEST_P(LLMTokenizeTests, tokenizeArrayWithEmptyString) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "text": ["hello world", ""],
            "add_special_tokens": false
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());

    ASSERT_EQ(parsedResponse.HasParseError(), false);
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tokens"));
    const auto& tokens = parsedResponse["tokens"];
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_EQ(tokens.Size(), 2);
    const auto& firstTokenArray = tokens[0];
    ASSERT_TRUE(firstTokenArray.IsArray());
    ASSERT_GT(firstTokenArray.Size(), 0);
    const auto& secondTokenArray = tokens[1];
    ASSERT_TRUE(secondTokenArray.IsArray());
    ASSERT_EQ(secondTokenArray.Size(), 0);
}

TEST_P(LLMTokenizeTests, tokenizeStringWithMaxLength) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "text": "Hello, how are you today?",
            "add_special_tokens": false,
            "max_length": 5
        }
    )";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());

    ASSERT_EQ(parsedResponse.HasParseError(), false);
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tokens"));
    const auto& tokens = parsedResponse["tokens"];
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_EQ(tokens.Size(), 5);
}

TEST_P(LLMTokenizeTests, tokenizeArrayOfStringsWithMaxLength) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "text": ["Hello, how are you?", "What is the capital of France?"],
            "max_length": 5
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());

    ASSERT_EQ(parsedResponse.HasParseError(), false);
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tokens"));
    const auto& tokens = parsedResponse["tokens"];
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_EQ(tokens.Size(), 2);
    for (rapidjson::SizeType i = 0; i < tokens.Size(); ++i) {
        const auto& tokenArray = tokens[i];
        ASSERT_TRUE(tokenArray.IsArray());
        ASSERT_EQ(tokenArray.Size(), 5);
    }
}

TEST_P(LLMTokenizeTests, tokenizeStringWithPadToMaxLength) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "text": "hello world",
            "add_special_tokens": false,
            "max_length": 25,
            "pad_to_max_length": true
        }
    )";
    auto expectedTokens = params.expectedTokens;
    expectedTokens.insert(expectedTokens.end(), 25 - expectedTokens.size(), params.paddingTokenId);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());

    ASSERT_EQ(parsedResponse.HasParseError(), false);
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tokens"));
    const auto& tokens = parsedResponse["tokens"];
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_EQ(tokens.Size(), 25);
}

TEST_P(LLMTokenizeTests, tokenizeArrayOfStringsWithPadToMaxLength) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "text": ["Hello, how are you?", "What is the capital of France?"],
            "max_length": 25,
            "pad_to_max_length": true
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());

    ASSERT_EQ(parsedResponse.HasParseError(), false);
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tokens"));
    const auto& tokens = parsedResponse["tokens"];
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_EQ(tokens.Size(), 2);
    for (rapidjson::SizeType i = 0; i < tokens.Size(); ++i) {
        const auto& tokenArray = tokens[i];
        ASSERT_TRUE(tokenArray.IsArray());
        ASSERT_EQ(tokenArray.Size(), 25);
    }
}

TEST_P(LLMTokenizeTests, tokenizeStringWithPaddingSideLeft) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "text": "hello world",
            "max_length": 25,
            "pad_to_max_length": true,
            "padding_side": "left",
            "add_special_tokens": false
        }
    )";
    auto expectedTokens = params.expectedTokens;
    expectedTokens.insert(expectedTokens.begin(), 25 - expectedTokens.size(), params.paddingTokenId);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());

    ASSERT_EQ(parsedResponse.HasParseError(), false);
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tokens"));
    const auto& tokens = parsedResponse["tokens"];
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_EQ(tokens.Size(), 25);
    for (rapidjson::SizeType i = 0; i < tokens.Size(); ++i) {
        const auto& token = tokens[i];
        ASSERT_TRUE(token.IsInt());
        ASSERT_EQ(token.GetInt(), expectedTokens[i]);
    }
}

TEST_P(LLMTokenizeTests, tokenizeArrayOfStringsWithPaddingSideLeft) {
    auto params = GetParam();
    std::string requestBody = R"(
        {
            "model": ")" + params.modelName +
                              R"(",
            "text": ["Hello, how are you?", "What is the capital of France?"],
            "max_length": 25,
            "pad_to_max_length": true,
            "padding_side": "left"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());

    ASSERT_EQ(parsedResponse.HasParseError(), false);
    ASSERT_TRUE(parsedResponse.IsObject());
    ASSERT_TRUE(parsedResponse.HasMember("tokens"));
    const auto& tokens = parsedResponse["tokens"];
    ASSERT_TRUE(tokens.IsArray());
    ASSERT_EQ(tokens.Size(), 2);
    for (rapidjson::SizeType i = 0; i < tokens.Size(); ++i) {
        const auto& tokenArray = tokens[i];
        ASSERT_TRUE(tokenArray.IsArray());
        ASSERT_EQ(tokenArray[0].GetInt(), params.paddingTokenId);
        ASSERT_EQ(tokenArray.Size(), 25);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LLMTokenizeTestInstances,
    LLMTokenizeTests,
    ::testing::Values(
        // params:     model name, padding token id
        TokenizeTestParameters{"lm_cb_regular", 1, {42891, 232}},
        TokenizeTestParameters{"lm_legacy_regular", 1, {42891, 232}},
        TokenizeTestParameters{"vlm_cb_regular", 151643, {14990, 1879}},
        TokenizeTestParameters{"vlm_legacy_regular", 151643, {14990, 1879}}));

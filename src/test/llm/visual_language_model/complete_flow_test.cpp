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

#include "../../../http_rest_api_handler.hpp"
#include "../../../http_status_code.hpp"
#include "../../../json_parser.hpp"
#include "../../../llm/apis/openai_completions.hpp"
#include "../../../ov_utils.hpp"
#include "../../../server.hpp"
#include "../../test_http_utils.hpp"
#include "../../test_utils.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

using namespace ovms;

class VLMServableExecutionTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;
    const std::string endpointChatCompletions = "/v3/chat/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::string response;
    rapidjson::Document parsedResponse;
    ovms::HttpResponseComponents responseComponents;
    static void SetUpTestSuite() {
        std::string port = "9173";
        ovms::Server& server = ovms::Server::instance();
        ::SetUpServer(t, server, port, getGenericFullPathForSrcTest("/ovms/src/test/llm/visual_language_model/config.json").c_str());
        auto start = std::chrono::high_resolution_clock::now();
        const int numberOfRetries = 20;
        while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < numberOfRetries)) {
        }

        ASSERT_EQ(server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME), ovms::ModuleState::INITIALIZED) << "Loading manager takes too long. Server cannot start in 20 seconds.";
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

std::unique_ptr<std::thread> VLMServableExecutionTest::t;

std::string createRequestBody(const std::string& modelName, const std::vector<std::pair<std::string, std::string>>& fields, bool includeText = true, bool includeImage = true) {
    std::ostringstream oss;
    oss << R"(
        {
            "model": ")"
        << modelName << R"(",
            "messages": [
            {
                "role": "user",
                "content": [)";
    if (includeText) {
        oss << R"(
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    })";
        if (includeImage) {
            oss << ",";
        }
    }
    if (includeImage) {
        oss << R"(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAEElEQVR4nGIy+/oREAAA//8DiQIftNKCRwAAAABJRU5ErkJggg=="
                        }
                    })";
    }
    oss << R"(
                ]
            }
            ]
        )";
    for (const auto& field : fields) {
        oss << R"(, ")" << field.first << R"(": )" << field.second << R"()"
            << "\n";
    }
    oss << "\n}";
    return oss.str();
}

class VLMServableExecutionTestParameterized : public VLMServableExecutionTest, public ::testing::WithParamInterface<std::string> {};

// Unary flow

TEST_P(VLMServableExecutionTestParameterized, unaryBasic) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "false"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (modelName.find("legacy") == std::string::npos) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
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
    EXPECT_STREQ(parsedResponse["model"].GetString(), modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

// Only image input is accepted, but expected output can't be predicted
TEST_P(VLMServableExecutionTestParameterized, unaryBasicOnlyImage) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "false"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields, false, true);

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);
    parsedResponse.Parse(response.c_str());
    ASSERT_TRUE(parsedResponse["choices"].IsArray());
    ASSERT_EQ(parsedResponse["choices"].Capacity(), 1);
    int i = 0;
    for (auto& choice : parsedResponse["choices"].GetArray()) {
        if (modelName.find("legacy") == std::string::npos) {
            ASSERT_TRUE(choice["finish_reason"].IsString());
            EXPECT_STREQ(choice["finish_reason"].GetString(), "length");
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
    EXPECT_STREQ(parsedResponse["model"].GetString(), modelName.c_str());
    EXPECT_STREQ(parsedResponse["object"].GetString(), "chat.completion");
}

// Stream flow

TEST_P(VLMServableExecutionTestParameterized, streamBasic) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "true"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields);

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    if (modelName.find("legacy") == std::string::npos) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
    }
}

// Only image input is accepted, but expected output can't be predicted
TEST_P(VLMServableExecutionTestParameterized, streamBasicOnlyImage) {
    auto modelName = GetParam();
    std::vector<std::pair<std::string, std::string>> fields = {
        {"temperature", "0.0"},
        {"stream", "true"},
        {"max_tokens", "5"},
        {"ignore_eos", "true"}};
    std::string requestBody = createRequestBody(modelName, fields, false, true);

    std::vector<std::string> responses;

    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([this, &responses](std::string response) {
            responses.push_back(response);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointChatCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);
    if (modelName.find("legacy") == std::string::npos) {
        ASSERT_TRUE(responses.back().find("\"finish_reason\":\"length\"") != std::string::npos);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VLMServableExecutionTests,
    VLMServableExecutionTestParameterized,
    ::testing::Values("vlm_cb_regular", "vlm_legacy_regular"));

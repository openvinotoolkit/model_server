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
#include <chrono>
#include <map>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../http_rest_api_handler.hpp"
#include "../module_names.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"

#include "test_http_utils.hpp"
#include "test_utils.hpp"

class HttpOpenAIHandlerTest : public ::testing::Test {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unique_ptr<std::thread> t;
    std::string port = "9173";

    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;
    const std::string endpoint = "/v3/chat/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    void SetUpServer(const char* configPath) {
        ::SetUpServer(this->t, this->server, this->port, configPath);
        auto start = std::chrono::high_resolution_clock::now();
        while ((server.getModuleState(ovms::SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
        }

        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/config_mediapipe_openai_chat_completions_mock.json").c_str());
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);
    }

    void TearDown() {
        handler.reset();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

TEST_F(HttpOpenAIHandlerTest, Unary) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor("/v3/v1/completions/", requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    std::string expectedResponse = R"(/v3/v1/completions/

        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    {"model":"gpt","stream":false,"messages":[]}0)";
    ASSERT_EQ(response, expectedResponse);
}

TEST_F(HttpOpenAIHandlerTest, UnaryWithHeaders) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    )";
    comp.headers.push_back(std::pair<std::string, std::string>("test1", "header"));
    comp.headers.push_back(std::pair<std::string, std::string>("test2", "header"));

    ASSERT_EQ(
        handler->dispatchToProcessor("/v3/completions/", requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::OK);

    std::string expectedResponse = R"(/v3/completions/
test1headertest2header
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    {"model":"gpt","stream":false,"messages":[]}0)";
    ASSERT_EQ(response, expectedResponse);
}

TEST_F(HttpOpenAIHandlerTest, Stream) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": true,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyBegin(::testing::_)).WillOnce(testing::Invoke([](std::function<void()> fn) { fn(); }));
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(9);
    EXPECT_CALL(*writer, IsDisconnected()).Times(9);

    ASSERT_EQ(
        handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_EQ(response, "");
}

TEST_F(HttpOpenAIHandlerTest, BodyNotAJson) {
    std::string requestBody = "not a json";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - Cannot parse JSON body");
}

TEST_F(HttpOpenAIHandlerTest, JsonBodyValidButNotAnObject) {
    std::string requestBody = "[1, 2, 3]";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - JSON body must be an object");
}

TEST_F(HttpOpenAIHandlerTest, ModelFieldMissing) {
    std::string requestBody = R"(
        {
            "stream": true,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - model field is missing in JSON body");
}

TEST_F(HttpOpenAIHandlerTest, ModelFieldNotAString) {
    std::string requestBody = R"(
        {
            "model": 2,
            "stream": true,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - model field is not a string");
}

TEST_F(HttpOpenAIHandlerTest, StreamFieldNotABoolean) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": 2,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyBegin(::testing::_)).Times(0);
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - stream field is not a boolean");
}

TEST_F(HttpOpenAIHandlerTest, GraphWithANameDoesNotExist) {
    std::string requestBody = R"(
        {
            "model": "not_exist",
            "stream": false,
            "messages": []
        }
    )";

    EXPECT_CALL(*writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(*writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(*writer, IsDisconnected()).Times(0);

    auto status = handler->dispatchToProcessor("/v3/completions", requestBody, &response, comp, responseComponents, writer);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING);
}

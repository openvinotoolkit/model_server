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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#pragma GCC diagnostic pop

#include "test_utils.hpp"

class MockedServerRequestInterface final : public tensorflow::serving::net_http::ServerRequestInterface {
public:
    MOCK_METHOD(absl::string_view, uri_path, (), (const, override));
    MOCK_METHOD(absl::string_view, http_method, (), (const, override));
    MOCK_METHOD(void, WriteResponseBytes, (const char*, int64_t), (override));
    MOCK_METHOD(void, WriteResponseString, (absl::string_view), (override));
    MOCK_METHOD((std::unique_ptr<char[], tensorflow::serving::net_http::ServerRequestInterface::BlockDeleter>), ReadRequestBytes, (int64_t*), (override));
    MOCK_METHOD(absl::string_view, GetRequestHeader, (absl::string_view), (const, override));
    MOCK_METHOD((std::vector<absl::string_view>), request_headers, (), (const, override));
    MOCK_METHOD(void, OverwriteResponseHeader, (absl::string_view, absl::string_view), (override));
    MOCK_METHOD(void, AppendResponseHeader, (absl::string_view, absl::string_view), (override));
    MOCK_METHOD(void, PartialReplyWithStatus, (tensorflow::serving::net_http::HTTPStatusCode), (override));
    MOCK_METHOD(void, PartialReply, (std::string), (override));
    MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::CallbackStatus, PartialReplyWithFlushCallback, ((std::function<void()>)), (override));
    MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::BodyStatus, response_body_status, (), (override));
    MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::BodyStatus, request_body_status, (), (override));
    MOCK_METHOD(void, ReplyWithStatus, (tensorflow::serving::net_http::HTTPStatusCode), (override));
    MOCK_METHOD(void, Reply, (), (override));
    MOCK_METHOD(void, Abort, (), (override));
    MOCK_METHOD(void, PartialReplyEnd, (), (override));
};

class HttpOpenAIHandlerTest : public ::testing::Test {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unique_ptr<std::thread> t;
    std::string port = "9173";

    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;
    const std::string endpoint = "/v3/chat/completions";
    MockedServerRequestInterface writer;
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
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_openai_chat_completions_mock.json");
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
        handler->dispatchToProcessor("/v3/test/", requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);

    std::string expectedResponse = R"(/v3/test/

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
        handler->dispatchToProcessor("/v3/test/", requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::OK);

    std::string expectedResponse = R"(/v3/test/
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

    EXPECT_CALL(writer, PartialReplyEnd()).Times(1);
    EXPECT_CALL(writer, PartialReply(::testing::_)).Times(9);
    EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);

    ASSERT_EQ(
        handler->dispatchToProcessor("/v3/test", requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_EQ(response, "");
}

TEST_F(HttpOpenAIHandlerTest, BodyNotAJson) {
    std::string requestBody = "not a json";

    EXPECT_CALL(writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);

    auto status = handler->dispatchToProcessor("/v3/test", requestBody, &response, comp, responseComponents, &writer);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - Cannot parse JSON body");
}

TEST_F(HttpOpenAIHandlerTest, JsonBodyValidButNotAnObject) {
    std::string requestBody = "[1, 2, 3]";

    EXPECT_CALL(writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);

    auto status = handler->dispatchToProcessor("/v3/test", requestBody, &response, comp, responseComponents, &writer);
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

    EXPECT_CALL(writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);

    auto status = handler->dispatchToProcessor("/v3/test", requestBody, &response, comp, responseComponents, &writer);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - \"model\" field is missing in JSON body");
}

TEST_F(HttpOpenAIHandlerTest, ModelFieldNotAString) {
    std::string requestBody = R"(
        {
            "model": 2,
            "stream": true,
            "messages": []
        }
    )";

    EXPECT_CALL(writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);

    auto status = handler->dispatchToProcessor("/v3/test", requestBody, &response, comp, responseComponents, &writer);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - \"model\" field is not a string");
}

TEST_F(HttpOpenAIHandlerTest, StreamFieldNotABoolean) {
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": 2,
            "messages": []
        }
    )";

    EXPECT_CALL(writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);

    auto status = handler->dispatchToProcessor("/v3/test", requestBody, &response, comp, responseComponents, &writer);
    ASSERT_EQ(status, ovms::StatusCode::JSON_INVALID);
    ASSERT_EQ(status.string(), "The file is not valid json - \"stream\" field is not a boolean");
}

TEST_F(HttpOpenAIHandlerTest, GraphWithANameDoesNotExist) {
    std::string requestBody = R"(
        {
            "model": "not_exist",
            "stream": false,
            "messages": []
        }
    )";

    EXPECT_CALL(writer, PartialReplyEnd()).Times(0);
    EXPECT_CALL(writer, PartialReply(::testing::_)).Times(0);
    EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(0);

    auto status = handler->dispatchToProcessor("/v3/test", requestBody, &response, comp, responseComponents, &writer);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING);
}

// TODO (negative paths):
// - test that /v3/chat/completions endpoint is not reachable for builds without MediaPipe
// - test negative path for accessing /v3/chat/completions graph via KFS API
// - test negative path for accessing regular graph via /v3/chat/completions endpoint

// TODO (positive paths):
// - partial error is sent via "req" object

// TODO(mkulakow)
// Test actual flow once the type is changed from std::string to HttpPayload

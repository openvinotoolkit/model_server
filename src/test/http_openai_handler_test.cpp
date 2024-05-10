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

class HttpOpenAIHandlerTest : public ::testing::Test {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unique_ptr<std::thread> t;
    std::string port = "9173";

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
    }

    void TearDown() {
        handler.reset();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

TEST_F(HttpOpenAIHandlerTest, Unary) {
    ASSERT_EQ(1, 1);

    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;

    const std::string endpoint = "/v3/chat/completions";
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);

    std::string response;
    ovms::HttpResponseComponents responseComponents;
    tensorflow::serving::net_http::ServerRequestInterface* writer{nullptr};  // unused here, used only in streaming
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": false,
            "messages": []
        }
    )";
    ASSERT_EQ(handler->dispatchToProcessor(requestBody, &response, comp, responseComponents, writer), ovms::StatusCode::OK);

    // The calculator produces X packets, each appending timestamp starting from 0.
    // This test has stream=false, meaning only first packet will be serialized.
    std::string expectedResponse = requestBody + std::string{"0"};
    ASSERT_EQ(response, expectedResponse);
}

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
    MOCK_METHOD(void, PartialReply, (), (override));
    MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::CallbackStatus, PartialReplyWithFlushCallback, ((std::function<void()>)), (override));
    MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::BodyStatus, response_body_status, (), (override));
    MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::BodyStatus, request_body_status, (), (override));
    MOCK_METHOD(void, ReplyWithStatus, (tensorflow::serving::net_http::HTTPStatusCode), (override));
    MOCK_METHOD(void, Reply, (), (override));
    MOCK_METHOD(void, Abort, (), (override));
    MOCK_METHOD(void, PartialReplyEnd, (), (override));
};

TEST_F(HttpOpenAIHandlerTest, Stream) {
    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;

    const std::string endpoint = "/v3/chat/completions";
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);

    std::string response;
    ovms::HttpResponseComponents responseComponents;
    MockedServerRequestInterface writer;
    std::string requestBody = R"(
        {
            "model": "gpt",
            "stream": true,
            "messages": []
        }
    )";

    EXPECT_CALL(writer, PartialReplyEnd()).Times(1);                  // libevent call to end the response
    EXPECT_CALL(writer, PartialReply()).Times(9);                     // libevent calls to send the buffered data (calculator has 9 loop iterations)
    EXPECT_CALL(writer, WriteResponseString(::testing::_)).Times(9);  // writing chunk of response to buffer
    // TODO: Assert actual responses once we have HttpPayload defined

    ASSERT_EQ(
        handler->dispatchToProcessor(requestBody, &response, comp, responseComponents, &writer),
        ovms::StatusCode::PARTIAL_END);  // indicating that processor should not write response back
    // writer was responsible for sending the response

    // The calculator produces X packets, but the responses are returned via writer
    std::string expectedResponse{""};
    ASSERT_EQ(response, expectedResponse);
}

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../http_rest_api_handler.hpp"
#include "../http_status_code.hpp"
#include "../ov_utils.hpp"
#include "../server.hpp"
#include "test_http_utils.hpp"
#include "test_utils.hpp"

using namespace ovms;

class MediapipeDisabledTest : public ::testing::Test {
protected:
    static std::unique_ptr<std::thread> t;

public:
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::vector<std::pair<std::string, std::string>> headers;
    ovms::HttpRequestComponents comp;
    const std::string endpointChatCompletions = "/v3/chat/completions";
    const std::string endpointCompletions = "/v3/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    ovms::HttpResponseComponents responseComponents;
    std::string response;
    std::vector<std::string> expectedMessages;

    static void SetUpTestSuite() {
        std::string port = "9173";
        randomizePort(port);
        ovms::Server& server = ovms::Server::instance();
        ::EnsureSetUpServer(t, server, port, getGenericFullPathForSrcTest("/ovms/src/test/configs/config_cpu_dummy.json").c_str(), 15);
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
std::unique_ptr<std::thread> MediapipeDisabledTest::t;

TEST_F(MediapipeDisabledTest, completionsRequest) {
    std::string requestBody = R"(
        {
            "model": "dummy",
            "stream": false,
            "seed" : 1,
            "best_of": 16,
            "max_tokens": 5,
            "prompt": "What is OpenVINO?"
        }
    )";

    ASSERT_EQ(
        handler->dispatchToProcessor(endpointCompletions, requestBody, &response, comp, responseComponents, writer),
        ovms::StatusCode::NOT_IMPLEMENTED);
}

TEST_F(MediapipeDisabledTest, chatCompletionsRequest) {
    std::string requestBody = R"(
        {
            "model": "dummy",
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
        ovms::StatusCode::NOT_IMPLEMENTED);
}

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
#include "../http_payload.hpp"
#include "../module_names.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "test_http_utils.hpp"
#include "test_utils.hpp"
#include "platform_utils.hpp"

class MultiPartCalculatorTest : public ::testing::Test {
protected:
    ovms::Server& server = ovms::Server::instance();
    std::unique_ptr<ovms::HttpRestApiHandler> handler;

    std::unique_ptr<std::thread> t;
    std::string port = "9173";

    std::unordered_map<std::string, std::string> headers{{"content-type", "application/json"}};
    ovms::HttpRequestComponents comp;
    const std::string endpoint = "/v3/chat/completions";
    std::shared_ptr<MockedServerRequestInterface> writer;
    std::shared_ptr<MockedMultiPartParser> multiPartParser;
    std::string response;
    ovms::HttpResponseComponents responseComponents;

    void SetUpServer(const char* configPath) {
        ::SetUpServer(this->t, this->server, this->port, configPath);
        EnsureServerStartedWithTimeout(this->server, 5);
        handler = std::make_unique<ovms::HttpRestApiHandler>(server, 5);
    }

    void SetUp() {
        writer = std::make_shared<MockedServerRequestInterface>();
        multiPartParser = std::make_shared<MockedMultiPartParser>();
        SetUpServer(getGenericFullPathForSrcTest("/ovms/src/test/mediapipe/config_mediapipe_multipart_mock.json").c_str());
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);
    }

    void TearDown() {
        handler.reset();
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

TEST_F(MultiPartCalculatorTest, UnaryWithModelField) {  // only unary, there is no way to stream
    headers["content-type"] = "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW";

    comp = ovms::HttpRequestComponents();
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);

    std::string requestBody = R"(
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="username"

john_doe
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="email"

john@example.com
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="model"

multipart
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="doc"; filename="notes.txt"
Content-Type: text/plain

this is file content
It has two lines.
------WebKitFormBoundary7MA4YWxkTrZu0gW--)";

    EXPECT_CALL(*multiPartParser, parse()).WillOnce(::testing::Return(true));
    EXPECT_CALL(*multiPartParser, getFieldByName(::testing::Eq("model"))).WillOnce(::testing::Return("multipart"));
    EXPECT_CALL(*multiPartParser, getFieldByName(::testing::Eq("email"))).WillOnce(::testing::Return("john@example.com"));
    EXPECT_CALL(*multiPartParser, getFieldByName(::testing::Eq("username"))).WillOnce(::testing::Return("john_doe"));
    EXPECT_CALL(*multiPartParser, getFileContentByFieldName(::testing::Eq("file"))).WillOnce([](const std::string& name) {
        static std::string retval{"this is file content\nIt has two lines."};
        return std::string_view(retval);
    });

    const std::string URI = "/v3/something";
    ASSERT_EQ(
        handler->dispatchToProcessor(URI, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    std::string expectedResponse = R"(john@example.com+john_doe
this is file content
It has two lines.)";
    ASSERT_EQ(response, expectedResponse);
}

TEST_F(MultiPartCalculatorTest, UnaryWithMissingModelFieldDefaultRouting) {
    headers["content-type"] = "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW";

    comp = ovms::HttpRequestComponents();
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);

    std::string requestBody = R"(
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="username"

john_doe
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="email"

john@example.com
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="doc"; filename="notes.txt"
Content-Type: text/plain

this is file content
It has two lines.
------WebKitFormBoundary7MA4YWxkTrZu0gW--)";

    EXPECT_CALL(*multiPartParser, parse()).WillOnce(::testing::Return(true));
    EXPECT_CALL(*multiPartParser, getFieldByName(::testing::Eq("model"))).WillOnce(::testing::Return(""));
    EXPECT_CALL(*multiPartParser, getFieldByName(::testing::Eq("email"))).WillOnce(::testing::Return("john@example.com"));
    EXPECT_CALL(*multiPartParser, getFieldByName(::testing::Eq("username"))).WillOnce(::testing::Return("john_doe"));
    EXPECT_CALL(*multiPartParser, getFileContentByFieldName(::testing::Eq("file"))).WillOnce([](const std::string& name) {
        static std::string retval{"this is file content\nIt has two lines."};
        return std::string_view(retval);
    });

    // Default routing uses everything that comes after /v3/ as graph name
    const std::string URI = "/v3/multipart";

    ASSERT_EQ(
        handler->dispatchToProcessor(URI, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);

    std::string expectedResponse = R"(john@example.com+john_doe
this is file content
It has two lines.)";
    ASSERT_EQ(response, expectedResponse);
}

TEST_F(MultiPartCalculatorTest, UnaryWithMissingModelFieldDefaultRoutingWrongGraphName) {
    headers["content-type"] = "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW";

    comp = ovms::HttpRequestComponents();
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);

    std::string requestBody = R"(
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="username"

john_doe
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="email"

john@example.com
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="doc"; filename="notes.txt"
Content-Type: text/plain

this is file content
It has two lines.
------WebKitFormBoundary7MA4YWxkTrZu0gW--)";

    EXPECT_CALL(*multiPartParser, parse()).WillOnce(::testing::Return(true));
    EXPECT_CALL(*multiPartParser, getFieldByName(::testing::_)).Times(0);
    EXPECT_CALL(*multiPartParser, getFieldByName(::testing::Eq("model"))).WillOnce(::testing::Return(""));
    EXPECT_CALL(*multiPartParser, getFileContentByFieldName(::testing::_)).Times(0);

    // Default routing uses everything that comes after /v3/ as graph name
    const std::string URI = "/v3/NON_EXISTENT";

    ASSERT_EQ(
        handler->dispatchToProcessor(URI, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING);
}

TEST_F(MultiPartCalculatorTest, UnaryWithMissingModelFieldDefaultRoutingMissingGraphNameInURI) {
    headers["content-type"] = "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW";

    comp = ovms::HttpRequestComponents();
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, headers), ovms::StatusCode::OK);

    std::string requestBody = R"(
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="username"

john_doe
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="email"

john@example.com
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="doc"; filename="notes.txt"
Content-Type: text/plain

this is file content
It has two lines.
------WebKitFormBoundary7MA4YWxkTrZu0gW--)";

    EXPECT_CALL(*multiPartParser, parse()).WillOnce(::testing::Return(true));
    EXPECT_CALL(*multiPartParser, getFieldByName(::testing::_)).Times(0);
    EXPECT_CALL(*multiPartParser, getFieldByName(::testing::Eq("model"))).WillOnce(::testing::Return(""));
    EXPECT_CALL(*multiPartParser, getFileContentByFieldName(::testing::_)).Times(0);

    // Default routing uses everything that comes after /v3/ as graph name
    const std::string URI = "/v3/";

    ASSERT_EQ(
        handler->dispatchToProcessor(URI, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::FAILED_TO_DEDUCE_MODEL_NAME_FROM_URI);
}

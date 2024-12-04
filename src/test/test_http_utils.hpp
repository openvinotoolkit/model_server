//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#pragma once
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../http_server.hpp"
#include "../http_status_code.hpp"
#include "../http_async_writer_interface.hpp"

//class MockedServerRequestInterface final : public tensorflow::serving::net_http::ServerRequestInterface {
class MockedServerRequestInterface final : public ovms::HttpAsyncWriter {
public:
    //MOCK_METHOD(absl::string_view, uri_path, (), (const, override));
    //MOCK_METHOD(absl::string_view, http_method, (), (const, override));
    //MOCK_METHOD(void, WriteResponseBytes, (const char*, int64_t), (override));
    //MOCK_METHOD(void, WriteResponseString, (absl::string_view), (override));
    //MOCK_METHOD((std::unique_ptr<char[], tensorflow::serving::net_http::ServerRequestInterface::BlockDeleter>), ReadRequestBytes, (int64_t*), (override));
    //MOCK_METHOD(absl::string_view, GetRequestHeader, (absl::string_view), (const, override));
    //MOCK_METHOD((std::vector<absl::string_view>), request_headers, (), (const, override));
    //MOCK_METHOD(void, OverwriteResponseHeader, (absl::string_view, absl::string_view), (override));
    MOCK_METHOD(void, OverwriteResponseHeader, (const std::string&, const std::string&), (override));
    //MOCK_METHOD(void, AppendResponseHeader, (absl::string_view, absl::string_view), (override));
    //MOCK_METHOD(void, PartialReplyWithStatus, (std::string, tensorflow::serving::net_http::HTTPStatusCode), (override));
    MOCK_METHOD(void, PartialReplyWithStatus, (std::string, ovms::HTTPStatusCode), (override));
    MOCK_METHOD(void, PartialReply, (std::string), (override));
    //MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::CallbackStatus, PartialReplyWithFlushCallback, ((std::function<void()>)), (override));
    //MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::BodyStatus, response_body_status, (), (override));
    //MOCK_METHOD(tensorflow::serving::net_http::ServerRequestInterface::BodyStatus, request_body_status, (), (override));
    //MOCK_METHOD(void, ReplyWithStatus, (tensorflow::serving::net_http::HTTPStatusCode), (override));
    //MOCK_METHOD(void, Reply, (), (override));
    //MOCK_METHOD(void, Abort, (), (override));
    MOCK_METHOD(void, PartialReplyEnd, (), (override));
    MOCK_METHOD(bool, IsDisconnected, (), (const override));
    MOCK_METHOD(void, RegisterDisconnectionCallback, (std::function<void()>), (override));

    MOCK_METHOD(void, PartialReplyBegin, (std::function<void()>), (override));
};

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
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../http_async_writer_interface.hpp"
#include "../http_server.hpp"
#include "../http_status_code.hpp"
#include "../multi_part_parser.hpp"

class MockedServerRequestInterface final : public ovms::HttpAsyncWriter {
public:
    MOCK_METHOD(void, OverwriteResponseHeader, (const std::string&, const std::string&), (override));
    MOCK_METHOD(void, PartialReplyWithStatus, (std::string, ovms::HTTPStatusCode), (override));
    MOCK_METHOD(void, PartialReply, (std::string), (override));
    MOCK_METHOD(void, PartialReplyEnd, (), (override));
    MOCK_METHOD(bool, IsDisconnected, (), (const override));
    MOCK_METHOD(void, RegisterDisconnectionCallback, (std::function<void()>), (override));
    MOCK_METHOD(void, PartialReplyBegin, (std::function<void()>), (override));
};

class MockedMultiPartParser final : public ovms::MultiPartParser {
public:
    MOCK_METHOD(bool, parse, (), (override));
    MOCK_METHOD(bool, hasParseError, (), (const override));
    MOCK_METHOD(std::string, getFieldByName, (const std::string&), (const override));
    MOCK_METHOD(std::string_view, getFileContentByName, (const std::string&), (const override));
};

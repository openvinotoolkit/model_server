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

#include <string>

#include "../http_frontend/multi_part_parser_drogon_impl.hpp"

// Sanity test, drogon already unit tests it in depth
TEST(MultiPartParserDrogonImpl, GetFieldName) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    req->setBody(
        "--12345\r\n"
        "Content-Disposition: form-data; name=\"somekey\"\r\n"
        "\r\n"
        "Hello; World\r\n"
        "--12345--");

    ovms::DrogonMultiPartParser parser(req);
    ASSERT_TRUE(parser.parse());
    ASSERT_FALSE(parser.hasParseError());
    std::string val = std::string(parser.getFieldByName("somekey"));
    EXPECT_EQ(val, "Hello; World");
}
TEST(MultiPartParserDrogonImpl, GetFileContentByName) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    req->setBody(
        "--12345\r\n"
        "Content-Disposition: form-data; name=\"somekey\"; "
        "filename=\"test\"\r\n"
        "\r\n"
        "Hello; World\r\n"
        "--12345--");

    ovms::DrogonMultiPartParser parser(req);
    ASSERT_TRUE(parser.parse());
    ASSERT_FALSE(parser.hasParseError());
    std::string val = std::string(parser.getFileContentByFieldName("somekey"));
    EXPECT_EQ(val, "Hello; World");
}

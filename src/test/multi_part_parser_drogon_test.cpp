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

#include <memory>
#include <string>
#include <type_traits>

#include "../http_frontend/multi_part_parser_drogon_impl.hpp"

static_assert(std::has_virtual_destructor<ovms::MultiPartParser>::value,
    "MultiPartParser must have a virtual destructor");
static_assert(std::has_virtual_destructor<ovms::DrogonMultiPartParser>::value,
    "DrogonMultiPartParser must have a virtual destructor");

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
TEST(MultiPartParserDrogonImpl, GetArrayFieldName) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    req->setBody(
        "--12345\r\n"
        "Content-Disposition: form-data; name=\"arraykey[]\"\r\n"
        "\r\n"
        "value1\r\n"
        "--12345\r\n"
        "Content-Disposition: form-data; name=\"arraykey[]\"\r\n"
        "\r\n"
        "value2\r\n"
        "--12345\r\n"
        "Content-Disposition: form-data; name=\"arraykey[]\"\r\n"
        "\r\n"
        "value3\r\n"
        "--12345--");

    ovms::DrogonMultiPartParser parser(req);
    ASSERT_TRUE(parser.parse());
    ASSERT_FALSE(parser.hasParseError());
    auto values = parser.getArrayFieldByName("arraykey[]");
    ASSERT_EQ(values.size(), 3);
    EXPECT_EQ(values[0], "value1");
    EXPECT_EQ(values[1], "value2");
    EXPECT_EQ(values[2], "value3");
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

TEST(MultiPartParserDrogonImpl, PolymorphicCleanupViaUniquePtr) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    req->setBody(
        "--12345\r\n"
        "Content-Disposition: form-data; name=\"test_field\"\r\n"
        "\r\n"
        "test_value\r\n"
        "--12345--");

    std::unique_ptr<ovms::MultiPartParser> parser = std::make_unique<ovms::DrogonMultiPartParser>(req);
    ASSERT_TRUE(parser->parse());
    EXPECT_FALSE(parser->hasParseError());
    EXPECT_EQ(parser->getFieldByName("test_field"), "test_value");
}

namespace {
class DestructorSpyParser : public ovms::MultiPartParser {
    bool& destroyedRef_;

public:
    explicit DestructorSpyParser(bool& destroyed) :
        destroyedRef_(destroyed) {}
    ~DestructorSpyParser() override { destroyedRef_ = true; }

    bool parse() override { return true; }
    bool hasParseError() const override { return false; }
    std::vector<std::string> getArrayFieldByName(const std::string&) const override { return {}; }
    std::string getFieldByName(const std::string&) const override { return {}; }
    std::string_view getFileContentByFieldName(const std::string&) const override { return ""; }
    std::vector<std::string_view> getFilesArrayByFieldName(const std::string&) const override { return {}; }
    std::set<std::string> getAllFieldNames() const override { return {}; }
};
}  // namespace

TEST(MultiPartParserTest, VerifiesBaseVirtualDestructorDispatchesProperly) {
    bool destroyed = false;
    {
        std::unique_ptr<ovms::MultiPartParser> basePtr =
            std::make_unique<DestructorSpyParser>(destroyed);
    }
    EXPECT_TRUE(destroyed);
}

TEST(MultiPartParserDrogonImpl, GetFilesArrayAndAllFieldNames) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    req->setBody(
        "--12345\r\n"
        "Content-Disposition: form-data; name=\"image[]\"; filename=\"file1.png\"\r\n"
        "\r\n"
        "content1\r\n"
        "--12345\r\n"
        "Content-Disposition: form-data; name=\"image[]\"; filename=\"file2.png\"\r\n"
        "\r\n"
        "content2\r\n"
        "--12345\r\n"
        "Content-Disposition: form-data; name=\"prompt\"\r\n"
        "\r\n"
        "sample prompt\r\n"
        "--12345--");

    ovms::DrogonMultiPartParser parser(req);
    ASSERT_TRUE(parser.parse());
    ASSERT_FALSE(parser.hasParseError());

    auto files = parser.getFilesArrayByFieldName("image[]");
    ASSERT_EQ(files.size(), 2);
    EXPECT_EQ(files[0], "content1");
    EXPECT_EQ(files[1], "content2");

    auto allFields = parser.getAllFieldNames();
    EXPECT_TRUE(allFields.find("prompt") != allFields.end());
    EXPECT_TRUE(allFields.find("image[]") != allFields.end());
}

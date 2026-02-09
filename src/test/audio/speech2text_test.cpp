//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <drogon/drogon.h>
#include "../../http_frontend/multi_part_parser_drogon_impl.hpp"
#include "../../audio/audio_utils.hpp"
#include "../../http_rest_api_handler.hpp"
#include "../../server.hpp"
#include "rapidjson/document.h"
#include "../test_http_utils.hpp"
#include "../test_utils.hpp"
#include "../platform_utils.hpp"
#include "../constructor_enabled_model_manager.hpp"

using namespace ovms;

class Speech2TextHttpTest : public V3HttpTest {
protected:
    std::string modelName = "speech2text";
    std::string endpoint = "/v3/audio/transcriptions";
    static std::unique_ptr<std::thread> t;
    std::unordered_map<std::string, std::string> multipartHeader{{"content-type", "multipart/form-data"}};
    static std::string modelNameForm;
    static std::string body;
    rapidjson::Document d;

public:
    static void SetUpTestSuite() {
        std::string port = "9173";
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/audio/config_stt.json");
        SetUpSuite(port, configPath, t);
        modelNameForm = "--12345\r\n"
                        "Content-Disposition: form-data;name=\"model\"\r\n"
                        "\r\n"
                        "speech2text\r\n";

        body = modelNameForm + "--12345\r\n"
                               "Content-Disposition: form-data;name=\"file\";\"filename=file\""
                               "\r\nContent-Type: application/octet-stream"
                               "\r\ncontent-transfer-encoding: quoted-printable\r\n\r\n";
        std::unique_ptr<char[]> imageBytes;
        size_t fileSize;
        readFile(getGenericFullPathForSrcTest("/ovms/src/test/audio/test.wav"), fileSize, imageBytes);
        Speech2TextHttpTest::body.append(imageBytes.get(), fileSize);
        Speech2TextHttpTest::body.append("12345");
    }

    void SetUp() {
        V3HttpTest::SetUp();
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpoint, multipartHeader), ovms::StatusCode::OK);
    }

    static void TearDownTestSuite() {
        TearDownSuite(t);
    }
};
std::unique_ptr<std::thread> Speech2TextHttpTest::t;
std::string Speech2TextHttpTest::body;
std::string Speech2TextHttpTest::modelNameForm;

TEST_F(Speech2TextHttpTest, simplePositive) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    req->setBody(Speech2TextHttpTest::body);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::OK);
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    EXPECT_EQ(ok.Code(), 0);
    EXPECT_TRUE(d.HasMember("text"));
    EXPECT_TRUE(d["text"].IsString());
    EXPECT_FALSE(d.HasMember("segments"));
    EXPECT_FALSE(d.HasMember("words"));
}

TEST_F(Speech2TextHttpTest, positiveLanguage) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string language = "\r\n"
                           "Content-Disposition: form-data;name=\"language\"\r\n"
                           "\r\n"
                           "en\r\n"
                           "--12345";
    req->setBody(Speech2TextHttpTest::body + language);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::OK);
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    EXPECT_EQ(ok.Code(), 0);
    EXPECT_TRUE(d.HasMember("text"));
    EXPECT_TRUE(d["text"].IsString());
    EXPECT_FALSE(d.HasMember("segments"));
    EXPECT_FALSE(d.HasMember("words"));
}

TEST_F(Speech2TextHttpTest, positiveTemperature) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string language = "\r\n"
                           "Content-Disposition: form-data;name=\"temperature\"\r\n"
                           "\r\n"
                           "1.0\r\n"
                           "--12345";
    req->setBody(Speech2TextHttpTest::body + language);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::OK);
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    EXPECT_EQ(ok.Code(), 0);
    EXPECT_TRUE(d.HasMember("text"));
    EXPECT_TRUE(d["text"].IsString());
    EXPECT_FALSE(d.HasMember("segments"));
    EXPECT_FALSE(d.HasMember("words"));
}

TEST_F(Speech2TextHttpTest, positiveSegmentTimestamps) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string language = "\r\n"
                           "Content-Disposition: form-data;name=\"timestamp_granularities[]\"\r\n"
                           "\r\n"
                           "segment\r\n"
                           "--12345";
    req->setBody(Speech2TextHttpTest::body + language);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::OK);
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    EXPECT_EQ(ok.Code(), 0);
    EXPECT_TRUE(d.HasMember("text"));
    EXPECT_TRUE(d["text"].IsString());
    EXPECT_TRUE(d.HasMember("segments"));
    EXPECT_TRUE(d["segments"].IsArray());
    EXPECT_FALSE(d.HasMember("words"));
}

TEST_F(Speech2TextHttpTest, positiveWordTimestamps) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string multipartBody = "--12345\r\n"
                                "Content-Disposition: form-data;name=\"model\"\r\n"
                                "\r\n"
                                "speech2textWordTimestamps\r\n"
                                "--12345\r\n"
                                "Content-Disposition: form-data;name=\"timestamp_granularities[]\"\r\n"
                                "\r\n"
                                "word\r\n"
                                "--12345\r\n"
                                "Content-Disposition: form-data;name=\"file\";\"filename=file\""
                                "\r\nContent-Type: application/octet-stream"
                                "\r\ncontent-transfer-encoding: quoted-printable\r\n\r\n";
    std::unique_ptr<char[]> imageBytes;
    size_t fileSize;
    readFile(getGenericFullPathForSrcTest("/ovms/src/test/audio/test.wav"), fileSize, imageBytes);
    multipartBody.append(imageBytes.get(), fileSize);
    multipartBody.append("12345");
    req->setBody(multipartBody);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::OK);
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    EXPECT_EQ(ok.Code(), 0);
    EXPECT_TRUE(d.HasMember("text"));
    EXPECT_TRUE(d["text"].IsString());
    EXPECT_TRUE(d.HasMember("words"));
    EXPECT_TRUE(d["words"].IsArray());
    EXPECT_FALSE(d.HasMember("segments"));
}

TEST_F(Speech2TextHttpTest, positiveBothTimestampsTypes) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string multipartBody = "--12345\r\n"
                                "Content-Disposition: form-data;name=\"model\"\r\n"
                                "\r\n"
                                "speech2textWordTimestamps\r\n"
                                "--12345\r\n"
                                "Content-Disposition: form-data;name=\"timestamp_granularities[]\"\r\n"
                                "\r\n"
                                "word\r\n"
                                "--12345\r\n"
                                "Content-Disposition: form-data;name=\"timestamp_granularities[]\"\r\n"
                                "\r\n"
                                "segment\r\n"
                                "--12345\r\n"
                                "Content-Disposition: form-data;name=\"file\";\"filename=file\""
                                "\r\nContent-Type: application/octet-stream"
                                "\r\ncontent-transfer-encoding: quoted-printable\r\n\r\n";
    std::unique_ptr<char[]> imageBytes;
    size_t fileSize;
    readFile(getGenericFullPathForSrcTest("/ovms/src/test/audio/test.wav"), fileSize, imageBytes);
    multipartBody.append(imageBytes.get(), fileSize);
    multipartBody.append("12345");
    req->setBody(multipartBody);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::OK);
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    EXPECT_EQ(ok.Code(), 0);
    EXPECT_TRUE(d.HasMember("text"));
    EXPECT_TRUE(d["text"].IsString());
    EXPECT_TRUE(d.HasMember("words"));
    EXPECT_TRUE(d["words"].IsArray());
    EXPECT_TRUE(d.HasMember("segments"));
    EXPECT_TRUE(d["segments"].IsArray());
}

TEST_F(Speech2TextHttpTest, invalidFile) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string invalidBody = modelNameForm + "--12345\r\n"
                                              "Content-Disposition: form-data;name=\"file\";\"filename=file\""
                                              "\r\nContent-Type: application/octet-stream"
                                              "\r\ncontent-transfer-encoding: quoted-printable\r\n\r\n";
    invalidBody.append("INVALID");
    req->setBody(invalidBody);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    auto status = handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest);
    ASSERT_EQ(
        status.getCode(),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    std::string expectedMsg = "Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \n"
                              "Calculator::Process() for node \"S2tExecutor\" failed: File parsing fails";
    EXPECT_EQ(status.string(), expectedMsg);
}

TEST_F(Speech2TextHttpTest, invalidLanguageCode) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string language = "\r\n"
                           "Content-Disposition: form-data;name=\"language\"\r\n"
                           "\r\n"
                           "xD\r\n"
                           "--12345";
    req->setBody(Speech2TextHttpTest::body + language);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    auto status = handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest);
    ASSERT_EQ(
        status.getCode(),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
}

TEST_F(Speech2TextHttpTest, invalidLanguageTooLong) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string language = "\r\n"
                           "Content-Disposition: form-data;name=\"language\"\r\n"
                           "\r\n"
                           "TOO_LONG\r\n"
                           "--12345";
    req->setBody(Speech2TextHttpTest::body + language);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    auto status = handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest);
    ASSERT_EQ(
        status.getCode(),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    std::string expectedMsg = "Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \n"
                              "Calculator::Process() for node \"S2tExecutor\" failed: Invalid language code.";
    EXPECT_EQ(status.string(), expectedMsg);
}

TEST_F(Speech2TextHttpTest, invalidTemperatureOutOfRange) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string language = "\r\n"
                           "Content-Disposition: form-data;name=\"temperature\"\r\n"
                           "\r\n"
                           "10.0\r\n"
                           "--12345";
    req->setBody(Speech2TextHttpTest::body + language);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    auto status = handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest);
    ASSERT_EQ(
        status.getCode(),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    std::string expectedMsg = "Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \n"
                              "Calculator::Process() for node \"S2tExecutor\" failed: Temperature out of range(0.0, 2.0)";
    EXPECT_EQ(status.string(), expectedMsg);
}

TEST_F(Speech2TextHttpTest, invalidTimestampType) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string language = "\r\n"
                           "Content-Disposition: form-data;name=\"timestamp_granularities[]\"\r\n"
                           "\r\n"
                           "INVALID\r\n"
                           "--12345";
    req->setBody(Speech2TextHttpTest::body + language);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    auto status = handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest);
    ASSERT_EQ(
        status.getCode(),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    std::string expectedMsg = "Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \n"
                              "Calculator::Process() for node \"S2tExecutor\" failed: Invalid timestamp_granularities type. Allowed types: \"segment\", \"word\"";
    EXPECT_EQ(status.string(), expectedMsg);
}

TEST_F(Speech2TextHttpTest, emptyTimestampType) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    std::string language = "\r\n"
                           "Content-Disposition: form-data;name=\"timestamp_granularities[]\"\r\n"
                           "\r\n"
                           "\r\n"
                           "--12345";
    req->setBody(Speech2TextHttpTest::body + language);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);
    std::string requestBody = "";
    auto status = handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest);
    ASSERT_EQ(
        status.getCode(),
        ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR);
    std::string expectedMsg = "Mediapipe execution failed. MP status - INVALID_ARGUMENT: CalculatorGraph::Run() failed: \n"
                              "Calculator::Process() for node \"S2tExecutor\" failed: Invalid timestamp_granularities type. Allowed types: \"segment\", \"word\"";
    EXPECT_EQ(status.string(), expectedMsg);
}

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
        std::unique_ptr<char[]> audioBytes;
        size_t fileSize;
        readFile(getGenericFullPathForSrcTest("/ovms/src/test/audio/test.wav"), fileSize, audioBytes);
        Speech2TextHttpTest::body.append(audioBytes.get(), fileSize);
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

// ====================== Speech2Text Streaming Tests ======================

class Speech2TextStreamingTest : public Speech2TextHttpTest {
protected:
    // Builds a multipart body identical to the base fixture body but with an
    // extra `stream=true` field appended.
    static std::string streamingBody() {
        const std::string streamField = "\r\n"
                                        "Content-Disposition: form-data;name=\"stream\"\r\n"
                                        "\r\n"
                                        "true\r\n"
                                        "--12345";
        return Speech2TextHttpTest::body + streamField;
    }

    void SetUp() override {
        Speech2TextHttpTest::SetUp();
        ON_CALL(*writer, PartialReplyBegin(::testing::_))
            .WillByDefault(testing::Invoke([](std::function<void()> fn) { fn(); }));
        ON_CALL(*writer, IsDisconnected())
            .WillByDefault(::testing::Return(false));
    }
};

TEST_F(Speech2TextStreamingTest, streamingTranscriptionReceivesDeltaAndDoneEvents) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    req->setBody(streamingBody());
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);

    std::vector<std::string> receivedChunks;
    EXPECT_CALL(*writer, PartialReply(::testing::_))
        .WillRepeatedly([&receivedChunks](std::string chunk) {
            receivedChunks.push_back(std::move(chunk));
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);

    std::string requestBody;
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_FALSE(receivedChunks.empty());

    const std::string dataPrefix = "data: ";
    // Validate each delta chunk (all but the last)
    for (size_t i = 0; i + 1 < receivedChunks.size(); ++i) {
        const std::string& chunk = receivedChunks[i];
        ASSERT_GE(chunk.size(), dataPrefix.size());
        EXPECT_EQ(chunk.substr(0, dataPrefix.size()), dataPrefix) << "Chunk " << i << " missing SSE prefix";
        std::string json = chunk.substr(dataPrefix.size());
        // Trim trailing newlines
        while (!json.empty() && (json.back() == '\n' || json.back() == '\r'))
            json.pop_back();
        rapidjson::Document d;
        ASSERT_EQ(d.Parse(json.c_str()).HasParseError(), false) << "Chunk " << i << " is not valid JSON";
        ASSERT_TRUE(d.HasMember("type")) << "Chunk " << i << " missing 'type'";
        EXPECT_STREQ(d["type"].GetString(), "transcript.text.delta") << "Chunk " << i;
        ASSERT_TRUE(d.HasMember("delta")) << "Chunk " << i << " missing 'delta'";
        EXPECT_TRUE(d["delta"].IsString()) << "Chunk " << i << " 'delta' is not a string";
        ASSERT_TRUE(d.HasMember("logprobs")) << "Chunk " << i << " missing 'logprobs'";
        EXPECT_TRUE(d["logprobs"].IsArray()) << "Chunk " << i << " 'logprobs' is not an array";
    }

    // Validate the final done event
    const std::string& lastChunk = receivedChunks.back();
    ASSERT_GE(lastChunk.size(), dataPrefix.size());
    EXPECT_EQ(lastChunk.substr(0, dataPrefix.size()), dataPrefix);
    std::string lastJson = lastChunk.substr(dataPrefix.size());
    while (!lastJson.empty() && (lastJson.back() == '\n' || lastJson.back() == '\r'))
        lastJson.pop_back();
    rapidjson::Document doneDoc;
    ASSERT_EQ(doneDoc.Parse(lastJson.c_str()).HasParseError(), false) << "Done event is not valid JSON";
    ASSERT_TRUE(doneDoc.HasMember("type"));
    EXPECT_STREQ(doneDoc["type"].GetString(), "transcript.text.done");
    ASSERT_TRUE(doneDoc.HasMember("text"));
    EXPECT_TRUE(doneDoc["text"].IsString());
    EXPECT_FALSE(std::string(doneDoc["text"].GetString()).empty());
    ASSERT_TRUE(doneDoc.HasMember("logprobs"));
    EXPECT_TRUE(doneDoc["logprobs"].IsArray());
}

TEST_F(Speech2TextStreamingTest, streamingTranscriptionDoneTextMatchesConcatenatedDeltas) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    req->setBody(streamingBody());
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);

    std::vector<std::string> receivedChunks;
    ON_CALL(*writer, PartialReply(::testing::_))
        .WillByDefault([&receivedChunks](std::string chunk) {
            receivedChunks.push_back(std::move(chunk));
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);

    std::string requestBody;
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_GE(receivedChunks.size(), 1u);

    const std::string dataPrefix = "data: ";
    auto parseChunkJson = [&dataPrefix](const std::string& chunk) {
        std::string json = chunk.substr(dataPrefix.size());
        while (!json.empty() && (json.back() == '\n' || json.back() == '\r'))
            json.pop_back();
        return json;
    };

    // Collect all delta text
    std::string concatenatedDeltas;
    for (size_t i = 0; i + 1 < receivedChunks.size(); ++i) {
        rapidjson::Document d;
        d.Parse(parseChunkJson(receivedChunks[i]).c_str());
        if (d.HasMember("delta") && d["delta"].IsString()) {
            concatenatedDeltas += d["delta"].GetString();
        }
    }

    // Get done event text
    rapidjson::Document doneDoc;
    doneDoc.Parse(parseChunkJson(receivedChunks.back()).c_str());
    ASSERT_EQ(doneDoc.HasParseError(), false);
    ASSERT_TRUE(doneDoc.HasMember("text"));
    const std::string doneText = doneDoc["text"].GetString();

    EXPECT_EQ(concatenatedDeltas, doneText)
        << "Concatenated deltas should equal the final 'done' text";
}

TEST_F(Speech2TextStreamingTest, streamingTranscriptionWithLanguage) {
    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    const std::string streamAndLanguage = "\r\n"
                                          "Content-Disposition: form-data;name=\"stream\"\r\n"
                                          "\r\n"
                                          "true\r\n"
                                          "--12345\r\n"
                                          "Content-Disposition: form-data;name=\"language\"\r\n"
                                          "\r\n"
                                          "en\r\n"
                                          "--12345";
    req->setBody(Speech2TextHttpTest::body + streamAndLanguage);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);

    std::vector<std::string> receivedChunks;
    ON_CALL(*writer, PartialReply(::testing::_))
        .WillByDefault([&receivedChunks](std::string chunk) {
            receivedChunks.push_back(std::move(chunk));
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);

    std::string requestBody;
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::PARTIAL_END);

    ASSERT_FALSE(receivedChunks.empty());
    // Verify last chunk is a done event
    const std::string dataPrefix = "data: ";
    std::string lastJson = receivedChunks.back().substr(dataPrefix.size());
    while (!lastJson.empty() && (lastJson.back() == '\n' || lastJson.back() == '\r'))
        lastJson.pop_back();
    rapidjson::Document doneDoc;
    ASSERT_EQ(doneDoc.Parse(lastJson.c_str()).HasParseError(), false);
    ASSERT_TRUE(doneDoc.HasMember("type"));
    EXPECT_STREQ(doneDoc["type"].GetString(), "transcript.text.done");
}

TEST_F(Speech2TextStreamingTest, streamingTranscriptionInvalidFileReturnsError) {
    const std::string invalidBody = Speech2TextHttpTest::modelNameForm +
                                    "--12345\r\n"
                                    "Content-Disposition: form-data;name=\"stream\"\r\n"
                                    "\r\n"
                                    "true\r\n"
                                    "--12345\r\n"
                                    "Content-Disposition: form-data;name=\"file\";\"filename=file\""
                                    "\r\nContent-Type: application/octet-stream"
                                    "\r\ncontent-transfer-encoding: quoted-printable\r\n\r\n"
                                    "INVALID_AUDIO12345";

    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    req->setBody(invalidBody);
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([](std::string responseBody, ovms::HTTPStatusCode code) {
            EXPECT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
            rapidjson::Document d;
            ASSERT_EQ(d.Parse(responseBody.c_str()).HasParseError(), false);
            ASSERT_TRUE(d.HasMember("error"));
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);

    std::string requestBody;
    ASSERT_EQ(
        handler->dispatchToProcessor(endpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::PARTIAL_END);
}

TEST_F(Speech2TextStreamingTest, streamingTranslationIsNotSupported) {
    const std::string translationEndpoint = "/v3/audio/translations";
    ASSERT_EQ(handler->parseRequestComponents(comp, "POST", translationEndpoint, multipartHeader), ovms::StatusCode::OK);

    auto req = drogon::HttpRequest::newHttpRequest();
    req->setMethod(drogon::Post);
    req->addHeader("content-type", "multipart/form-data; boundary=\"12345\"");
    req->setBody(streamingBody());
    std::shared_ptr<MultiPartParser> multiPartParserWithRequest = std::make_shared<DrogonMultiPartParser>(req);

    EXPECT_CALL(*writer, PartialReplyWithStatus(::testing::_, ::testing::_))
        .WillOnce([](std::string responseBody, ovms::HTTPStatusCode code) {
            EXPECT_EQ(code, ovms::HTTPStatusCode::BAD_REQUEST);
            rapidjson::Document d;
            ASSERT_EQ(d.Parse(responseBody.c_str()).HasParseError(), false);
            ASSERT_TRUE(d.HasMember("error"));
            ASSERT_TRUE(d["error"].IsString());
            EXPECT_NE(std::string(d["error"].GetString()).find("streaming is not supported for translations endpoint"), std::string::npos);
        });
    EXPECT_CALL(*writer, PartialReplyEnd()).Times(1);

    std::string requestBody;
    ASSERT_EQ(
        handler->dispatchToProcessor(translationEndpoint, requestBody, &response, comp, responseComponents, writer, multiPartParserWithRequest),
        ovms::StatusCode::PARTIAL_END);
}

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
    std::unique_ptr<char[]> audioBytes;
    size_t fileSize;
    readFile(getGenericFullPathForSrcTest("/ovms/src/test/audio/test.wav"), fileSize, audioBytes);
    multipartBody.append(audioBytes.get(), fileSize);
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
    std::unique_ptr<char[]> audioBytes;
    size_t fileSize;
    readFile(getGenericFullPathForSrcTest("/ovms/src/test/audio/test.wav"), fileSize, audioBytes);
    multipartBody.append(audioBytes.get(), fileSize);
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

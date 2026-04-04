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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../http_rest_api_handler.hpp"
#include "../ov_utils.hpp"
#include "../server.hpp"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"  // TODO: Move out together with rerank tests
#include "rapidjson/writer.h"        // TODO: Move out together with rerank tests
#include "test_http_utils.hpp"
#include "test_utils.hpp"
#include "platform_utils.hpp"

using namespace ovms;

class RerankTokenizeHttpTest : public V3HttpTest {
protected:
    static std::unique_ptr<std::thread> t;

public:
    const std::string endpointTokenize = "/v3/tokenize";
    static void SetUpTestSuite() {
        std::string port = "9173";
        std::string configPath = getGenericFullPathForSrcTest("/ovms/src/test/rerank/config.json");
        SetUpSuite(port, configPath, t);
    }

    void SetUp() {
        V3HttpTest::SetUp();
        ASSERT_EQ(handler->parseRequestComponents(comp, "POST", endpointTokenize, headers), ovms::StatusCode::OK);
    }

    static void TearDownTestSuite() {
        TearDownSuite(t);
    }

    static void AssertTokenizationResult(const std::string& response, const std::vector<int>& expectedTokens) {
        rapidjson::Document d;
        rapidjson::ParseResult ok = d.Parse(response.c_str());
        ASSERT_EQ(ok.Code(), 0);
        ASSERT_TRUE(d.HasMember("tokens"));
        ASSERT_TRUE(d["tokens"].IsArray());
        ASSERT_EQ(d["tokens"].Size(), expectedTokens.size());
        for (size_t i = 0; i < expectedTokens.size(); ++i) {
            ASSERT_EQ(d["tokens"][(rapidjson::SizeType)i].GetInt(), expectedTokens[i]);
        }
    }

    static void AssertTokenizationResult(const std::string& response, const std::vector<std::vector<int>>& expectedTokensBatch) {
        rapidjson::Document d;
        rapidjson::ParseResult ok = d.Parse(response.c_str());
        ASSERT_EQ(ok.Code(), 0);
        ASSERT_TRUE(d.HasMember("tokens"));
        ASSERT_TRUE(d["tokens"].IsArray());
        ASSERT_EQ(d["tokens"].Size(), expectedTokensBatch.size());
        for (size_t i = 0; i < expectedTokensBatch.size(); ++i) {
            const auto& expectedTokens = expectedTokensBatch[i];
            ASSERT_TRUE(d["tokens"][(rapidjson::SizeType)i].IsArray());
            ASSERT_EQ(d["tokens"][(rapidjson::SizeType)i].Size(), expectedTokens.size());
            for (size_t j = 0; j < expectedTokens.size(); ++j) {
                ASSERT_EQ(d["tokens"][(rapidjson::SizeType)i][(rapidjson::SizeType)j].GetInt(), expectedTokens[j]);
            }
        }
    }
};

std::unique_ptr<std::thread> RerankTokenizeHttpTest::t;

TEST_F(RerankTokenizeHttpTest, tokenizePositive) {
    std::string requestBody = R"(
        {
            "model": "rerank_ov",
            "text": "hello world"
        }
    )";
    std::vector<int> expectedTokens = {33600, 31, 8999};
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    AssertTokenizationResult(response, expectedTokens);
}

TEST_F(RerankTokenizeHttpTest, tokenizeNegativeMissingText) {
    std::string requestBody = R"(
        {
                "model": "rerank_ov"
        }
    )";
    Status status = handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
}

TEST_F(RerankTokenizeHttpTest, tokenizeNegativeInvalidModel) {
    std::string requestBody = R"(
        {
            "model": "non_existing_model",
            "text": "hello world"
        }
    )";
    Status status = handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING) << status.string();
}

TEST_F(RerankTokenizeHttpTest, tokenizePositiveMaxLenParam) {
    std::string requestBody = R"(
        {
            "model": "rerank_ov",
            "text": "hello world hello world",
            "max_length": 3
        }
    )";
    std::vector<int> expectedTokens = {33600, 31, 8999};
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    AssertTokenizationResult(response, expectedTokens);
}

TEST_F(RerankTokenizeHttpTest, tokenizePositivePadToMaxLenParam) {
    std::string requestBody = R"(
        {
            "model": "rerank_ov",
            "text": "hello world",
            "max_length": 100,
            "pad_to_max_length": true
        }
    )";
    std::vector<int> expectedTokens(97, 1);
    expectedTokens.insert(expectedTokens.begin(), {33600, 31, 8999});
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    AssertTokenizationResult(response, expectedTokens);
}

TEST_F(RerankTokenizeHttpTest, tokenizePositivePaddingSideLeft) {
    std::string requestBody = R"(
        {
            "model": "rerank_ov",
            "text": "hello world",
            "max_length": 100,
            "pad_to_max_length": true,
            "padding_side": "left"
        }
    )";
    std::vector<int> expectedTokens(97, 1);
    expectedTokens.insert(expectedTokens.end(), {33600, 31, 8999});
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    AssertTokenizationResult(response, expectedTokens);
}

TEST_F(RerankTokenizeHttpTest, tokenizePositivePaddingSideRight) {
    std::string requestBody = R"(
        {
            "model": "rerank_ov",
            "text": "hello world",
            "max_length": 100,
            "pad_to_max_length": true,
            "padding_side": "right"
        }
    )";
    std::vector<int> expectedTokens(97, 1);
    expectedTokens.insert(expectedTokens.begin(), {33600, 31, 8999});
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    AssertTokenizationResult(response, expectedTokens);
}

TEST_F(RerankTokenizeHttpTest, tokenizeNegativeInvalidPaddingSide) {
    std::string requestBody = R"(
        {
            "model": "rerank_ov",
            "text": "hello world",
            "padding_side": "invalid_value"
        }
    )";
    Status status = handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    ASSERT_EQ(status, ovms::StatusCode::MEDIAPIPE_EXECUTION_ERROR) << status.string();
}

TEST_F(RerankTokenizeHttpTest, tokenizePositiveMaxLengthIgnored) {
    std::string requestBody = R"(
        {
            "model": "rerank_ov",
            "text": "hello world",
            "max_length": 513,
            "pad_to_max_length": true
        }
    )";
    std::vector<int> expectedTokens(510, 1);
    expectedTokens.insert(expectedTokens.begin(), {33600, 31, 8999});
    ASSERT_EQ(handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    AssertTokenizationResult(response, expectedTokens);
}

TEST_F(RerankTokenizeHttpTest, tokenizePositiveBatch) {
    std::string requestBody = R"(
        {
            "model": "rerank_ov",
            "text": ["hello", "hello world", "hello hello hello world"]
        }
    )";
    Status status = handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    std::vector<std::vector<int>> expectedTokens = {
        {33600, 31},
        {33600, 31, 8999},
        {33600, 31, 33600, 31, 33600, 31, 8999}};
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    AssertTokenizationResult(response, expectedTokens);
}

TEST_F(RerankTokenizeHttpTest, tokenizeBatchWithPadToMaxLen) {
    std::string requestBody = R"(
        {
            "model": "rerank_ov",
            "text": ["hello", "hello world", "hello hello hello world"],
            "max_length": 6,
            "pad_to_max_length": true
        }
    )";
    Status status = handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser);
    std::vector<std::vector<int>> expectedTokens = {
        {33600, 31, 1, 1, 1, 1},
        {33600, 31, 8999, 1, 1, 1},
        {33600, 31, 33600, 31, 33600, 31}};
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(response.c_str());
    ASSERT_EQ(ok.Code(), 0);
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    AssertTokenizationResult(response, expectedTokens);
}

TEST_F(RerankTokenizeHttpTest, tokenizeIgnoreAddSpecialTokensParameter) {
    std::string requestBody = R"(
        {
            "model": "rerank_ov",
            "text": "hello world",
            "max_length": 3,
            "add_special_tokens": true
        }
    )";
    std::vector<int> expectedTokens = {33600, 31, 8999};
    ASSERT_EQ(
        handler->dispatchToProcessor(endpointTokenize, requestBody, &response, comp, responseComponents, writer, multiPartParser),
        ovms::StatusCode::OK);
    AssertTokenizationResult(response, expectedTokens);
}
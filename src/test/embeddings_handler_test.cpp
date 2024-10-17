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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../embeddings/embeddings_api.hpp"
#include "rapidjson/document.h"

TEST(EmbeddingsDeserialization, singleStringInput) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": "dummyInput"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, EncodingFormat::FLOAT);
    auto strings = std::get_if<std::vector<std::string>>(&embeddingsRequest.input);
    ASSERT_NE(strings, nullptr);
    ASSERT_EQ(strings->size(), 1);
    ASSERT_EQ(strings->at(0), "dummyInput");
}

TEST(EmbeddingsDeserialization, multipleStringInput) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": ["one", "two", "three"]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, EncodingFormat::FLOAT);
    auto strings = std::get_if<std::vector<std::string>>(&embeddingsRequest.input);
    ASSERT_NE(strings, nullptr);
    ASSERT_EQ(strings->size(), 3);
    ASSERT_EQ(strings->at(0), "one");
    ASSERT_EQ(strings->at(1), "two");
    ASSERT_EQ(strings->at(2), "three");
}

TEST(EmbeddingsDeserialization, handler) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": ["one", "two", "three"]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    EmbeddingsHandler handler(d);
    ASSERT_EQ(handler.parseRequest(), absl::OkStatus());
    ASSERT_EQ(handler.getEncodingFormat(), EncodingFormat::FLOAT);
    auto input = handler.getInput();
    auto strings = std::get_if<std::vector<std::string>>(&input);
    ASSERT_NE(strings, nullptr);
    ASSERT_EQ(strings->size(), 3);
    ASSERT_EQ(strings->at(0), "one");
    ASSERT_EQ(strings->at(1), "two");
    ASSERT_EQ(strings->at(2), "three");
}

TEST(EmbeddingsDeserialization, malformedInput) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": ["one", 2, "three"]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "every element in input array should be string");
}

TEST(EmbeddingsDeserialization, invalidEncoding) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": ["one", "three"],
"encoding_format": "dummy"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "encoding_format should either base64 or float");
}

TEST(EmbeddingsDeserialization, invalidEncodingType) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": ["one", "three"],
"encoding_format": 42
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "encoding_format should be string");
}

TEST(EmbeddingsDeserialization, malformedInputType) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": 1
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "input should be string or array of strings");
}

TEST(EmbeddingsDeserialization, noInput) {
    std::string requestBody = R"(
        {
            "model": "embeddings"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "input field is required");
}

TEST(EmbeddingsDeserialization, multipleStringInputBase64) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": ["one", "two", "three"],
"encoding_format": "base64"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, EncodingFormat::BASE64);
    auto strings = std::get_if<std::vector<std::string>>(&embeddingsRequest.input);
    ASSERT_NE(strings, nullptr);
    ASSERT_EQ(strings->size(), 3);
    ASSERT_EQ(strings->at(0), "one");
    ASSERT_EQ(strings->at(1), "two");
    ASSERT_EQ(strings->at(2), "three");
}

TEST(EmbeddingsDeserialization, multipleStringInputFloat) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": ["one", "two", "three"],
"encoding_format": "float"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, EncodingFormat::FLOAT);
    auto strings = std::get_if<std::vector<std::string>>(&embeddingsRequest.input);
    ASSERT_NE(strings, nullptr);
    ASSERT_EQ(strings->size(), 3);
    ASSERT_EQ(strings->at(0), "one");
    ASSERT_EQ(strings->at(1), "two");
    ASSERT_EQ(strings->at(2), "three");
}

TEST(EmbeddingsSerialization, simplePositive) {
    bool normalieEmbeddings = false;
    rapidjson::StringBuffer buffer;
    std::vector<float> tensorsData{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    std::vector<size_t> shape{2, 3, 3};
    ov::Tensor embeddingsTensor = ov::Tensor(ov::element::Type_t::f32, shape, tensorsData.data());
    rapidjson::Document notUsed;
    EmbeddingsHandler handler(notUsed);
    auto status = handler.parseResponse(buffer, embeddingsTensor, normalieEmbeddings);
    ASSERT_TRUE(status.ok());
    std::string expectedResponse = R"({"object":"list","data":[{"object":"embedding","embedding":[1.0,2.0,3.0],"index":0},{"object":"embedding","embedding":[1.0,2.0,3.0],"index":1}],"usage":{"prompt_tokens":0,"total_tokens":0}})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}

TEST(EmbeddingsSerialization, positiveNormalization) {
    bool normalieEmbeddings = true;
    rapidjson::StringBuffer buffer;
    std::vector<float> tensorsData{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    std::vector<size_t> shape{2, 3, 3};
    ov::Tensor embeddingsTensor = ov::Tensor(ov::element::Type_t::f32, shape, tensorsData.data());
    rapidjson::Document notUsed;
    EmbeddingsHandler handler(notUsed);
    auto status = handler.parseResponse(buffer, embeddingsTensor, normalieEmbeddings);
    ASSERT_TRUE(status.ok());
    std::string expectedResponse = R"({"object":"list","data":[{"object":"embedding","embedding":[0.26726123690605164,0.5345224738121033,0.8017837405204773],"index":0},{"object":"embedding","embedding":[0.26726123690605164,0.5345224738121033,0.8017837405204773],"index":1}],"usage":{"prompt_tokens":0,"total_tokens":0}})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}

TEST(EmbeddingsSerialization, positiveBase64) {
    bool normalieEmbeddings = false;
    rapidjson::StringBuffer buffer;
    std::vector<float> tensorsData{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    std::vector<size_t> shape{2, 3, 3};
    ov::Tensor embeddingsTensor = ov::Tensor(ov::element::Type_t::f32, shape, tensorsData.data());
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": ["one", "two", "three"],
            "encoding_format": "base64"
        }
    )";
    rapidjson::Document document;
    rapidjson::ParseResult ok = document.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    EmbeddingsHandler handler(document);
    auto status = handler.parseRequest();
    ASSERT_TRUE(status.ok());
    status = handler.parseResponse(buffer, embeddingsTensor, normalieEmbeddings);
    ASSERT_TRUE(status.ok());
    std::string expectedResponse = R"({"object":"list","data":[{"object":"embedding","embedding":"AACAPwAAAEAAAEBA","index":0},{"object":"embedding","embedding":"AACAPwAAAEAAAEBA","index":1}],"usage":{"prompt_tokens":0,"total_tokens":0}})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}

TEST(EmbeddingsSerialization, positiveUsage) {
    bool normalieEmbeddings = false;
    rapidjson::StringBuffer buffer;
    std::vector<float> tensorsData{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    std::vector<size_t> shape{2, 3, 3};
    ov::Tensor embeddingsTensor = ov::Tensor(ov::element::Type_t::f32, shape, tensorsData.data());
    rapidjson::Document notUsed;
    EmbeddingsHandler handler(notUsed);
    handler.setPromptTokensUsage(50);
    auto status = handler.parseResponse(buffer, embeddingsTensor, normalieEmbeddings);
    ASSERT_TRUE(status.ok());
    std::string expectedResponse = R"({"object":"list","data":[{"object":"embedding","embedding":[1.0,2.0,3.0],"index":0},{"object":"embedding","embedding":[1.0,2.0,3.0],"index":1}],"usage":{"prompt_tokens":50,"total_tokens":50}})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}

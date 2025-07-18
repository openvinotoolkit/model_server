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
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<ovms::EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, ovms::EmbeddingsRequest::EncodingFormat::FLOAT);
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
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<ovms::EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, ovms::EmbeddingsRequest::EncodingFormat::FLOAT);
    auto strings = std::get_if<std::vector<std::string>>(&embeddingsRequest.input);
    ASSERT_NE(strings, nullptr);
    ASSERT_EQ(strings->size(), 3);
    ASSERT_EQ(strings->at(0), "one");
    ASSERT_EQ(strings->at(1), "two");
    ASSERT_EQ(strings->at(2), "three");
}

TEST(EmbeddingsDeserialization, intInput) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [1, 2, 3]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<ovms::EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, ovms::EmbeddingsRequest::EncodingFormat::FLOAT);
    auto ints = std::get_if<std::vector<std::vector<int64_t>>>(&embeddingsRequest.input);
    ASSERT_NE(ints, nullptr);
    ASSERT_EQ(ints->size(), 1);
    ASSERT_EQ(ints->at(0).size(), 3);
    ASSERT_EQ(ints->at(0).at(0), 1);
    ASSERT_EQ(ints->at(0).at(1), 2);
    ASSERT_EQ(ints->at(0).at(2), 3);
}

TEST(EmbeddingsDeserialization, multipleIntInput) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [[1, 2, 3], [4, 5, 6]]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<ovms::EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, ovms::EmbeddingsRequest::EncodingFormat::FLOAT);
    auto ints = std::get_if<std::vector<std::vector<int64_t>>>(&embeddingsRequest.input);
    ASSERT_NE(ints, nullptr);
    ASSERT_EQ(ints->size(), 2);
    ASSERT_EQ(ints->at(0).size(), 3);
    ASSERT_EQ(ints->at(0).at(0), 1);
    ASSERT_EQ(ints->at(0).at(1), 2);
    ASSERT_EQ(ints->at(0).at(2), 3);
    ASSERT_EQ(ints->at(1).size(), 3);
    ASSERT_EQ(ints->at(1).at(0), 4);
    ASSERT_EQ(ints->at(1).at(1), 5);
    ASSERT_EQ(ints->at(1).at(2), 6);
}

TEST(EmbeddingsDeserialization, multipleIntInputLengths) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [[1, 2, 3, 4, 5, 6], [4, 5, 6, 7], [7, 8]]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<ovms::EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, ovms::EmbeddingsRequest::EncodingFormat::FLOAT);
    auto ints = std::get_if<std::vector<std::vector<int64_t>>>(&embeddingsRequest.input);
    ASSERT_NE(ints, nullptr);
    ASSERT_EQ(ints->size(), 3);
    ASSERT_EQ(ints->at(0).size(), 6);
    ASSERT_EQ(ints->at(0).at(0), 1);
    ASSERT_EQ(ints->at(0).at(1), 2);
    ASSERT_EQ(ints->at(0).at(2), 3);
    ASSERT_EQ(ints->at(0).at(3), 4);
    ASSERT_EQ(ints->at(0).at(4), 5);
    ASSERT_EQ(ints->at(0).at(5), 6);
    ASSERT_EQ(ints->at(1).size(), 4);
    ASSERT_EQ(ints->at(1).at(0), 4);
    ASSERT_EQ(ints->at(1).at(1), 5);
    ASSERT_EQ(ints->at(1).at(2), 6);
    ASSERT_EQ(ints->at(2).size(), 2);
    ASSERT_EQ(ints->at(2).at(0), 7);
    ASSERT_EQ(ints->at(2).at(1), 8);
}

TEST(EmbeddingsDeserialization, malformedMultipleIntInput) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [[1, 2, 3], "string", [4, 5, 6]]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "input must be homogeneous");
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
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "input must be homogeneous");
}

TEST(EmbeddingsDeserialization, malformedInput2) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [[62, 12, 4], 5, 2 ]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "input must be homogeneous");
}

TEST(EmbeddingsDeserialization, malformedInput3) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [[62, 71, true, 5, "abc", 1 ], [1, 2]]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "input must be homogeneous");
}

TEST(EmbeddingsDeserialization, malformedInput4) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [[62, 71, 5, 1 ], ["string"],  [1, 2]]
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "input must be homogeneous");
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
    ovms::EmbeddingsHandler handler(d);
    ASSERT_EQ(handler.parseRequest(), absl::OkStatus());
    ASSERT_EQ(handler.getEncodingFormat(), ovms::EmbeddingsRequest::EncodingFormat::FLOAT);
    auto input = handler.getInput();
    ASSERT_EQ(std::get_if<std::vector<std::vector<int64_t>>>(&input), nullptr);
    auto strings = std::get_if<std::vector<std::string>>(&input);
    ASSERT_NE(strings, nullptr);
    ASSERT_EQ(strings->size(), 3);
    ASSERT_EQ(strings->at(0), "one");
    ASSERT_EQ(strings->at(1), "two");
    ASSERT_EQ(strings->at(2), "three");
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
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
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
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
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
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "input should be string, array of strings or array of integers");
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
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
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
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<ovms::EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, ovms::EmbeddingsRequest::EncodingFormat::BASE64);
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
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_EQ(std::get_if<std::string>(&request), nullptr);
    auto embeddingsRequest = std::get<ovms::EmbeddingsRequest>(request);
    ASSERT_EQ(embeddingsRequest.encoding_format, ovms::EmbeddingsRequest::EncodingFormat::FLOAT);
    auto strings = std::get_if<std::vector<std::string>>(&embeddingsRequest.input);
    ASSERT_NE(strings, nullptr);
    ASSERT_EQ(strings->size(), 3);
    ASSERT_EQ(strings->at(0), "one");
    ASSERT_EQ(strings->at(1), "two");
    ASSERT_EQ(strings->at(2), "three");
}

TEST(EmbeddingsDeserialization, emptyInputArray) {
    std::string requestBody = R"(
        {
            "model": "embeddings",
            "input": [  ],
            "encoding_format": "float"
        }
    )";
    rapidjson::Document d;
    rapidjson::ParseResult ok = d.Parse(requestBody.c_str());
    ASSERT_EQ(ok.Code(), 0);
    auto request = ovms::EmbeddingsRequest::fromJson(&d);
    ASSERT_NE(std::get_if<std::string>(&request), nullptr);
    auto error = *std::get_if<std::string>(&request);
    ASSERT_EQ(error, "input array should not be empty");
}

TEST(EmbeddingsSerialization, simplePositive) {
    bool normalieEmbeddings = false;
    rapidjson::StringBuffer buffer;
    std::vector<float> tensorsData{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    std::vector<size_t> shape{2, 3, 3};
    ov::Tensor embeddingsTensor = ov::Tensor(ov::element::Type_t::f32, shape, tensorsData.data());
    rapidjson::Document notUsed;
    ovms::EmbeddingsHandler handler(notUsed);
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
    ovms::EmbeddingsHandler handler(notUsed);
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
    ovms::EmbeddingsHandler handler(document);
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
    ovms::EmbeddingsHandler handler(notUsed);
    handler.setPromptTokensUsage(50);
    auto status = handler.parseResponse(buffer, embeddingsTensor, normalieEmbeddings);
    ASSERT_TRUE(status.ok());
    std::string expectedResponse = R"({"object":"list","data":[{"object":"embedding","embedding":[1.0,2.0,3.0],"index":0},{"object":"embedding","embedding":[1.0,2.0,3.0],"index":1}],"usage":{"prompt_tokens":50,"total_tokens":50}})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}

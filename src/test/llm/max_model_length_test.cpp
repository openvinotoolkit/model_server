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
#include <rapidjson/document.h>

#include <openvino/genai/tokenizer.hpp>

#include "../../llm/servable_initializer.hpp"
#include "../test_utils.hpp"

using namespace ovms;

class MaxModelLengthTest : public TestWithTempDir {
protected:
    std::string configFilePath;
    rapidjson::Document doc;
    ov::genai::Tokenizer dummyTokenizer;

    void SetUp() {
        TestWithTempDir::SetUp();
        configFilePath = directoryPath + "/config.json";
    }
};

TEST_F(MaxModelLengthTest, maxModelLength_MaxPositionEmbeddings_VALID) {
    std::string modelConfigContent = R"({"max_position_embeddings" : 5})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    ASSERT_TRUE(maxModelLength.has_value());
    EXPECT_EQ(maxModelLength.value(), 5);
}

TEST_F(MaxModelLengthTest, maxModelLength_MaxPositionEmbeddings_INVALID) {
    std::string modelConfigContent = R"({"max_position_embeddings" : "INVALID"})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    EXPECT_FALSE(maxModelLength.has_value());
}

TEST_F(MaxModelLengthTest, maxModelLength_nPositions_VALID) {
    std::string modelConfigContent = R"({"n_positions" : 5})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    ASSERT_TRUE(maxModelLength.has_value());
    EXPECT_EQ(maxModelLength.value(), 5);
}

TEST_F(MaxModelLengthTest, maxModelLength_nPositions_INVALID) {
    std::string modelConfigContent = R"({"n_positions" : "INVALID"})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    EXPECT_FALSE(maxModelLength.has_value());
}

TEST_F(MaxModelLengthTest, maxModelLength_seqLen_VALID) {
    std::string modelConfigContent = R"({"seq_len" : 5})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    ASSERT_TRUE(maxModelLength.has_value());
    EXPECT_EQ(maxModelLength.value(), 5);
}

TEST_F(MaxModelLengthTest, maxModelLength_seqLen_INVALID) {
    std::string modelConfigContent = R"({"seq_len" : "INVALID"})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    EXPECT_FALSE(maxModelLength.has_value());
}

TEST_F(MaxModelLengthTest, maxModelLength_seqLength_VALID) {
    std::string modelConfigContent = R"({"seq_length" : 5})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    ASSERT_TRUE(maxModelLength.has_value());
    EXPECT_EQ(maxModelLength.value(), 5);
}

TEST_F(MaxModelLengthTest, maxModelLength_seqLength_INVALID) {
    std::string modelConfigContent = R"({"seq_length" : "INVALID"})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    EXPECT_FALSE(maxModelLength.has_value());
}

TEST_F(MaxModelLengthTest, maxModelLength_nCtx_VALID) {
    std::string modelConfigContent = R"({"n_ctx" : 5})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    ASSERT_TRUE(maxModelLength.has_value());
    EXPECT_EQ(maxModelLength.value(), 5);
}

TEST_F(MaxModelLengthTest, maxModelLength_nCtx_INVALID) {
    std::string modelConfigContent = R"({"n_ctx" : "INVALID"})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    EXPECT_FALSE(maxModelLength.has_value());
}

TEST_F(MaxModelLengthTest, maxModelLength_slidingWindow_VALID) {
    std::string modelConfigContent = R"({"sliding_window" : 5})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    ASSERT_TRUE(maxModelLength.has_value());
    EXPECT_EQ(maxModelLength.value(), 5);
}

TEST_F(MaxModelLengthTest, maxModelLength_slidingWindow_INVALID) {
    std::string modelConfigContent = R"({"sliding_window" : "INVALID"})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    EXPECT_FALSE(maxModelLength.has_value());
}

TEST_F(MaxModelLengthTest, maxModelLength_emptyConfig) {
    std::string modelConfigContent = R"({})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    EXPECT_FALSE(maxModelLength.has_value());
}

TEST_F(MaxModelLengthTest, maxModelLength_parsingOrder) {
    std::string modelConfigContent = R"({"max_position_embeddings" : 5, "seq_length" : 6, "n_positions" : 7, "sliding_window" : 8, "seq_len" : 9, "n_ctx" : 10})";
    createConfigFileWithContent(modelConfigContent, configFilePath);
    auto maxModelLength = parseMaxModelLength(directoryPath);
    ASSERT_TRUE(maxModelLength.has_value());
    EXPECT_EQ(maxModelLength.value(), 8);
}

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

#include "../llm/llmnoderesources.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

class TextStreamerTest : public ::testing::Test {
public:
    ::mediapipe::CalculatorGraphConfig config;
    std::shared_ptr<ovms::LLMNodeResources> nodeResources = nullptr;
    std::shared_ptr<Tokenizer> tokenizer;
    std::shared_ptr<ovms::TextStreamer> streamer;
    std::string testPbtxt = R"(
    node: {
    name: "llmNode"
    calculator: "HttpLLMCalculator"
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            models_path: "/ovms/llm_testing/facebook/opt-125m"
        }
    }
    }
)";

    void SetUp() override {
        py::initialize_interpreter();
        ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(testPbtxt, &config));
        ASSERT_EQ(ovms::LLMNodeResources::createLLMNodeResources(nodeResources, config.node(0), ""), ovms::StatusCode::OK);
        tokenizer = std::make_shared<Tokenizer>("/ovms/llm_testing/facebook/opt-125m");
        streamer = std::make_shared<ovms::TextStreamer>(tokenizer);
    }
    void TearDown() {
        nodeResources.reset();
        py::finalize_interpreter();
    }
};

TEST_F(TextStreamerTest, noValueReturned) {
    auto tokens = tokenizer->encode("TEST");
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::optional<std::string> rval = this->streamer->put(static_cast<int64_t*>(tokens.data())[i]);
        EXPECT_FALSE(rval.has_value());
    }
}

TEST_F(TextStreamerTest, valueReturned) {
    std::string testPrompt = "TEST\n";
    auto tokens = tokenizer->encode(testPrompt);
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::optional<std::string> rval = this->streamer->put(static_cast<int64_t*>(tokens.data())[i]);
        if (i < tokens.get_size() - 1) {
            EXPECT_FALSE(rval.has_value());
        } else {
            EXPECT_TRUE(rval.has_value());
            EXPECT_EQ(rval.value().compare(testPrompt), 0);
        }
    }
}

TEST_F(TextStreamerTest, valueReturnedCacheNotCleared) {
    std::string testPrompt1 = "TEST";
    auto tokens = tokenizer->encode(testPrompt1);
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::optional<std::string> rval = this->streamer->put(static_cast<int64_t*>(tokens.data())[i]);
        EXPECT_FALSE(rval.has_value());
    }
    std::string testPrompt2 = "TEST\n";
    tokens = tokenizer->encode(testPrompt2);
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::optional<std::string> rval = this->streamer->put(static_cast<int64_t*>(tokens.data())[i]);
        if (i < tokens.get_size() - 1) {
            EXPECT_FALSE(rval.has_value());
        } else {
            EXPECT_TRUE(rval.has_value());
            EXPECT_EQ(rval.value().compare(testPrompt1 + testPrompt2), 0);
        }
    }
}

TEST_F(TextStreamerTest, valueReturnedCacheCleared) {
    std::string testPrompt = "TEST\n";
    auto tokens = tokenizer->encode(testPrompt);
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::optional<std::string> rval = this->streamer->put(static_cast<int64_t*>(tokens.data())[i]);
        if (i < tokens.get_size() - 1) {
            EXPECT_FALSE(rval.has_value());
        } else {
            EXPECT_TRUE(rval.has_value());
            EXPECT_EQ(rval.value().compare(testPrompt), 0);
        }
    }
    tokens = tokenizer->encode(testPrompt);
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::optional<std::string> rval = this->streamer->put(static_cast<int64_t*>(tokens.data())[i]);
        if (i < tokens.get_size() - 1) {
            EXPECT_FALSE(rval.has_value());
        } else {
            EXPECT_TRUE(rval.has_value());
            EXPECT_EQ(rval.value().compare(testPrompt), 0);
        }
    }
}

TEST_F(TextStreamerTest, valueReturnedTextWithSpaces) {
    std::string testPrompt = "TEST TEST TEST TEST";
    auto tokens = tokenizer->encode(testPrompt);
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::optional<std::string> rval = this->streamer->put(static_cast<int64_t*>(tokens.data())[i]);
        size_t numberOfTokensBeforeValueReturned = 2;
        if (i < numberOfTokensBeforeValueReturned) {
            EXPECT_FALSE(rval.has_value());
        } else {
            EXPECT_TRUE(rval.has_value());
            EXPECT_EQ(rval.value().compare("TEST "), 0);
        }
    }
}

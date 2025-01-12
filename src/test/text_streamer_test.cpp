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

#include "../llm/llm_executor.hpp"
#include "../llm/llmnoderesources.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

class TextStreamerTest : public ::testing::Test {
public:
    static inline ::mediapipe::CalculatorGraphConfig config;
    static inline std::shared_ptr<ovms::LLMNodeResources> nodeResources = std::make_shared<ovms::LLMNodeResources>();
    static inline std::shared_ptr<ov::genai::Tokenizer> tokenizer;
    static inline std::shared_ptr<ovms::TextStreamer> streamer;
    static inline const std::string testPbtxt = R"(
    node: {
    name: "llmNode"
    calculator: "HttpLLMCalculator"
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            models_path: "/ovms/src/test/llm_testing/facebook/opt-125m"
        }
    }
    }
)";

    static void SetUpTestSuite() {
        py::initialize_interpreter();
        std::string adjustedPbtxt = testPbtxt;
        adjustConfigForTargetPlatform(adjustedPbtxt);
        ASSERT_TRUE(::google::protobuf::TextFormat::ParseFromString(adjustedPbtxt, &config));
        ASSERT_EQ(ovms::LLMNodeResources::initializeLLMNodeResources(*nodeResources, config.node(0), ""), ovms::StatusCode::OK);
        tokenizer = std::make_shared<ov::genai::Tokenizer>(getGenericFullPathForSrcTest("/ovms/src/test/llm_testing/facebook/opt-125m"));
        streamer = std::make_shared<ovms::TextStreamer>(tokenizer);
    }
    static void TearDownTestSuite() {
        nodeResources.reset();
        py::finalize_interpreter();
    }
    void assertTokensValues(ov::Tensor generatedTokens, std::vector<int64_t> expectedTokens) {
        ASSERT_EQ(generatedTokens.get_size(), expectedTokens.size());
        for (size_t i = 0; i < generatedTokens.get_size(); i++) {
            ASSERT_EQ(static_cast<int64_t*>(generatedTokens.data())[i], expectedTokens[i]);
        }
    }
};

TEST_F(TextStreamerTest, noValueReturnedStringWithoutNewLineOrSpace) {
    std::string testPrompt = "TEST";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::vector<int64_t> singleTokenVector = {tokens.data<int64_t>()[i]};
        std::optional<std::string> partialResponseText = this->streamer->put(singleTokenVector);
        EXPECT_FALSE(partialResponseText.has_value());
    }
    std::string endOfMessage = this->streamer->end();
    ASSERT_EQ(endOfMessage.compare("TEST"), 0);
}

TEST_F(TextStreamerTest, putReturnsValue) {
    std::string testPrompt = "TEST\n";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923, 50118});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::vector<int64_t> singleTokenVector = {tokens.data<int64_t>()[i]};
        std::optional<std::string> partialResponseText = this->streamer->put(singleTokenVector);
        if (i < tokens.get_size() - 1) {  // No value returned until last token with new line passed to tokenizer
            EXPECT_FALSE(partialResponseText.has_value());
        } else {
            EXPECT_TRUE(partialResponseText.has_value());
            EXPECT_EQ(partialResponseText.value().compare(testPrompt), 0);
        }
    }
}

TEST_F(TextStreamerTest, putDoesNotReturnValueUntilNewLineDetected) {
    std::string testPrompt1 = "TEST";
    auto tokens = tokenizer->encode(testPrompt1, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::vector<int64_t> singleTokenVector = {tokens.data<int64_t>()[i]};
        std::optional<std::string> partialResponseText = this->streamer->put(singleTokenVector);
        EXPECT_FALSE(partialResponseText.has_value());
    }
    std::string testPrompt2 = "TEST\n";
    tokens = tokenizer->encode(testPrompt2, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923, 50118});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::vector<int64_t> singleTokenVector = {tokens.data<int64_t>()[i]};
        std::optional<std::string> partialResponseText = this->streamer->put(singleTokenVector);
        if (i < tokens.get_size() - 1) {  // No value returned until last token with new line passed to tokenizer
            EXPECT_FALSE(partialResponseText.has_value());
        } else {
            EXPECT_TRUE(partialResponseText.has_value());
            EXPECT_EQ(partialResponseText.value().compare(testPrompt1 + testPrompt2), 0);
        }
    }
}

TEST_F(TextStreamerTest, valueReturnedCacheCleared) {
    std::string testPrompt = "TEST\n";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923, 50118});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::vector<int64_t> singleTokenVector = {tokens.data<int64_t>()[i]};
        std::optional<std::string> partialResponseText = this->streamer->put(singleTokenVector);
        if (i < tokens.get_size() - 1) {  // No value returned until last token with new line passed to tokenizer
            EXPECT_FALSE(partialResponseText.has_value());
        } else {
            EXPECT_TRUE(partialResponseText.has_value());
            EXPECT_EQ(partialResponseText.value().compare(testPrompt), 0);
        }
    }
    tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::vector<int64_t> singleTokenVector = {tokens.data<int64_t>()[i]};
        std::optional<std::string> partialResponseText = this->streamer->put(singleTokenVector);
        if (i < tokens.get_size() - 1) {  // No value returned until last token with new line passed to tokenizer
            EXPECT_FALSE(partialResponseText.has_value());
        } else {
            EXPECT_TRUE(partialResponseText.has_value());
            EXPECT_EQ(partialResponseText.value().compare(testPrompt), 0);
        }
    }
}

TEST_F(TextStreamerTest, putReturnsValueTextWithSpaces) {
    std::string testPrompt = "TEST TEST TEST TEST";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> expectedTokens = {565, 4923, 41759, 41759, 41759};
    assertTokensValues(tokens, expectedTokens);
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::vector<int64_t> singleTokenVector = {tokens.data<int64_t>()[i]};
        std::optional<std::string> partialResponseText = this->streamer->put(singleTokenVector);
        size_t numberOfTokensBeforeValueReturned = 2;  // No value returned until third token with space passed to tokenizer
        if (i < numberOfTokensBeforeValueReturned) {
            EXPECT_FALSE(partialResponseText.has_value());
        } else {
            std::cout << "\n"
                      << partialResponseText.value();
            EXPECT_TRUE(partialResponseText.has_value());
            EXPECT_EQ(partialResponseText.value().compare("TEST "), 0);
        }
    }
    std::string endOfMessage = this->streamer->end();
    ASSERT_EQ(endOfMessage.compare("TEST"), 0);
}

TEST_F(TextStreamerTest, putReturnsValueTextWithNewLineInTheMiddle) {
    std::string testPrompt = "TEST\nTEST";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> expectedTokens = {565, 4923, 50118, 565, 4923};
    assertTokensValues(tokens, expectedTokens);
    std::vector<int64_t> singleTokenVector = {tokens.data<int64_t>()[0]};
    std::optional<std::string> partialResponseText = this->streamer->put(singleTokenVector);
    EXPECT_FALSE(partialResponseText.has_value());
    singleTokenVector = {tokens.data<int64_t>()[1]};
    partialResponseText = this->streamer->put(singleTokenVector);
    EXPECT_FALSE(partialResponseText.has_value());
    singleTokenVector = {tokens.data<int64_t>()[2]};
    partialResponseText = this->streamer->put(singleTokenVector);
    EXPECT_TRUE(partialResponseText.has_value());
    EXPECT_EQ(partialResponseText.value().compare("TEST\n"), 0);
    singleTokenVector = {tokens.data<int64_t>()[3]};
    partialResponseText = this->streamer->put(singleTokenVector);
    EXPECT_FALSE(partialResponseText.has_value());
    singleTokenVector = {tokens.data<int64_t>()[4]};
    partialResponseText = this->streamer->put(singleTokenVector);
    EXPECT_FALSE(partialResponseText.has_value());
    std::string endOfMessage = this->streamer->end();
    ASSERT_EQ(endOfMessage.compare("TEST"), 0);
}

TEST_F(TextStreamerTest, putReturnsValueAfterEndCalled) {
    std::string testPrompt = "TEST";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::vector<int64_t> singleTokenVector = {tokens.data<int64_t>()[i]};
        std::optional<std::string> partialResponseText = this->streamer->put(singleTokenVector);
        EXPECT_FALSE(partialResponseText.has_value());
    }
    std::string endOfMessage = this->streamer->end();
    ASSERT_EQ(endOfMessage.compare("TEST"), 0);

    testPrompt = "TEST\n";
    tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923, 50118});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        std::vector<int64_t> singleTokenVector = {tokens.data<int64_t>()[i]};
        std::optional<std::string> partialResponseText = this->streamer->put(singleTokenVector);
        if (i < tokens.get_size() - 1) {  // No value returned until last token with new line passed to tokenizer
            EXPECT_FALSE(partialResponseText.has_value());
        } else {
            EXPECT_TRUE(partialResponseText.has_value());
            EXPECT_EQ(partialResponseText.value().compare(testPrompt), 0);
        }
    }
}

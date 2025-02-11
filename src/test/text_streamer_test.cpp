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
    static inline std::shared_ptr<ov::genai::TextCallbackStreamer> streamer;
    static inline std::string lastTextChunk;
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
        auto callback = [](std::string text) {
            lastTextChunk = text;
            return false;
        };
        streamer = std::make_shared<ov::genai::TextCallbackStreamer>(*tokenizer, callback);
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
        this->streamer->put(tokens.data<int64_t>()[i]);
        EXPECT_TRUE(lastTextChunk.empty());
    }
    this->streamer->end();
    ASSERT_EQ(lastTextChunk.compare("TEST"), 0);
}

TEST_F(TextStreamerTest, putReturnsValue) {
    std::string testPrompt = "TEST\n";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923, 50118});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        this->streamer->put(tokens.data<int64_t>()[i]);
        if (i < tokens.get_size() - 1) {  // No value returned until last token with new line passed to tokenizer
            EXPECT_TRUE(lastTextChunk.empty());
        } else {
            EXPECT_EQ(lastTextChunk.compare(testPrompt), 0);
        }
    }
}

TEST_F(TextStreamerTest, putDoesNotReturnValueUntilNewLineDetected) {
    std::string testPrompt1 = "TEST";
    auto tokens = tokenizer->encode(testPrompt1, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        this->streamer->put(tokens.data<int64_t>()[i]);
        EXPECT_TRUE(lastTextChunk.empty());
    }
    std::string testPrompt2 = "TEST\n";
    tokens = tokenizer->encode(testPrompt2, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923, 50118});
    // Next pushed token will be 3rd so callback shall return non-empty string
    std::vector<std::string> expectedValues = {
        "T",      // due to 3-token delay in text streamer, putting third token will return result of first token decoding
        "EST",    // as above, but for second token
        "TEST\n"  // last put token is new line, so streamer flushes the cache and callback returns all remaining tokens decoded
    };
    for (size_t i = 0; i < tokens.get_size(); i++) {
        this->streamer->put(tokens.data<int64_t>()[i]);
        EXPECT_EQ(lastTextChunk.compare(expectedValues[i]), 0);
    }
}

TEST_F(TextStreamerTest, valueReturnedCacheCleared) {
    std::string testPrompt = "TEST\n";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923, 50118});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        this->streamer->put(tokens.data<int64_t>()[i]);
        if (i < tokens.get_size() - 1) {  // No value returned until last token with new line passed to tokenizer
            EXPECT_TRUE(lastTextChunk.empty());
        } else {
            EXPECT_EQ(lastTextChunk.compare(testPrompt), 0);
        }
    }
    tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    for (size_t i = 0; i < tokens.get_size(); i++) {
        this->streamer->put(tokens.data<int64_t>()[i]);
        if (i < tokens.get_size() - 1) {  // No value returned until last token with new line passed to tokenizer
            EXPECT_TRUE(lastTextChunk.empty());
        } else {
            EXPECT_EQ(lastTextChunk.compare(testPrompt), 0);
        }
    }
}

TEST_F(TextStreamerTest, putReturnsValueTextWithSpaces) {
    std::string testPrompt = "TEST TEST TEST TEST";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> expectedTokens = {565, 4923, 41759, 41759, 41759};
    std::vector<std::string> callbackExpectedValues = {"", "", "T", "EST", " TEST"};
    assertTokensValues(tokens, expectedTokens);
    for (size_t i = 0; i < tokens.get_size(); i++) {
        this->streamer->put(tokens.data<int64_t>()[i]);
        EXPECT_EQ(lastTextChunk.compare(callbackExpectedValues[i]), 0);
    }
    this->streamer->end();
    ASSERT_EQ(lastTextChunk.compare(" TEST TEST"), 0);
}

TEST_F(TextStreamerTest, putReturnsValueTextWithNewLineInTheMiddle) {
    std::string testPrompt = "TEST\nTEST";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    std::vector<int64_t> expectedTokens = {565, 4923, 50118, 565, 4923};
    assertTokensValues(tokens, expectedTokens);
    for (size_t i = 0; i < tokens.get_size(); i++) {
        this->streamer->put(tokens.data<int64_t>()[i]);
        if (i == 2) {
            EXPECT_EQ(lastTextChunk.compare("TEST\n"), 0);
        } else {
            EXPECT_TRUE(lastTextChunk.empty());
        }
    }
    this->streamer->end();
    ASSERT_EQ(lastTextChunk.compare("TEST"), 0);
}

TEST_F(TextStreamerTest, putReturnsValueAfterEndCalled) {
    std::string testPrompt = "TEST";
    auto tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        this->streamer->put(tokens.data<int64_t>()[i]);
        EXPECT_TRUE(lastTextChunk.empty());
    }
    this->streamer->end();
    ASSERT_EQ(lastTextChunk.compare("TEST"), 0);

    testPrompt = "TEST\n";
    tokens = tokenizer->encode(testPrompt, ov::genai::add_special_tokens(false)).input_ids;
    assertTokensValues(tokens, {565, 4923, 50118});
    for (size_t i = 0; i < tokens.get_size(); i++) {
        this->streamer->put(tokens.data<int64_t>()[i]);
        if (i < tokens.get_size() - 1) {  // No value returned until last token with new line passed to tokenizer
            EXPECT_TRUE(lastTextChunk.empty());
        } else {
            EXPECT_EQ(lastTextChunk.compare(testPrompt), 0);
        }
    }
}

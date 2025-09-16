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
#include <gtest/gtest.h>
#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>

#include "../../../llm/io_processing/base_output_parser.hpp"
#include "../../../llm/io_processing/output_parser.hpp"
#include "../../../llm/io_processing/openai/harmony.hpp"
#include "../../platform_utils.hpp"

using namespace ovms;
using namespace ovms::openai;

#ifdef _WIN32
const std::string tokenizerPath = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\openai\\gpt-oss-20b";
#else
// Hardcoded for usage in docker container
const std::string tokenizerPath = "/ovms/src/test/llm_testing/openai/gpt-oss-20b";
#endif

static std::unique_ptr<ov::genai::Tokenizer> gptOssTokenizer;

static std::vector<int64_t> getTokens(const std::string& text) {
    ov::Tensor t = gptOssTokenizer->encode(text).input_ids;
    const int64_t* data_ptr = t.data<const int64_t>();
    size_t length = t.get_shape()[1];  // assuming shape is [1, seq_len]
    return std::vector<int64_t>(data_ptr, data_ptr + length);
}

class TokenBuilder {
public:
    TokenBuilder& add(const std::string& text) {
        auto tokens = getTokens(text);
        tokenIDs.insert(tokenIDs.end(), tokens.begin(), tokens.end());
        return *this;
    }

    TokenBuilder& add(int64_t tokenID) {
        tokenIDs.push_back(tokenID);
        return *this;
    }

    TokenBuilder& add(Harmony::TokenID tokenID) {
        tokenIDs.push_back(static_cast<int64_t>(tokenID));
        return *this;
    }

    std::vector<int64_t> build() {
        return tokenIDs;
    }

    TokenBuilder& clear() {
        tokenIDs.clear();
        return *this;
    }

private:
    std::vector<int64_t> tokenIDs;
};

class GptOssOutputParserTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            gptOssTokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPath, ov::AnyMap{ov::genai::skip_special_tokens(false)});
        } catch (const std::exception& e) {
            FAIL() << "Failed to initialize gptOss tokenizer: " << e.what();
        } catch (...) {
            FAIL() << "Failed to initialize gptOss tokenizer due to unknown error.";
        }
    }

protected:
    void pushTokens(const std::string& text, std::vector<int64_t>& tt) {
        auto tokens = getTokens(text);
        tt.insert(tt.end(), tokens.begin(), tokens.end());
    }

    static void TearDownTestSuite() {
        gptOssTokenizer.reset();
    }

    void SetUp() override {
    }

    TokenBuilder builder;
};

TEST_F(GptOssOutputParserTest, SimpleContent) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
        Harmony::TokenID::RETURN,  // <|channel|>final<|message|>Hello, world!<|return|>
        Harmony::TokenID::END,  // <|channel|>final<|message|>Hello, world!<|end|>
        Harmony::TokenID::CALL}) {  // <|channel|>final<|message|>Hello, world!<|call|>
        builder
            .clear()
            .add(Harmony::TokenID::CHANNEL)
            .add("final")
            .add(Harmony::TokenID::MESSAGE)
            .add("Hello, world!")
            .add(closureToken);

        Harmony harmony(*gptOssTokenizer, builder.build());

        ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getContent(), "Hello, world!") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
    }
}

TEST_F(GptOssOutputParserTest, NegativeFinalChannel) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
        Harmony::TokenID::RETURN,  // <|channel|>?<|message|>Hello, world!<|return|>
        Harmony::TokenID::END,  // <|channel|>?<|message|>Hello, world!<|end|>
        Harmony::TokenID::CALL}) {  // <|channel|>?<|message|>Hello, world!<|call|>

        for (auto wrongChannel : std::vector<std::string>{
                 "finalextra",  // finalextra is not final
                 "Final",      // case sensitive
                 " finale",    // leading space
                 "final ",     // trailing space
                 " final",     // trailing space
                 "fi nal",     // space inside
                 ""}) {        // empty channel

            builder
                .clear()
                .add(Harmony::TokenID::CHANNEL)
                .add(wrongChannel)
                .add(Harmony::TokenID::MESSAGE)
                .add("Hello, world!")
                .add(closureToken);

            Harmony harmony(*gptOssTokenizer, builder.build());

            // TODO: Fail such responses completely instead of ignoring them?
            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getContent(), "2") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
        }
    }
}

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
        Harmony::TokenID::RETURN,  // <|channel|>WRONG<|message|>Hello, world!<|return|>
        Harmony::TokenID::END,  // <|channel|>WRONG<|message|>Hello, world!<|end|>
        Harmony::TokenID::CALL}) {  // <|channel|>WRONG<|message|>Hello, world!<|call|>

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
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
        }
    }
}

TEST_F(GptOssOutputParserTest, PreambleOnly) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
        Harmony::TokenID::RETURN,  // <|channel|>commentary<|message|>Hello, world!<|return|>
        Harmony::TokenID::END,  // <|channel|>commentary<|message|>Hello, world!<|end|>
        Harmony::TokenID::CALL}) {  // <|channel|>commentary<|message|>Hello, world!<|call|>
        builder
            .clear()
            .add(Harmony::TokenID::CHANNEL)
            .add("commentary")
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

TEST_F(GptOssOutputParserTest, NegativePreamble) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
        Harmony::TokenID::RETURN,  // <|channel|>WRONG PREAMBLE<|message|>Hello, world!<|return|>
        Harmony::TokenID::END,  // <|channel|>WRONG PREAMBLE<|message|>Hello, world!<|end|>
        Harmony::TokenID::CALL}) {  // <|channel|>WRONG PREAMBLE<|message|>Hello, world!<|call|>

        for (auto wrongChannel : std::vector<std::string>{
                 "commentary ", 
                 " commentary", 
                 " commentary ",
                 "comment ary",  // space inside
                 "commenTary",   // case sensitive
                 ""}) {

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
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
        }
    }
}

TEST_F(GptOssOutputParserTest, ReasoningOnly) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
        Harmony::TokenID::RETURN,  // <|channel|>commentary<|message|>Hello, world!<|return|>
        Harmony::TokenID::END,  // <|channel|>commentary<|message|>Hello, world!<|end|>
        Harmony::TokenID::CALL}) {  // <|channel|>commentary<|message|>Hello, world!<|call|>
        builder
            .clear()
            .add(Harmony::TokenID::CHANNEL)
            .add("analysis")
            .add(Harmony::TokenID::MESSAGE)
            .add("Hello, world!")
            .add(closureToken);

        Harmony harmony(*gptOssTokenizer, builder.build());

        ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getReasoning(), "Hello, world!") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
    }
}

TEST_F(GptOssOutputParserTest, NegativeReasoning) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
        Harmony::TokenID::RETURN,  // <|channel|>WRONG REASONING<|message|>Hello, world!<|return|>
        Harmony::TokenID::END,  // <|channel|>WRONG REASONING<|message|>Hello, world!<|end|>
        Harmony::TokenID::CALL}) {  // <|channel|>WRONG REASONING<|message|>Hello, world!<|call|>

        for (auto wrongChannel : std::vector<std::string>{
                 "analysis ",
                 " analysis ",
                 "analy sis",  // space inside
                 "analYsis",   // case sensitive
                 ""}) {

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
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
        }
    }
}

TEST_F(GptOssOutputParserTest, SingleToolCallWithConstrain) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
        Harmony::TokenID::RETURN,  // <|channel|>commentary to=functions.hello<|constrain|>json<|message|>{"Hello": "world!"}<|return|>
        Harmony::TokenID::END,  // <|channel|>commentary to=functions.hello<|constrain|>json<|message|>{"Hello": "world!"}<|end|>
        Harmony::TokenID::CALL}) {  // <|channel|>commentary to=functions.hello<|constrain|>json<|message|>{"Hello": "world!"}<|call|>
        for (auto functionDeclaration : std::vector<std::string>{
                "commentary to=functions.hello",  // valid channel with to=
                "commentary to=functions.hello ",
                "commentary   to=functions.hello",
                "commentary  ANYTHING IN BETWEEN to=functions.hello",
            }) { // spaces after hello
            builder
                .clear()
                .add(Harmony::TokenID::CHANNEL)
                .add(functionDeclaration)
                .add(Harmony::TokenID::MESSAGE)
                .add(R"({"Hello": "world!"})")
                .add(closureToken);

            Harmony harmony(*gptOssTokenizer, builder.build());

            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls().size(), 1) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls()[0].name, "hello") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls()[0].arguments, R"({"Hello": "world!"})") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
        }
    }
}

TEST_F(GptOssOutputParserTest, InvalidSingleToolCallWithConstrain) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
        Harmony::TokenID::RETURN,  // <|channel|>commentary to=functions.hello<|constrain|>json<|message|>{"Hello": "world!"}<|return|>
        Harmony::TokenID::END,  // <|channel|>commentary to=functions.hello<|constrain|>json<|message|>{"Hello": "world!"}<|end|>
        Harmony::TokenID::CALL}) {  // <|channel|>commentary to=functions.hello<|constrain|>json<|message|>{"Hello": "world!"}<|call|>
        for (auto functionDeclaration : std::vector<std::string>{
                "commentary to = functions.hello",
                "commentary to= functions.hello ",
                "commentary functions.hello",
                "commentary to=hello",
                "commentary hello"
            }) {
            builder
                .clear()
                .add(Harmony::TokenID::CHANNEL)
                .add(functionDeclaration)
                .add(Harmony::TokenID::MESSAGE)
                .add(R"({"Hello": "world!"})")
                .add(closureToken);

            Harmony harmony(*gptOssTokenizer, builder.build());

            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
            ASSERT_EQ(harmony.getToolCalls().size(), 0) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " function declaration: " << functionDeclaration;
        }
    }
}

TEST_F(GptOssOutputParserTest, HolisticMultiTurn) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
        Harmony::TokenID::RETURN,  // <|channel|>commentary to=functions.hello<|constrain|>json<|message|>{"Hello": "world!"}<|return|>
        Harmony::TokenID::END,  // <|channel|>commentary to=functions.hello<|constrain|>json<|message|>{"Hello": "world!"}<|end|>
        Harmony::TokenID::CALL}) {  // <|channel|>commentary to=functions.hello<|constrain|>json<|message|>{"Hello": "world!"}<|call|>

        // In regular scenarios it is never that complicated. But we test the parser, so why not.
        // Usually the order is as follows:
        // - Analysis (reasoning)
        // - (optional) commentary (preamble, counts as final content as well)
        // - (optional, multiple) commentary to=functions.* + constrain json (tool calls)
        // - final (content)
        builder
            .clear()
            .add(Harmony::TokenID::CHANNEL)
            .add("analysis")
            .add(Harmony::TokenID::MESSAGE)
            .add("I need to call a function.")
            .add(closureToken)

            // With constrain, but ignored anyway
            .add(Harmony::TokenID::CHANNEL)
            .add("commentary to=functions.hello")  // strict
            .add(Harmony::TokenID::CONSTRAIN)
            .add("json")
            .add(Harmony::TokenID::MESSAGE)
            .add(R"({"Hello": "world!"})")
            .add(closureToken)

            .add(Harmony::TokenID::CHANNEL)
            .add("final")
            .add(Harmony::TokenID::MESSAGE)
            .add("Dear User, I called function!")
            .add(closureToken)

            // Without constrain, it is ignored anyway
            .add(Harmony::TokenID::CHANNEL)
            .add("commentary ? to=functions.goodbye ")  // with space and anything in the middle
            .add(Harmony::TokenID::MESSAGE)
            .add("NOT A JSON")
            .add(closureToken)

            // Preamble
            .add(Harmony::TokenID::CHANNEL)
            .add("commentary")
            .add(Harmony::TokenID::MESSAGE)
            .add("I called some functions. Will summarize now.")
            .add(closureToken)

            // Final v2
            .add(Harmony::TokenID::CHANNEL)
            .add("final")
            .add(Harmony::TokenID::MESSAGE)
            .add("Dear User, I called second function!")
            .add(closureToken);

        Harmony harmony(*gptOssTokenizer, builder.build());

        ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getContent(), "Dear User, I called function! I called some functions. Will summarize now. Dear User, I called second function!") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getReasoning(), "I need to call a function.") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getToolCalls().size(), 2) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getToolCalls()[0].name, "hello") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getToolCalls()[0].arguments, R"({"Hello": "world!"})") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getToolCalls()[1].name, "goodbye") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getToolCalls()[1].arguments, "NOT A JSON") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
    }
}

// Negative
TEST_F(GptOssOutputParserTest, MissingChannel) {
    builder
        .clear()
        // .add(Harmony::TokenID::CHANNEL)  // no channel
        .add("commentary to=functions.hello")
        .add(Harmony::TokenID::MESSAGE)
        .add(R"({"Hello": "world!"})")
        .add(Harmony::TokenID::END);

    Harmony harmony(*gptOssTokenizer, builder.build());

    ASSERT_TRUE(harmony.parse());
    ASSERT_EQ(harmony.getContent(), "");
    ASSERT_EQ(harmony.getReasoning(), "");
    ASSERT_EQ(harmony.getToolCalls().size(), 0);
}

TEST_F(GptOssOutputParserTest, MissingMessageTag) {
    builder
        .clear()
        .add(Harmony::TokenID::CHANNEL)
        .add("commentary to=functions.hello")
        //  .add(Harmony::TokenID::MESSAGE)  // no message tag
        .add(R"({"Hello": "world!"})")
        .add(Harmony::TokenID::END);

    Harmony harmony(*gptOssTokenizer, builder.build());

    ASSERT_TRUE(harmony.parse());
    ASSERT_EQ(harmony.getContent(), "");
    ASSERT_EQ(harmony.getReasoning(), "");
    ASSERT_EQ(harmony.getToolCalls().size(), 0);
}

TEST_F(GptOssOutputParserTest, MissingEndTag) {
    builder
        .clear()
        .add(Harmony::TokenID::CHANNEL)
        .add("commentary to=functions.hello")
        .add(Harmony::TokenID::MESSAGE)
        .add(R"({"Hello": "world!"})");
        // .add(Harmony::TokenID::END);  // no end tag

    Harmony harmony(*gptOssTokenizer, builder.build());

    ASSERT_TRUE(harmony.parse());
    ASSERT_EQ(harmony.getContent(), "");
    ASSERT_EQ(harmony.getReasoning(), "");
    ASSERT_EQ(harmony.getToolCalls().size(), 0);
}

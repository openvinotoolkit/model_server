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
#include "../../../llm/io_processing/gptoss/harmony.hpp"
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

class GptOssOutputUnaryParserTest : public ::testing::Test {
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

    void assertParseIgnoredNoResults(Harmony& harmony) {
        ASSERT_TRUE(harmony.parse());
        ASSERT_EQ(harmony.getContent(), "");
        ASSERT_EQ(harmony.getReasoning(), "");
        ASSERT_EQ(harmony.getToolCalls().size(), 0);
    }

    TokenBuilder builder;
};

//
//
// Unary
//
//

TEST_F(GptOssOutputUnaryParserTest, SimpleContent) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        builder
            .clear()
            .add(Harmony::TokenID::CHANNEL)  // <|channel|>
            .add("final")
            .add(Harmony::TokenID::MESSAGE)  // <|message|>
            .add("Hello, world!")
            .add(closureToken);  // <|end|> or <|return|> or <|call|>
        Harmony harmony(*gptOssTokenizer, builder.build());
        ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getContent(), "Hello, world!") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken);
        ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken);
    }
}

TEST_F(GptOssOutputUnaryParserTest, NegativeFinalChannel) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        for (auto wrongChannel : std::vector<std::string>{
                 "finalextra",  // finalextra is not final
                 "Final",       // case sensitive
                 " finale",     // leading space
                 "final ",      // trailing space
                 " final",      // trailing space
                 "fi nal",      // space inside
                 ""}) {         // empty channel
            builder
                .clear()
                .add(Harmony::TokenID::CHANNEL)
                .add(wrongChannel)
                .add(Harmony::TokenID::MESSAGE)
                .add("Hello, world!")
                .add(closureToken);
            Harmony harmony(*gptOssTokenizer, builder.build());
            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
        }
    }
}

TEST_F(GptOssOutputUnaryParserTest, PreambleOnly) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
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

TEST_F(GptOssOutputUnaryParserTest, NegativePreamble) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
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
            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
        }
    }
}

TEST_F(GptOssOutputUnaryParserTest, ReasoningOnly) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
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

TEST_F(GptOssOutputUnaryParserTest, NegativeReasoning) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
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
            ASSERT_TRUE(harmony.parse()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getContent(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_EQ(harmony.getReasoning(), "") << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
            ASSERT_TRUE(harmony.getToolCalls().empty()) << "Failed for closure token: " << static_cast<int64_t>(closureToken) << " channel " << wrongChannel;
        }
    }
}

TEST_F(GptOssOutputUnaryParserTest, SingleToolCallWithConstrain) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        for (auto functionDeclaration : std::vector<std::string>{
                 "commentary to=functions.hello",  // valid channel with to=
                 "commentary to=functions.hello ",
                 "commentary   to=functions.hello",
                 "commentary  ANYTHING IN BETWEEN to=functions.hello",
             }) {  // spaces after hello
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

TEST_F(GptOssOutputUnaryParserTest, InvalidSingleToolCallWithConstrain) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
        for (auto functionDeclaration : std::vector<std::string>{
                 "commentary to = functions.hello",
                 "commentary to= functions.hello ",
                 "commentary functions.hello",
                 "commentary to=hello",
                 "commentary hello"}) {
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

TEST_F(GptOssOutputUnaryParserTest, HolisticMultiTurn) {
    for (auto closureToken : std::vector<Harmony::TokenID>{
             Harmony::TokenID::RETURN,   // ending with <|return|>
             Harmony::TokenID::END,      // ending with <|end|>
             Harmony::TokenID::CALL}) {  // ending with <|call|>
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
TEST_F(GptOssOutputUnaryParserTest, MissingChannel) {
    builder
        .clear()
        // .add(Harmony::TokenID::CHANNEL)  // no channel
        .add("commentary to=functions.hello")
        .add(Harmony::TokenID::MESSAGE)
        .add(R"({"Hello": "world!"})")
        .add(Harmony::TokenID::END);
    Harmony harmony(*gptOssTokenizer, builder.build());
    assertParseIgnoredNoResults(harmony);
}

TEST_F(GptOssOutputUnaryParserTest, MissingMessageTag) {
    builder
        .clear()
        .add(Harmony::TokenID::CHANNEL)
        .add("commentary to=functions.hello")
        //  .add(Harmony::TokenID::MESSAGE)  // no message tag
        .add(R"({"Hello": "world!"})")
        .add(Harmony::TokenID::END);
    Harmony harmony(*gptOssTokenizer, builder.build());
    assertParseIgnoredNoResults(harmony);
}

TEST_F(GptOssOutputUnaryParserTest, MissingEndTag) {
    builder
        .clear()
        .add(Harmony::TokenID::CHANNEL)
        .add("commentary to=functions.hello")
        .add(Harmony::TokenID::MESSAGE)
        .add(R"({"Hello": "world!"})");
    // .add(Harmony::TokenID::END);  // no end tag
    Harmony harmony(*gptOssTokenizer, builder.build());
    assertParseIgnoredNoResults(harmony);
}

//
//
// Streaming
//
//

class GptOssOutputStreamParserTest : public GptOssOutputUnaryParserTest {
protected:
    std::unique_ptr<OutputParser> outputParser;

    void SetUp() override {
        GptOssOutputUnaryParserTest::SetUp();
        outputParser = std::make_unique<OutputParser>(*gptOssTokenizer, "gptoss", "gptoss");
    }

    void test(const std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>>& chunkToDeltaVec) {
        // Need to have new output parser per case to simulate separate request processing
        outputParser = std::make_unique<OutputParser>(*gptOssTokenizer, "gptoss", "gptoss");
        auto chunkToDeltaVecCopy = chunkToDeltaVec;
        int64_t chunkIteration = -1;
        for (const auto& [chunk, finishReason, expectedDelta] : chunkToDeltaVecCopy) {
            chunkIteration++;
            std::optional<rapidjson::Document> doc = outputParser->parseChunk(chunk, true, finishReason);
            if (!expectedDelta.has_value() && !doc.has_value()) {
                continue;  // Both are nullopt, OK
            }
            if (expectedDelta.has_value() && doc.has_value()) {
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                doc->Accept(writer);
                std::string docStr = buffer.GetString();
                // If both strings contain "id":"...", compare id values by length and alphanumeric, else compare whole strings
                std::string expected = expectedDelta.value();
                std::string idKey = "\"id\":\"";
                auto docIdPos = docStr.find(idKey);
                auto expectedIdPos = expected.find(idKey);
                if (docIdPos != std::string::npos && expectedIdPos != std::string::npos) {
                    auto docIdStart = docIdPos + idKey.size();
                    auto docIdEnd = docStr.find("\"", docIdStart);
                    auto expectedIdStart = expectedIdPos + idKey.size();
                    auto expectedIdEnd = expected.find("\"", expectedIdStart);
                    ASSERT_NE(docIdEnd, std::string::npos);
                    ASSERT_NE(expectedIdEnd, std::string::npos);
                    std::string docId = docStr.substr(docIdStart, docIdEnd - docIdStart);
                    std::string expectedId = expected.substr(expectedIdStart, expectedIdEnd - expectedIdStart);
                    EXPECT_EQ(docId.size(), expectedId.size()) << "ID length mismatch for chunk: " << chunk;
                    EXPECT_TRUE(std::all_of(docId.begin(), docId.end(), ::isalnum)) << "ID not alphanumeric for chunk: " << chunk;
                    // Compare everything except the id value
                    std::string docStrNoId = docStr;
                    std::string expectedNoId = expected;
                    docStrNoId.replace(docIdStart, docId.size(), std::string(docId.size(), '*'));
                    expectedNoId.replace(expectedIdStart, expectedId.size(), std::string(expectedId.size(), '*'));
                    EXPECT_EQ(docStrNoId, expectedNoId) << "Mismatch for chunk (ignoring id value): " << chunk;
                } else {
                    EXPECT_EQ(docStr, expected) << "Mismatch for chunk: [" << chunk << "] got [" << docStr << "] but expected [" << expected << "]" << chunkIteration;
                }
            } else if (expectedDelta.has_value()) {
                FAIL() << "Mismatch for chunk: [" << chunk << "] got nothing but expected [" << expectedDelta.value() << "]" << chunkIteration;
            } else if (doc.has_value()) {
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                doc->Accept(writer);
                std::string docStr = buffer.GetString();
                FAIL() << "Mismatch for chunk: [" << chunk << "] expected nothing but got [" << docStr << "]" << chunkIteration;
            } else {
                FAIL() << "Mismatch for chunk: [" << chunk << "] " << chunkIteration;
            }
        }
    }
};

TEST_F(GptOssOutputStreamParserTest, HolisticStreamingReasoning) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Reasoning
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"analysis", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"I"}})"}},
        {" am", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" am"}})"}},
        {" reaso", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" reaso"}})"}},
        {"ning.", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"ning."}})"}},
        {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Preamble
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"I"}})"}},
        {" am", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" am"}})"}},
        {" producing", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" producing"}})"}},
        {" preamble", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" preamble"}})"}},
        {".", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"."}})"}},
        {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Final content
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"final", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"Dear", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"Dear"}})"}},
        {" User,", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" User,"}})"}},
        {" I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" I"}})"}},
        {" reason!", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" reason!"}})"}},
        {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
    };
    test(chunkToDeltaVec);
}

TEST_F(GptOssOutputStreamParserTest, HolisticStreamingTools) {
    std::vector<std::tuple<std::string, ov::genai::GenerationFinishReason, std::optional<std::string>>> chunkToDeltaVec{
        // Reasoning
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"analysis", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"I"}})"}},
        {" will", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" will"}})"}},
        {" call", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" call"}})"}},
        {" fun", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":" fun"}})"}},
        {"ction.", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"reasoning_content":"ction."}})"}},
        {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Preamble
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"I", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"I"}})"}},
        {" have", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" have"}})"}},
        {" to", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" to"}})"}},
        {" call", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" call"}})"}},
        {" fun", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":" fun"}})"}},
        {"ction.", ov::genai::GenerationFinishReason::NONE, {R"({"delta":{"content":"ction."}})"}},
        {"<|end|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Tool 1
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {".hello ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":0,\"function\":{\"name\":\"hello\"}}]}}"},
        {" {\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" {\""}}]}})"},
        {"location", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"location"}}]}})"},
        {"\":", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":"}}]}})"},
        {" \"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \""}}]}})"},
        {"Paris", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Paris"}}]}})"},
        {"\"}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}})"},
        {"<|call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        // Tool 2 (with ignored constrain)
        {"<|channel|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"commentary", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" to=", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"fun", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"ctions", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {".world ", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|constrain|>", ov::genai::GenerationFinishReason::NONE, "{\"delta\":{\"tool_calls\":[{\"id\":\"XXXXXXXXX\",\"type\":\"function\",\"index\":1,\"function\":{\"name\":\"world\"}}]}}"},
        {"json", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {"<|message|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
        {" {\"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":" {\""}}]}})"},
        {"location", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"location"}}]}})"},
        {"\":", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\":"}}]}})"},
        {" \"", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":" \""}}]}})"},
        {"Warsaw", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"Warsaw"}}]}})"},
        {"\"}", ov::genai::GenerationFinishReason::NONE, R"({"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"}"}}]}})"},
        {"<|call|>", ov::genai::GenerationFinishReason::NONE, std::nullopt},
    };
    test(chunkToDeltaVec);
}

// Copyright 2026 Intel Corporation
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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/tokenizer.hpp>

#include "src/llm/io_processing/default_content_parser.hpp"
#include "src/test/platform_utils.hpp"

using namespace ovms;

#ifdef _WIN32
const std::string tokenizerPathDCP = getWindowsRepoRootPath() + "\\src\\test\\llm_testing\\facebook\\opt-125m";
#else
const std::string tokenizerPathDCP = "/ovms/src/test/llm_testing/facebook/opt-125m";
#endif

class DefaultContentParserTest : public ::testing::Test {
protected:
    static std::unique_ptr<ov::genai::Tokenizer> tokenizer;

    static void SetUpTestSuite() {
        try {
            tokenizer = std::make_unique<ov::genai::Tokenizer>(tokenizerPathDCP);
        } catch (...) {
            tokenizer = nullptr;
        }
    }

    static void TearDownTestSuite() {
        tokenizer.reset();
    }

    // Helper: invoke parseChunk and return the content string, or nullopt if the parser returned nullopt.
    static std::optional<std::string> parseContent(DefaultContentParser& parser, const std::string& buf) {
        auto result = parser.parseChunk(buf, {}, ov::genai::GenerationFinishReason::NONE);
        if (!result.has_value())
            return std::nullopt;
        const auto* cd = std::get_if<ContentDelta>(&*result);
        EXPECT_NE(cd, nullptr) << "Expected ContentDelta";
        if (!cd)
            return std::nullopt;
        return cd->text;
    }
};

std::unique_ptr<ov::genai::Tokenizer> DefaultContentParserTest::tokenizer;

// ── No erase tags ──────────────────────────────────────────────────────────

TEST_F(DefaultContentParserTest, PassThrough_NoTags) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer);
    EXPECT_EQ(parseContent(parser, "hello world"), "hello world");
}

TEST_F(DefaultContentParserTest, EmptyBuffer_NoTags) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer);
    EXPECT_EQ(parseContent(parser, ""), "");
}

// ── Single erase tag ───────────────────────────────────────────────────────

TEST_F(DefaultContentParserTest, TagFullyPresent_Erased) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer, {"<|eot|>"});
    EXPECT_EQ(parseContent(parser, "<|eot|>"), "");
}

TEST_F(DefaultContentParserTest, TagInMiddle_SurroundingsKept) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer, {"<|eom|>"});
    EXPECT_EQ(parseContent(parser, "before<|eom|>after"), "beforeafter");
}

TEST_F(DefaultContentParserTest, TagAtEnd_Erased) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer, {"<|im_end|>"});
    EXPECT_EQ(parseContent(parser, "text<|im_end|>"), "text");
}

TEST_F(DefaultContentParserTest, TagAtStart_Erased) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer, {"<s>"});
    EXPECT_EQ(parseContent(parser, "<s>text"), "text");
}

TEST_F(DefaultContentParserTest, TagRepeated_AllInstancesErased) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer, {"<|eot|>"});
    EXPECT_EQ(parseContent(parser, "a<|eot|>b<|eot|>c"), "abc");
}

// ── Partial match → hold (return nullopt) ──────────────────────────────────

TEST_F(DefaultContentParserTest, PartialTag_Hold) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer, {"<|eot|>"});
    // Buffer ends with "<|eot" — suffix overlaps with prefix of "<|eot|>"
    EXPECT_EQ(parseContent(parser, "text.<|eot"), std::nullopt);
}

TEST_F(DefaultContentParserTest, SingleCharOverlap_Hold) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer, {"<|im_end|>"});
    // Buffer ends with "<" which is the first char of "<|im_end|>"
    EXPECT_EQ(parseContent(parser, "text<"), std::nullopt);
}

// ── Multiple erase tags ────────────────────────────────────────────────────

TEST_F(DefaultContentParserTest, MultipleTags_AllErased) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer, {"<s>", "<|im_end|>"});
    EXPECT_EQ(parseContent(parser, "<s>hello<|im_end|>"), "hello");
}

TEST_F(DefaultContentParserTest, MultipleTags_OnlyPresentOnesErased) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer, {"<s>", "<|im_end|>"});
    EXPECT_EQ(parseContent(parser, "<s>hello"), "hello");
}

// ── FOUND_COMPLETE takes priority over FOUND_INCOMPLETE ───────────────────

TEST_F(DefaultContentParserTest, CompleteWinsOverIncomplete_Emits) {
    if (!tokenizer)
        GTEST_SKIP();
    // tag1 = "<turn|>" is fully present; tag2 = "<|tool_response>" is partially overlapping.
    // Expected: content is emitted (not held), with tag1 erased.
    DefaultContentParser parser(*tokenizer, {"<turn|>", "<|tool_response>"});
    // Buffer: "text<turn|>end<" — "<turn|>" FOUND_COMPLETE, "<" is first char of "<|tool_response>" FOUND_INCOMPLETE
    auto result = parseContent(parser, "text<turn|>end<");
    ASSERT_TRUE(result.has_value()) << "Should emit, not hold: FOUND_COMPLETE beats FOUND_INCOMPLETE";
    EXPECT_EQ(*result, "textend<");
}

// ── Content parser ignores tokens parameter ────────────────────────────────

TEST_F(DefaultContentParserTest, NonEmptyTokensIgnored) {
    if (!tokenizer)
        GTEST_SKIP();
    DefaultContentParser parser(*tokenizer, {"<|eot|>"});
    auto result = parser.parseChunk("hello", {1, 2, 3}, ov::genai::GenerationFinishReason::NONE);
    ASSERT_TRUE(result.has_value());
    const auto* cd = std::get_if<ContentDelta>(&*result);
    ASSERT_NE(cd, nullptr);
    EXPECT_EQ(cd->text, "hello");
}

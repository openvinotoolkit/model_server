//*****************************************************************************
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
#include <string>

#include "src/llm/io_processing/utils.hpp"

using namespace ovms;

// ── basic hit / miss ─────────────────────────────────────────────────────────

TEST(FindInStringTest, FindsTargetOutsideAllQuoting) {
    EXPECT_EQ(findInStringRespectingSpecialChars("a, b, c", ",", 0), 1u);
}

TEST(FindInStringTest, ReturnsNposWhenNotFound) {
    EXPECT_EQ(findInStringRespectingSpecialChars("abc", ",", 0), std::string::npos);
}

TEST(FindInStringTest, RespectsStartPos) {
    EXPECT_EQ(findInStringRespectingSpecialChars("a, b, c", ",", 2), 4u);
}

// ── double-quote depth ───────────────────────────────────────────────────────

TEST(FindInStringTest, SkipsTargetInsideDoubleQuotes) {
    EXPECT_EQ(findInStringRespectingSpecialChars(R"("a, b", c)", ",", 0), 6u);
}

TEST(FindInStringTest, FindsTargetAfterClosingDoubleQuote) {
    EXPECT_EQ(findInStringRespectingSpecialChars(R"("abc", x)", ",", 0), 5u);
}

TEST(FindInStringTest, EscapedDoubleQuoteDoesNotCloseQuoting) {
    // \" inside a double-quoted string must not close it; comma is at 6.
    EXPECT_EQ(findInStringRespectingSpecialChars(R"("a\"b", c)", ",", 0), 6u);
}

TEST(FindInStringTest, SingleQuoteInsideDoubleQuoteIsIgnored) {
    // Single quotes inside "..." must not affect singleQuoteDepth; comma at 6.
    EXPECT_EQ(findInStringRespectingSpecialChars(R"("it's", b)", ",", 0), 6u);
}

// ── brace / bracket depth ────────────────────────────────────────────────────

TEST(FindInStringTest, SkipsTargetInsideBraces) {
    EXPECT_EQ(findInStringRespectingSpecialChars("{a, b}, c", ",", 0), 6u);
}

TEST(FindInStringTest, SkipsTargetInsideBrackets) {
    EXPECT_EQ(findInStringRespectingSpecialChars("[a, b], c", ",", 0), 6u);
}

TEST(FindInStringTest, SkipsNestedDepth) {
    // Comma inside inner [] still hidden; outer comma is the first one found.
    EXPECT_EQ(findInStringRespectingSpecialChars("[a, [b, c], d], e", ",", 0), 14u);
}

TEST(FindInStringTest, MixedBraceAndBracket) {
    EXPECT_EQ(findInStringRespectingSpecialChars("{[a, b]}, c", ",", 0), 8u);
}

// ── single-quote depth: opening ──────────────────────────────────────────────

TEST(FindInStringTest, SingleQuoteAtStringStartOpens) {
    // i == 0 → prevIsWord = false → quote opens; comma at 5.
    EXPECT_EQ(findInStringRespectingSpecialChars("'abc', 'def'", ",", 0), 5u);
}

TEST(FindInStringTest, SingleQuoteAfterNonWordOpens) {
    // Non-word char before the quote → opens; comma at 6 (starting at 1).
    EXPECT_EQ(findInStringRespectingSpecialChars("['abc', 'x']", ",", 1), 6u);
}

TEST(FindInStringTest, ApostropheInWordDoesNotOpen) {
    // Single quote between two word chars (it's) → not treated as opening quote.
    EXPECT_EQ(findInStringRespectingSpecialChars("it's a test, b", ",", 0), 11u);
}

// ── single-quote depth: closing ──────────────────────────────────────────────

TEST(FindInStringTest, SingleQuoteClosedBeforeDelimiter) {
    // Closing quote immediately followed by comma → singleQuoteDepth = 0.
    std::string input = "['hello', 'world']";
    size_t pos = findInStringRespectingSpecialChars(input, ",", 1);
    ASSERT_NE(pos, std::string::npos);
    EXPECT_EQ(input[pos], ',');
    EXPECT_EQ(pos, 8u);
}

TEST(FindInStringTest, SingleQuoteClosedBeforeColon) {
    // Closing quote followed by colon → closes; first comma is at 13.
    EXPECT_EQ(findInStringRespectingSpecialChars("{'key': 'val', 'k2': 'v2'}", ",", 1), 13u);
}

TEST(FindInStringTest, SingleQuoteClosedAtEndOfString) {
    // j == str.size() branch: quote closes but target is absent → npos.
    EXPECT_EQ(findInStringRespectingSpecialChars("'abc'", ",", 0), std::string::npos);
}

TEST(FindInStringTest, SingleQuoteNotClosedBeforeNonDelimiter) {
    // Closing quote followed by a regular letter → does NOT close; comma hidden.
    EXPECT_EQ(findInStringRespectingSpecialChars("'abc' z, b", ",", 0), std::string::npos);
}

TEST(FindInStringTest, ApostropheInWordNotClosingSingleQuote) {
    // prevIsWord && nextIsWord → treated as plain char inside quoted string.
    std::string input = "['it's the day', 'next']";
    size_t pos = findInStringRespectingSpecialChars(input, ",", 1);
    EXPECT_NE(pos, std::string::npos);
    EXPECT_GT(pos, 14u) << "comma found inside the single-quoted string";
}

TEST(FindInStringTest, PossessiveApostropheInWordNotClosingSingleQuote) {
    // Word char before apostrophe, non-word non-delimiter after → does not close.
    std::string input = "['Johns' car', 'other']";
    size_t pos = findInStringRespectingSpecialChars(input, ",", 1);
    EXPECT_NE(pos, std::string::npos);
    EXPECT_GT(pos, 12u) << "comma found inside the single-quoted string";
}

TEST(FindInStringTest, SingleQuoteClosedWithSpaceBeforeDelimiter) {
    // j skips whitespace before checking for delimiter; quote still closes.
    std::string input = "['abc'  , 'def']";
    size_t pos = findInStringRespectingSpecialChars(input, ",", 1);
    EXPECT_NE(pos, std::string::npos);
    EXPECT_EQ(input[pos], ',');
}

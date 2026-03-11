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
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "src/pull_module/libgit2.hpp"

#include "environment.hpp"

namespace fs = std::filesystem;

TEST(LibGit2RtrimCrLfWhitespace, EmptyString) {
    std::string s;
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_TRUE(s.empty());
}

TEST(LibGit2RtrimCrLfWhitespace, NoWhitespace) {
    std::string s = "abc";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, OnlySpaces) {
    std::string s = "     ";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "");
}

TEST(LibGit2RtrimCrLfWhitespace, LeadingSpacesOnly) {
    std::string s = "   abc";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, TrailingSpacesOnly) {
    std::string s = "abc   ";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, LeadingAndTrailingSpaces) {
    std::string s = "   abc   ";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, TabsAndNewlinesAround) {
    std::string s = "\t\n  abc  \n\t";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, AllCWhitespaceAround) {
    // Include space, tab, newline, vertical tab, form feed, carriage return
    std::string s = " \t\n\v\f\rabc\r\f\v\n\t ";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, PreserveInternalSpaces) {
    std::string s = "  a  b   c  ";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "a  b   c");
}

TEST(LibGit2RtrimCrLfWhitespace, TrailingCRLF) {
    // Windows-style line ending: "\r\n"
    std::string s = "abc\r\n";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, TrailingCROnly) {
    std::string s = "abc\r";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, TrailingLFOnly) {
    std::string s = "abc\n";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, MultipleTrailingCRs) {
    // Only one trailing '\r' is specially removed first, but then trailing
    // whitespace loop will remove any remaining CRs (since isspace('\r') == true).
    std::string s = "abc\r\r\r";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, LeadingCRLFAndSpaces) {
    std::string s = "\r\n  abc";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "abc");
}

TEST(LibGit2RtrimCrLfWhitespace, InternalCRLFShouldRemainIfNotLeadingOrTrailing) {
    // Internal whitespace should be preserved
    std::string s = "a\r\nb";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "a\r\nb");
}

TEST(LibGit2RtrimCrLfWhitespace, OnlyCRLFAndWhitespace) {
    std::string s = "\r\n\t \r";
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, "");
}

TEST(LibGit2RtrimCrLfWhitespace, NonAsciiBytesAreNotTrimmedByIsspace) {
    // 0xC2 0xA0 is UTF-8 for NO-BREAK SPACE; bytes individually are not ASCII spaces.
    // isspace() on unsigned char typically returns false for these bytes in the "C" locale.
    // So they should remain unless at edges and recognized by the current locale (usually not).
    std::string s = "\xC2""\xA0""abc""\xC2""\xA0";
    ovms::rtrimCrLfWhitespace(s);
    // Expect unchanged because these bytes are not recognized by std::isspace in C locale
    EXPECT_EQ(s, "\xC2""\xA0""abc""\xC2""\xA0");
}

TEST(LibGit2RtrimCrLfWhitespace, Idempotent) {
    std::string s = "  abc  \n";
    ovms::rtrimCrLfWhitespace(s);
    auto once = s;
    ovms::rtrimCrLfWhitespace(s);
    EXPECT_EQ(s, once);
}


TEST(LibGit2ContainsCaseInsensitiveTest, ExactMatch) {
    EXPECT_TRUE(ovms::containsCaseInsensitive("hello", "hello"));
}

TEST(LibGit2ContainsCaseInsensitiveTest, MixedCaseMatch) {
    EXPECT_TRUE(ovms::containsCaseInsensitive("HeLLo WoRLD", "world"));
    EXPECT_TRUE(ovms::containsCaseInsensitive("HeLLo WoRLD", "HELLO"));
}

TEST(LibGit2ContainsCaseInsensitiveTest, NoMatch) {
    EXPECT_FALSE(ovms::containsCaseInsensitive("abcdef", "gh"));
}

TEST(LibGit2ContainsCaseInsensitiveTest, EmptyNeedleReturnsTrue) {
    // Consistent with std::string::find("") → 0
    EXPECT_TRUE(ovms::containsCaseInsensitive("something", ""));
}

TEST(LibGit2ContainsCaseInsensitiveTest, EmptyHaystackNonEmptyNeedleReturnsFalse) {
    EXPECT_FALSE(ovms::containsCaseInsensitive("", "abc"));
}

TEST(LibGit2ContainsCaseInsensitiveTest, BothEmptyReturnsTrue) {
    EXPECT_TRUE(ovms::containsCaseInsensitive("", ""));
}

TEST(LibGit2ContainsCaseInsensitiveTest, SubstringAtBeginning) {
    EXPECT_TRUE(ovms::containsCaseInsensitive("HelloWorld", "hello"));
}

TEST(LibGit2ContainsCaseInsensitiveTest, SubstringInMiddle) {
    EXPECT_TRUE(ovms::containsCaseInsensitive("abcHELLOxyz", "hello"));
}

TEST(LibGit2ContainsCaseInsensitiveTest, SubstringAtEnd) {
    EXPECT_TRUE(ovms::containsCaseInsensitive("testCASE", "case"));
}

TEST(LibGit2ContainsCaseInsensitiveTest, NoFalsePositives) {
    EXPECT_FALSE(ovms::containsCaseInsensitive("aaaaa", "b"));
}

TEST(LibGit2ContainsCaseInsensitiveTest, UnicodeCharactersSafeButNotSpecialHandled) {
    // std::tolower only reliably handles unsigned char range.
    // This ensures your implementation does not crash or behave strangely.
    EXPECT_FALSE(ovms::containsCaseInsensitive("ĄĆĘŁ", "ę")); // depends on locale; ASCII-only expected false
}


// A helper for writing test files.
static fs::path writeTempFile(const std::string& filename,
                              const std::string& content) {
    fs::path p = fs::temp_directory_path() / filename;
    std::ofstream out(p, std::ios::binary);
    out << content;
    return p;
}

TEST(LibGit2ReadFirstThreeLinesTest, FileNotFoundReturnsFalse) {
    std::vector<std::string> lines;
    fs::path p = fs::temp_directory_path() / "nonexistent_12345.txt";
    EXPECT_FALSE(ovms::readFirstThreeLines(p, lines));
    EXPECT_TRUE(lines.empty());
}

TEST(LibGit2ReadFirstThreeLinesTest, ReadsExactlyThreeLines) {
    fs::path p = writeTempFile("three_lines.txt",
        "line1\n"
        "line2\n"
        "line3\n"
        "extra\n"); // should be ignored

    std::vector<std::string> out;
    EXPECT_TRUE(ovms::readFirstThreeLines(p, out));
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], "line1");
    EXPECT_EQ(out[1], "line2");
    EXPECT_EQ(out[2], "line3");
}

TEST(LibGit2ReadFirstThreeLinesTest, ReadsFewerThanThreeLines) {
    fs::path p = writeTempFile("two_lines.txt",
        "alpha\n"
        "beta\n");

    std::vector<std::string> out;
    EXPECT_TRUE(ovms::readFirstThreeLines(p, out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], "alpha");
    EXPECT_EQ(out[1], "beta");
}

TEST(LibGit2ReadFirstThreeLinesTest, ReadsOneLineOnly) {
    fs::path p = writeTempFile("one_line.txt", "solo\n");

    std::vector<std::string> out;
    EXPECT_TRUE(ovms::readFirstThreeLines(p, out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], "solo");
}

TEST(LibGit2ReadFirstThreeLinesTest, EmptyFileProducesZeroLinesAndReturnsTrue) {
    fs::path p = writeTempFile("empty.txt", "");

    std::vector<std::string> out;
    EXPECT_TRUE(ovms::readFirstThreeLines(p, out));
    EXPECT_TRUE(out.empty());
}

TEST(LibGit2ReadFirstThreeLinesTest, CRLFIsTrimmedCorrectly) {
    fs::path p = writeTempFile("crlf.txt",
        "hello\r\n"
        "world\r\n");

    std::vector<std::string> out;
    EXPECT_TRUE(ovms::readFirstThreeLines(p, out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], "hello");
    EXPECT_EQ(out[1], "world");
}

TEST(LibGit2ReadFirstThreeLinesTest, LoneCRAndLFAreTrimmed) {
    fs::path p = writeTempFile("mixed_newlines.txt",
        "a\r"
        "b\n"
        "c\r\n");

    std::vector<std::string> out;
    EXPECT_TRUE(ovms::readFirstThreeLines(p, out));

    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], "a");
    EXPECT_EQ(out[1], "b");
    EXPECT_EQ(out[2], "c");
}

TEST(LibGit2ReadFirstThreeLinesTest, VeryLongLineTriggersDrainLogic) {
    constexpr size_t kMax = 8192;
    std::string longLine(kMax, 'x');
    std::string content = longLine + "OVERFLOWTHATSHOULDBEDISCARDED\n"  // should be truncated
                          "line2\n"
                          "line3\n";

    fs::path p = writeTempFile("long_line.txt", content);

    std::vector<std::string> out;
    EXPECT_TRUE(ovms::readFirstThreeLines(p, out));

    ASSERT_EQ(out.size(), 3u);

    // First line should be exactly kMax chars of 'x'
    ASSERT_EQ(out[0].size(), kMax);
    EXPECT_EQ(out[0], std::string(kMax, 'x'));

    EXPECT_EQ(out[1], "line2");
    EXPECT_EQ(out[2], "line3");
}

TEST(LibGit2ReadFirstThreeLinesTest, HandlesEOFWithoutNewlineAtEnd) {
    fs::path p = writeTempFile("eof_no_newline.txt",
                               "first\n"
                               "second\n"
                               "third_without_newline");

    std::vector<std::string> out;
    EXPECT_TRUE(ovms::readFirstThreeLines(p, out));

    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], "first");
    EXPECT_EQ(out[1], "second");
    EXPECT_EQ(out[2], "third_without_newline");
}

TEST(LibGit2ReadFirstThreeLinesTest, TrailingWhitespacePreservedExceptCRLF) {
    fs::path p = writeTempFile("spaces.txt",
        "abc   \n"
        "def\t\t\n");

    std::vector<std::string> out;
    EXPECT_TRUE(ovms::readFirstThreeLines(p, out));

    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], "abc   ");  // spaces preserved
    EXPECT_EQ(out[1], "def\t\t"); // tabs preserved
}

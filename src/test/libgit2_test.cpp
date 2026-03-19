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
#include <random>
#include <string>
#include <system_error>
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
    std::string s = "\xC2"
                    "\xA0"
                    "abc"
                    "\xC2"
                    "\xA0";
    ovms::rtrimCrLfWhitespace(s);
    // Expect unchanged because these bytes are not recognized by std::isspace in C locale
    EXPECT_EQ(s, "\xC2"
                 "\xA0"
                 "abc"
                 "\xC2"
                 "\xA0");
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
    EXPECT_FALSE(ovms::containsCaseInsensitive("ĄĆĘŁ", "ę"));  // depends on locale; ASCII-only expected false
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
        "extra\n");  // should be ignored

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

TEST(LibGit2ReadFirstThreeLinesTest, TrailingWhitespaceNotPreserved) {
    fs::path p = writeTempFile("spaces.txt",
        "abc   \n"
        "def\t\t\n");

    std::vector<std::string> out;
    EXPECT_TRUE(ovms::readFirstThreeLines(p, out));

    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], "abc");  // spaces preserved
    EXPECT_EQ(out[1], "def");  // tabs preserved
}

// Optional: If you need to call readFirstThreeLines in any test-specific checks,
// declare it too (remove if unused here).
// bool readFirstThreeLines(const fs::path& p, std::vector<std::string>& out);

// ---- Test Utilities ----

// Create a unique temporary directory inside the system temp directory.
static fs::path createTempDir() {
    const fs::path base = fs::temp_directory_path();
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    // Try a reasonable number of times to avoid rare collisions
    for (int attempt = 0; attempt < 100; ++attempt) {
        auto candidate = base / ("lfs_kw_tests_" + std::to_string(dist(gen)));
        std::error_code ec;
        if (fs::create_directory(candidate, ec)) {
            return candidate;
        }
        // If creation failed due to existing path, loop and try another name
        // Otherwise (e.g., permissions), fall through and try again up to limit
    }

    throw std::runtime_error("Failed to create a unique temporary directory");
}

static fs::path writeFile(const fs::path& dir, const std::string& name, const std::string& content) {
    fs::path p = dir / name;
    std::ofstream out(p, std::ios::binary);
    if (!out)
        throw std::runtime_error("Failed to create file: " + p.string());
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    return p;
}

// A simple RAII for a temp directory
struct TempDir {
    fs::path dir;
    TempDir() :
        dir(createTempDir()) {
        if (dir.empty())
            throw std::runtime_error("Failed to create temp directory");
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

class LibGit2FileHasLfsKeywordsFirst3PositionalTest : public ::testing::Test {
protected:
    TempDir td;
};

// ---- Tests ----

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, ReturnsFalseForNonExistingFile) {
    fs::path p = td.dir / "does_not_exist.txt";
    EXPECT_FALSE(ovms::fileHasLfsKeywordsFirst3Positional(p));
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, ReturnsFalseForDirectoryPath) {
    // Passing the directory itself (not a regular file)
    EXPECT_FALSE(ovms::fileHasLfsKeywordsFirst3Positional(td.dir));
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, ReturnsFalseForEmptyFile) {
    auto p = writeFile(td.dir, "empty.txt", "");
    EXPECT_FALSE(ovms::fileHasLfsKeywordsFirst3Positional(p));
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, ReturnsFalseForLessThanThreeLines) {
    {
        auto p = writeFile(td.dir, "one_line.txt", "version something\n");
        EXPECT_FALSE(ovms::fileHasLfsKeywordsFirst3Positional(p));
    }
    {
        auto p = writeFile(td.dir, "two_lines.txt", "version x\n"
                                                    "oid y\n");
        EXPECT_FALSE(ovms::fileHasLfsKeywordsFirst3Positional(p));
    }
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, HappyPathCaseInsensitiveAndExtraContent) {
    // Lines contain the keywords somewhere (case-insensitive), extra content is okay.
    const std::string content =
        "  VeRsIoN https://git-lfs.github.com/spec/v1 \n"
        "\toid Sha256:abcdef1234567890\n"
        "size 999999 \t  \n";
    auto p = writeFile(td.dir, "ok.txt", content);
    EXPECT_TRUE(ovms::fileHasLfsKeywordsFirst3Positional(p));
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, WrongOrderShouldFail) {
    // Put keywords in wrong lines
    const std::string content =
        "size 100\n"
        "version something\n"
        "oid abc\n";
    auto p = writeFile(td.dir, "wrong_order.txt", content);
    EXPECT_FALSE(ovms::fileHasLfsKeywordsFirst3Positional(p));
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, MissingKeywordShouldFail) {
    // Line1 has version, line2 missing oid, line3 has size
    const std::string content =
        "version v1\n"
        "hash sha256:abc\n"
        "size 42\n";
    auto p = writeFile(td.dir, "missing_keyword.txt", content);
    EXPECT_FALSE(ovms::fileHasLfsKeywordsFirst3Positional(p));
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, MixedNewlines_CR_LF_CRLF_ShouldPass) {
    // Requires readFirstThreeLines to treat \r, \n, and \r\n as line breaks.
    const std::string content =
        "version one\r"
        "oid two\n"
        "size three\r\n";
    auto p = writeFile(td.dir, "mixed_newlines.txt", content);
    EXPECT_TRUE(ovms::fileHasLfsKeywordsFirst3Positional(p));
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, LeadingAndTrailingWhitespaceDoesNotBreak) {
    // Assuming readFirstThreeLines trims edge whitespace; otherwise contains() still works
    const std::string content =
        "   version   \n"
        "\t oid\t\n"
        " size \t\n";
    auto p = writeFile(td.dir, "whitespace.txt", content);
    EXPECT_TRUE(ovms::fileHasLfsKeywordsFirst3Positional(p));
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, KeywordsMayAppearWithinLongerTextOnEachLine) {
    const std::string content =
        "prefix-version-suffix\n"
        "some_oid_here\n"
        "the_size_is_here\n";
    auto p = writeFile(td.dir, "contains_substrings.txt", content);
    EXPECT_TRUE(ovms::fileHasLfsKeywordsFirst3Positional(p));
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, CaseInsensitiveCheck) {
    const std::string content =
        "VerSiOn 1\n"
        "OID something\n"
        "SiZe 123\n";
    auto p = writeFile(td.dir, "case_insensitive.txt", content);
    EXPECT_TRUE(ovms::fileHasLfsKeywordsFirst3Positional(p));
}

TEST_F(LibGit2FileHasLfsKeywordsFirst3PositionalTest, ExtraLinesAfterFirstThreeDoNotMatter) {
    const std::string content =
        "version v1\n"
        "oid abc\n"
        "size 42\n"
        "EXTRA LINE THAT SHOULD NOT AFFECT RESULT\n";
    auto p = writeFile(td.dir, "extra_lines.txt", content);
    EXPECT_TRUE(ovms::fileHasLfsKeywordsFirst3Positional(p));
}

class LibGit2MakeRelativeToBaseTest : public ::testing::Test {
protected:
    TempDir td;
};

// Base is an ancestor of path → should return the relative tail.
TEST_F(LibGit2MakeRelativeToBaseTest, BaseIsAncestor) {
    fs::path base = td.dir / "root";
    fs::path sub = base / "a" / "b" / "file.txt";

    std::error_code ec;
    fs::create_directories(sub.parent_path(), ec);

    fs::path rel = ovms::makeRelativeToBase(sub, base);
    // Expected: "a/b/file.txt" (platform-correct separators)
    EXPECT_EQ(rel, fs::path("a") / "b" / "file.txt");
}

// Path equals base → fs::relative returns "." (non-empty), we keep it.
TEST_F(LibGit2MakeRelativeToBaseTest, PathEqualsBase) {
    fs::path base = td.dir / "same";
    std::error_code ec;
    fs::create_directories(base, ec);

    fs::path rel = ovms::makeRelativeToBase(base, base);
    EXPECT_EQ(rel, fs::path("."));
}

// Sibling subtree: base is ancestor of both; result is still relative path from base.
TEST_F(LibGit2MakeRelativeToBaseTest, SiblingSubtree) {
    fs::path base = td.dir / "root2";
    fs::path a = base / "a" / "deep" / "fileA.txt";
    fs::path b = base / "b";

    std::error_code ec;
    fs::create_directories(a.parent_path(), ec);
    fs::create_directories(b, ec);

    fs::path rel = ovms::makeRelativeToBase(a, base);
    EXPECT_EQ(rel, fs::path("a") / "deep" / "fileA.txt");
}

// Base is not an ancestor but on same root → return a proper upward relative like "../x/y".
TEST_F(LibGit2MakeRelativeToBaseTest, BaseIsNotAncestorButSameRoot) {
    fs::path base = td.dir / "top" / "left";
    fs::path path = td.dir / "top" / "right" / "x" / "y.txt";

    std::error_code ec;
    fs::create_directories(base, ec);
    fs::create_directories(path.parent_path(), ec);

    fs::path rel = ovms::makeRelativeToBase(path, base);
    // From .../top/left to .../top/right/x/y.txt → "../right/x/y.txt"
    EXPECT_EQ(rel, fs::path("..") / "right" / "x" / "y.txt");
}

// Works even if paths do not exist (lexical computation should still yield a sensible result)
TEST_F(LibGit2MakeRelativeToBaseTest, NonExistingPathsLexicalStillWorks) {
    fs::path base = td.dir / "ghost" / "base";
    fs::path path = td.dir / "ghost" / "base" / "sub" / "file.dat";
    // No directories created

    fs::path rel = ovms::makeRelativeToBase(path, base);
    EXPECT_EQ(rel, fs::path("sub") / "file.dat");
}

// Last resort on Windows: different drive letters → fs::relative fails,
// lexically_relative returns empty → function should return filename only.
#ifdef _WIN32
TEST_F(LibGit2MakeRelativeToBaseTest, DifferentDrivesReturnsFilenameOnly) {
    // NOTE: We don't touch the filesystem; we only test the path logic.
    // Choose typical drive letters; test won't fail if the drive doesn't exist
    // because we don't access the filesystem in lexically_relative path.
    fs::path path = fs::path("D:\\folder\\file.txt");
    fs::path base = fs::path("C:\\another\\base");

    fs::path rel = ovms::makeRelativeToBase(path, base);
    EXPECT_EQ(rel, fs::path("file.txt"));
}
#endif

// If path has no filename (e.g., it's a root), last resort returns path itself.
// On POSIX, "/" has no filename; on Windows, "C:\\" has no filename either.
TEST_F(LibGit2MakeRelativeToBaseTest, NoFilenameEdgeCaseReturnsPathItself) {
    fs::path base = td.dir;  // arbitrary
#if defined(_WIN32)
    fs::path path = fs::path("C:\\");  // has no filename
#else
    fs::path path = fs::path("/");  // root directory, has no filename
#endif

    fs::path rel = ovms::makeRelativeToBase(path, base);
    EXPECT_EQ(rel, path);
}

static void mkdirs(const fs::path& p) {
    std::error_code ec;
    fs::create_directories(p, ec);
}

class LibGit2FindLfsLikeFilesTest : public ::testing::Test {
protected:
    TempDir td;

    // Utility: sort paths lexicographically for deterministic comparison
    static void sortPaths(std::vector<fs::path>& v) {
        std::sort(v.begin(), v.end(), [](const fs::path& a, const fs::path& b) {
            return a.generic_string() < b.generic_string();
        });
    }
};

// --- Tests ---

TEST_F(LibGit2FindLfsLikeFilesTest, NonExistingDirectoryReturnsEmpty) {
    fs::path nonexist = td.dir / "does_not_exist";
    auto matches = ovms::findLfsLikeFiles(nonexist.string(), /*recursive=*/true);
    EXPECT_TRUE(matches.empty());
}

TEST_F(LibGit2FindLfsLikeFilesTest, EmptyDirectoryReturnsEmpty) {
    auto matches = ovms::findLfsLikeFiles(td.dir.string(), /*recursive=*/true);
    EXPECT_TRUE(matches.empty());
}

TEST_F(LibGit2FindLfsLikeFilesTest, NonRecursiveFindsOnlyTopLevelMatches) {
    // Layout:
    //   td.dir/
    //     match_top.txt      (should match)
    //     nomatch_top.txt    (should not match)
    //     sub/
    //       match_nested.txt (should match but NOT included in non-recursive)
    // Matching condition: lines[0] contains "version", lines[1] contains "oid", lines[2] contains "size"

    // Create top-level files
    writeFile(td.dir, "match_top.txt",
        "version v1\n"
        "oid sha256:abc\n"
        "size 123\n");

    writeFile(td.dir, "nomatch_top.txt",
        "version v1\n"
        "hash something\n"  // missing "oid" on line 2
        "size 123\n");

    // Create nested directory and file
    fs::path sub = td.dir / "sub";
    mkdirs(sub);
    writeFile(sub, "match_nested.txt",
        "  VERSION v1  \n"
        "\toid: 123\n"
        "size: 42\n");

    auto matches = ovms::findLfsLikeFiles(td.dir.string(), /*recursive=*/false);
    sortPaths(matches);

    std::vector<fs::path> expected = {fs::path("match_top.txt")};
    sortPaths(expected);

    EXPECT_EQ(matches, expected);
}

TEST_F(LibGit2FindLfsLikeFilesTest, RecursiveFindsNestedMatches) {
    // Same layout as previous test but recursive = true; should include nested match as relative path
    writeFile(td.dir, "top_match.txt",
        "version spec\n"
        "oid hash\n"
        "size 1\n");

    fs::path sub = td.dir / "a" / "b";
    mkdirs(sub);
    writeFile(sub, "nested_match.txt",
        "VeRsIoN\n"
        "OID x\n"
        "SiZe y\n");

    // Add a deeper non-match to ensure it is ignored
    fs::path deeper = td.dir / "a" / "b" / "c";
    mkdirs(deeper);
    writeFile(deeper, "deep_nomatch.txt",
        "hello\n"
        "world\n"
        "!\n");

    auto matches = ovms::findLfsLikeFiles(td.dir.string(), /*recursive=*/true);
    sortPaths(matches);

    std::vector<fs::path> expected = {
        fs::path("top_match.txt"),
        fs::path("a") / "b" / "nested_match.txt"};
    sortPaths(expected);

    EXPECT_EQ(matches, expected);
}

TEST_F(LibGit2FindLfsLikeFilesTest, MixedNewlinesInMatchingFilesAreHandled) {
    // Requires underlying readFirstThreeLines + fileHasLfsKeywordsFirst3Positional to handle \r, \n, \r\n
    writeFile(td.dir, "mixed1.txt",
        "version one\r"
        "oid two\n"
        "size three\r\n");

    auto matches = ovms::findLfsLikeFiles(td.dir.string(), /*recursive=*/false);

    ASSERT_EQ(matches.size(), 1u);
    EXPECT_EQ(matches[0], fs::path("mixed1.txt"));
}

TEST_F(LibGit2FindLfsLikeFilesTest, WrongOrderOrMissingKeywordsAreNotIncluded) {
    writeFile(td.dir, "wrong_order.txt",
        "size 1\n"
        "version 2\n"
        "oid 3\n");  // wrong order → should not match

    writeFile(td.dir, "missing_second.txt",
        "version v1\n"
        "hash something\n"  // missing "oid"
        "size 3\n");

    auto matches = ovms::findLfsLikeFiles(td.dir.string(), /*recursive=*/false);
    EXPECT_TRUE(matches.empty());
}

TEST_F(LibGit2FindLfsLikeFilesTest, OnlyRegularFilesConsidered) {
    // Create a directory with LFS-like name to ensure it isn't treated as a file
    fs::path lfsdir = td.dir / "version_oid_size_dir";
    mkdirs(lfsdir);

    // No files → nothing should match
    auto matches = ovms::findLfsLikeFiles(td.dir.string(), /*recursive=*/true);
    EXPECT_TRUE(matches.empty());
}

TEST_F(LibGit2FindLfsLikeFilesTest, ReturnsPathsRelativeToBaseDirectory) {
    // Ensure results are made relative to the provided base dir.
    writeFile(td.dir, "root_match.txt",
        "version v\n"
        "oid o\n"
        "size s\n");
    fs::path sub = td.dir / "x" / "y";
    mkdirs(sub);
    writeFile(sub, "nested_match.txt",
        "version v\n"
        "oid o\n"
        "size s\n");

    auto matches = ovms::findLfsLikeFiles(td.dir.string(), /*recursive=*/true);
    sortPaths(matches);

    std::vector<fs::path> expected = {
        fs::path("root_match.txt"),
        fs::path("x") / "y" / "nested_match.txt"};
    sortPaths(expected);

    EXPECT_EQ(matches, expected);
}

TEST_F(LibGit2FindLfsLikeFilesTest, NonRecursiveDoesNotDescendButStillUsesRelativePaths) {
    fs::path sub = td.dir / "subdir";
    mkdirs(sub);

    writeFile(td.dir, "toplevel.txt",
        "version a\n"
        "oid b\n"
        "size c\n");

    writeFile(sub, "nested.txt",
        "version a\n"
        "oid b\n"
        "size c\n");

    auto matches_nonrec = ovms::findLfsLikeFiles(td.dir.string(), /*recursive=*/false);
    auto matches_rec = ovms::findLfsLikeFiles(td.dir.string(), /*recursive=*/true);

    // Non-recursive: only top-level
    ASSERT_EQ(matches_nonrec.size(), 1u);
    EXPECT_EQ(matches_nonrec[0], fs::path("toplevel.txt"));

    // Recursive: both, relative to base dir
    sortPaths(matches_rec);
    std::vector<fs::path> expected = {
        fs::path("toplevel.txt"),
        fs::path("subdir") / "nested.txt"};
    sortPaths(expected);
    EXPECT_EQ(matches_rec, expected);
}

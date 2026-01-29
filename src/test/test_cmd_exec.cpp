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

#include <filesystem>
#include <string>

#include "src/pull_module/cmd_exec.hpp"

namespace fs = std::filesystem;

namespace {

// Helper to remove a file if it exists
void removeFileIfExists(const fs::path& path) {
    std::error_code ec;
    fs::remove(path, ec);
}

}  // namespace

class ExecCmdTest : public ::testing::Test {
protected:
    // Injection attempt files in temp directory
    const fs::path injectionFile1 = fs::temp_directory_path() / "OWNED.txt";
    const fs::path injectionFile2 = fs::temp_directory_path() / "OWNED2.txt";

    void SetUp() override {
        // Clean up any leftover files from previous test runs
        removeFileIfExists(injectionFile1);
        removeFileIfExists(injectionFile2);
    }

    void TearDown() override {
        // Clean up after tests
        removeFileIfExists(injectionFile1);
        removeFileIfExists(injectionFile2);
    }
};

// Test that command separator injection is blocked (; on Linux, & on Windows)
TEST_F(ExecCmdTest, CommandSeparatorInjectionBlocked) {
    int returnCode = 0;

#ifdef _WIN32
    // Windows: CreateProcess without cmd.exe doesn't interpret & as command separator
    // The & and everything after becomes arguments to echo, not a separate command
    // Note: We can't easily test this on Windows without cmd.exe since most commands
    // are cmd.exe built-ins. Instead, test that arguments with shell metacharacters
    // are passed literally to the program.
    std::string maliciousCmd = "python.exe -c \"import sys; print(sys.argv)\" \"& echo PWNED > " + injectionFile1.string() + "\"";
    ovms::exec_cmd(maliciousCmd, returnCode);
#else
    // Linux: Attempt command injection with ; (command separator)
    // If vulnerable, this would create the file via shell interpretation
    std::string maliciousCmd = "echo safe; touch " + injectionFile1.string();
    ovms::exec_cmd(maliciousCmd, returnCode);
#endif

    // The injection file should NOT exist if we're secure
    EXPECT_FALSE(fs::exists(injectionFile1))
        << "Command injection via semicolon/ampersand was successful - SECURITY VULNERABILITY!";
}

// Test that command substitution injection is blocked
TEST_F(ExecCmdTest, CommandSubstitutionInjectionBlocked) {
    int returnCode = 0;

#ifdef _WIN32
    // Windows: Without cmd.exe, | is just a character passed to the program
    std::string maliciousCmd = "python.exe -c \"import sys; print(sys.argv)\" \"| echo PWNED > " + injectionFile1.string() + "\"";
    ovms::exec_cmd(maliciousCmd, returnCode);
#else
    // Linux: Attempt injection via subshell $()
    std::string maliciousCmd = "echo $(touch " + injectionFile1.string() + ")";
    ovms::exec_cmd(maliciousCmd, returnCode);
#endif

    EXPECT_FALSE(fs::exists(injectionFile1))
        << "Command injection via subshell was successful - SECURITY VULNERABILITY!";
}

// Test that alternative command substitution injection is blocked
TEST_F(ExecCmdTest, AlternativeSubstitutionInjectionBlocked) {
    int returnCode = 0;

#ifdef _WIN32
    // Windows: Parentheses without cmd.exe are just characters
    std::string maliciousCmd = "python.exe -c \"import sys; print(sys.argv)\" \"(echo PWNED > " + injectionFile1.string() + ")\"";
    ovms::exec_cmd(maliciousCmd, returnCode);
#else
    // Linux: Attempt injection via backticks
    std::string maliciousCmd = "echo `touch " + injectionFile1.string() + "`";
    ovms::exec_cmd(maliciousCmd, returnCode);
#endif

    EXPECT_FALSE(fs::exists(injectionFile1))
        << "Command injection via backticks was successful - SECURITY VULNERABILITY!";
}

// Test complex injection attempt similar to the user's example
TEST_F(ExecCmdTest, ComplexInjectionBlocked) {
    int returnCode = 0;

#ifdef _WIN32
    // Windows: Complex injection attempt - all metacharacters are literal without cmd.exe
    std::string maliciousCmd = "python.exe -c \"import sys; print(sys.argv)\" \"& cmd.exe /c echo PWNED > " + injectionFile1.string() + " & rem\"";
    ovms::exec_cmd(maliciousCmd, returnCode);
#else
    // Linux: Complex injection attempt like: touch /tmp/safe.txt; sh -c 'id >/tmp/OWNED.txt'; #
    std::string maliciousCmd = "echo safe; sh -c 'touch " + injectionFile1.string() + "'; #";
    ovms::exec_cmd(maliciousCmd, returnCode);
#endif

    EXPECT_FALSE(fs::exists(injectionFile1))
        << "Complex command injection was successful - SECURITY VULNERABILITY!";
}

// Test that pipe injection is blocked
TEST_F(ExecCmdTest, PipeInjectionBlocked) {
    int returnCode = 0;

#ifdef _WIN32
    // Windows: Pipe character without cmd.exe is just passed as argument
    std::string maliciousCmd = "python.exe -c \"import sys; print(sys.argv)\" \"| cmd.exe /c echo PWNED > " + injectionFile1.string() + "\"";
    ovms::exec_cmd(maliciousCmd, returnCode);
#else
    // Linux: Attempt injection via pipe
    std::string maliciousCmd = "echo safe | touch " + injectionFile1.string();
    ovms::exec_cmd(maliciousCmd, returnCode);
#endif

    EXPECT_FALSE(fs::exists(injectionFile1))
        << "Command injection via pipe was successful - SECURITY VULNERABILITY!";
}

// Test that legitimate commands still work
TEST_F(ExecCmdTest, LegitimateCommandWorks) {
    int returnCode = -1;

#ifdef _WIN32
    std::string output = ovms::exec_cmd("python.exe -c \"print('hello')\"", returnCode);
    EXPECT_TRUE(output.find("hello") != std::string::npos);
    EXPECT_EQ(returnCode, 0);
#else
    std::string output = ovms::exec_cmd("echo hello", returnCode);
    EXPECT_EQ(output, "hello\n");
    EXPECT_EQ(returnCode, 0);
#endif
}

// Test exec_cmd_utf8 also blocks command separator injection
TEST_F(ExecCmdTest, CommandSeparatorInjectionBlockedUtf8) {
    int returnCode = 0;

#ifdef _WIN32
    std::string maliciousCmd = "python.exe -c \"import sys; print(sys.argv)\" \"& echo PWNED > " + injectionFile1.string() + "\"";
    ovms::exec_cmd_utf8(maliciousCmd, returnCode);
#else
    std::string maliciousCmd = "echo safe; touch " + injectionFile1.string();
    ovms::exec_cmd_utf8(maliciousCmd, returnCode);
#endif

    EXPECT_FALSE(fs::exists(injectionFile1))
        << "Command injection via exec_cmd_utf8 was successful - SECURITY VULNERABILITY!";
}

#ifndef _WIN32
// Tests for parseArguments escape sequence handling (Linux only)
// These tests verify that the argument parser correctly handles quoted strings and escape sequences

// Test that double quotes group arguments correctly
TEST_F(ExecCmdTest, DoubleQuotesGroupArguments) {
    int returnCode = -1;
    // echo should receive "hello world" as a single argument
    std::string output = ovms::exec_cmd("echo \"hello world\"", returnCode);
    EXPECT_EQ(output, "hello world\n");
    EXPECT_EQ(returnCode, 0);
}

// Test that single quotes group arguments correctly
TEST_F(ExecCmdTest, SingleQuotesGroupArguments) {
    int returnCode = -1;
    // echo should receive 'hello world' as a single argument
    std::string output = ovms::exec_cmd("echo 'hello world'", returnCode);
    EXPECT_EQ(output, "hello world\n");
    EXPECT_EQ(returnCode, 0);
}

// Test that escaped quotes inside double quotes work
TEST_F(ExecCmdTest, EscapedQuotesInsideDoubleQuotes) {
    int returnCode = -1;
    // The argument should be: He said "Hello"
    std::string output = ovms::exec_cmd("echo \"He said \\\"Hello\\\"\"", returnCode);
    EXPECT_EQ(output, "He said \"Hello\"\n");
    EXPECT_EQ(returnCode, 0);
}

// Test that escaped backslash inside double quotes work
TEST_F(ExecCmdTest, EscapedBackslashInsideDoubleQuotes) {
    int returnCode = -1;
    // The argument should be: path\to\file
    std::string output = ovms::exec_cmd("echo \"path\\\\to\\\\file\"", returnCode);
    EXPECT_EQ(output, "path\\to\\file\n");
    EXPECT_EQ(returnCode, 0);
}

// Test that backslash outside quotes escapes the next character
TEST_F(ExecCmdTest, BackslashEscapesOutsideQuotes) {
    int returnCode = -1;
    // Escaped space should not split the argument
    std::string output = ovms::exec_cmd("echo hello\\ world", returnCode);
    EXPECT_EQ(output, "hello world\n");
    EXPECT_EQ(returnCode, 0);
}

// Test that single quotes preserve everything literally (no escape processing)
TEST_F(ExecCmdTest, SingleQuotesPreserveLiterally) {
    int returnCode = -1;
    // Inside single quotes, backslash is literal
    std::string output = ovms::exec_cmd("echo 'hello\\nworld'", returnCode);
    EXPECT_EQ(output, "hello\\nworld\n");
    EXPECT_EQ(returnCode, 0);
}

// Test mixed quoting styles
TEST_F(ExecCmdTest, MixedQuotingStyles) {
    int returnCode = -1;
    // Combine single and double quotes
    std::string output = ovms::exec_cmd("echo \"double\"'single'unquoted", returnCode);
    EXPECT_EQ(output, "doublesingleunquoted\n");
    EXPECT_EQ(returnCode, 0);
}

// Test that special shell characters are not interpreted
TEST_F(ExecCmdTest, SpecialCharactersNotInterpreted) {
    int returnCode = -1;
    // These shell metacharacters should be passed literally to echo
    std::string output = ovms::exec_cmd("echo '$HOME $(whoami) `id` ; | & < >'", returnCode);
    EXPECT_EQ(output, "$HOME $(whoami) `id` ; | & < >\n");
    EXPECT_EQ(returnCode, 0);
}
#endif

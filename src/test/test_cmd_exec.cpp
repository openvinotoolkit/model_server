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
    // Windows: Attempt command injection with & (command separator)
    // If vulnerable, this would create the file
    std::string maliciousCmd = "cmd.exe /c echo safe & echo PWNED > " + injectionFile1.string();
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
    // Windows: Attempt injection via pipe
    std::string maliciousCmd = "cmd.exe /c echo safe | echo PWNED > " + injectionFile1.string();
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
    // Windows doesn't use backticks, test another vector
    std::string maliciousCmd = "cmd.exe /c (echo PWNED > " + injectionFile1.string() + ")";
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
    // Windows: Complex injection attempt
    std::string maliciousCmd = "echo.exe safe & cmd.exe /c \"echo PWNED > " + injectionFile1.string() + "\" & rem ";
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
    std::string maliciousCmd = "echo safe | cmd.exe /c echo PWNED > " + injectionFile1.string();
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
    std::string output = ovms::exec_cmd("cmd.exe /c echo hello", returnCode);
    EXPECT_TRUE(output.find("hello") != std::string::npos);
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
    std::string maliciousCmd = "cmd.exe /c echo safe & echo PWNED > " + injectionFile1.string();
    ovms::exec_cmd_utf8(maliciousCmd, returnCode);
#else
    std::string maliciousCmd = "echo safe; touch " + injectionFile1.string();
    ovms::exec_cmd_utf8(maliciousCmd, returnCode);
#endif

    EXPECT_FALSE(fs::exists(injectionFile1))
        << "Command injection via exec_cmd_utf8 was successful - SECURITY VULNERABILITY!";
}

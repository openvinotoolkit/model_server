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
#include "test_with_temp_dir.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>

#include "platform_utils.hpp"

void TestWithTempDir::SetUp() {
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss;
    ss << std::string(test_info->test_suite_name())
       << "/"
       << std::string(test_info->name());
    const std::string directoryName = ss.str();
    directoryPath = getGenericFullPathForTmp("/tmp/" + directoryName);
    std::filesystem::remove_all(directoryPath);
    std::filesystem::create_directories(directoryPath);
}

void TestWithTempDir::TearDown() {
    SPDLOG_DEBUG("Directory tree of: {}\n{}", directoryPath, dirTree(directoryPath));
    if (::testing::Test::HasFailure()) {
        auto filePathsToPrint = searchFilesRecursively(directoryPath, filesToPrintInCaseOfFailure);
        for (const auto& filePath : filePathsToPrint) {
            std::stringstream content;
            std::ifstream file(filePath);
            if (file.is_open()) {
                content << file.rdbuf();
                SPDLOG_ERROR("File:{} Contents:\n{}", filePath, content.str());
            } else {
                SPDLOG_ERROR("Could not open file: {}", filePath);
                continue;
            }
        }
    }
    std::filesystem::remove_all(directoryPath);
}
std::vector<std::string> TestWithTempDir::searchFilesRecursively(const std::string& directoryPath, const std::vector<std::string>& filesToSearch) const {
    std::vector<std::string> foundFiles;
    for (const auto& file : filesToSearch) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && entry.path().filename() == file) {
                foundFiles.push_back(entry.path().string());
                SPDLOG_DEBUG("Found file: {}", entry.path().string());
            }
        }
    }
    return foundFiles;
}

std::string dirTree(const std::string& path, const std::string& indent) {
    if (!std::filesystem::exists(path)) {
        SPDLOG_ERROR("Path does not exist: {}", path);
        return "NON_EXISTENT_PATH";
    }
    std::stringstream tree;
    // if is directory, add to stream its name followed by "/"
    // if is file, add to stream its name

    tree << indent;
    if (!indent.empty()) {
        tree << "|-- ";
    }

    tree << std::filesystem::path(path).filename().string();
    if (std::filesystem::is_directory(path)) {
        tree << "/";
    }
    tree << std::endl;
    if (!std::filesystem::is_directory(path)) {
        return tree.str();
    }
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        std::string passDownIndent = indent.empty() ? "|   " : (indent + "    ");
        tree << dirTree(entry.path().string(), passDownIndent);
    }
    return tree.str();
}

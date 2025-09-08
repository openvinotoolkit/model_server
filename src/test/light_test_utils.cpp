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
#include "light_test_utils.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <iostream>
#include <fstream>

std::string GetFileContents(const std::string& filePath) {
    if (!std::filesystem::exists(filePath)) {
        std::cout << "File does not exist: " << filePath << std::endl;
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    std::ifstream file(filePath, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cout << "File could not be opened: " << filePath << std::endl;
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    std::string content{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    file.close();
    return content;
}
bool createConfigFileWithContent(const std::string& content, std::string filename) {
    std::ofstream configFile{filename};
    // Check if the file was successfully opened
    if (!configFile.is_open()) {
        SPDLOG_ERROR("Failed to open file: {}", filename);
        throw std::runtime_error("Failed to open file: " + filename);
    }
    SPDLOG_INFO("Creating config file: {}\n with content:\n{}", filename, content);
    configFile << content << std::endl;
    configFile.close();
    if (configFile.fail()) {
        SPDLOG_INFO("Closing configFile failed");
        return false;
    } else {
        SPDLOG_INFO("Closing configFile succeed");
    }
    return true;
}

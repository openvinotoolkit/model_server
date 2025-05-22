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
#include "config_export.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "src/filesystem.hpp"
#include "src/modelextensions.hpp"
#include "src/logging.hpp"

namespace ovms {

bool isMediapipeGraphDir(const std::string& path) {
    std::filesystem::path graphPath(path);
    graphPath /= "graph.pbtxt";
    return std::filesystem::exists(graphPath) && std::filesystem::is_regular_file(graphPath);
}
bool isVersionDir(const std::string& path) {
    std::string dirName = std::filesystem::path(path).filename().string();
    return !dirName.empty() &&
           std::all_of(dirName.begin(), dirName.end(), ::isdigit) &&
           dirName[0] != '0' &&
           std::filesystem::is_directory(path);
}

std::string getPartialPath(const std::filesystem::path& path, int depth) {
    std::string partialPath = path.filename().string();
    std::filesystem::path parentPath = path.parent_path();
    for (int i = 0; i < depth; ++i) {
        if (parentPath.empty() || parentPath == parentPath.root_path()) {
            SPDLOG_ERROR("Error trying to get partial path:{}, {}", partialPath, i);
            throw std::runtime_error("Depth is greater than the number of directories");
        }
        partialPath = ovms::FileSystem::appendSlash(parentPath.filename().string()) + partialPath;
        parentPath = parentPath.parent_path();
    }
    SPDLOG_TRACE("Current partial path:{}", partialPath);
    return partialPath;
}
static bool registeredModelFromThisDirectory(const std::filesystem::path& path, std::unordered_map<std::string, ServableType_t>& servablesList, int depth) {
    const std::string partialPath(getPartialPath(path, depth));
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (!isVersionDir(entry.path().string())) {
            continue;
        }
        SPDLOG_TRACE("Entry is a version directory: {}", entry.path().string());
        // i want to have filename + up to 3 levels of directories
        if (hasRequiredExtensions(entry.path().string(), OV_MODEL_FILES_EXTENSIONS)) {
            servablesList[partialPath] = SERVABLE_TYPE_MODEL;
            return true;
        } else if (hasRequiredExtensions(entry.path().string(), ONNX_MODEL_FILES_EXTENSIONS)) {
            servablesList[partialPath] = SERVABLE_TYPE_MODEL;
            return true;
        } else if (hasRequiredExtensions(entry.path().string(), PADDLE_MODEL_FILES_EXTENSIONS)) {
            servablesList[partialPath] = SERVABLE_TYPE_MODEL;
            return true;
        } else if (hasRequiredExtensions(entry.path().string(), TF_MODEL_FILES_EXTENSIONS)) {
            servablesList[partialPath] = SERVABLE_TYPE_MODEL;
            return true;
        } else if (hasRequiredExtensions(entry.path().string(), TFLITE_MODEL_FILES_EXTENSIONS)) {
            servablesList[partialPath] = SERVABLE_TYPE_MODEL;
            return true;
        }
    }
    return false;
}

static bool registeredGraphFromThisDirectory(const std::filesystem::path& path, std::unordered_map<std::string, ServableType_t>& servablesList, int depth) {
    if (isMediapipeGraphDir(path.string())) {
        SPDLOG_TRACE("Found mediapipe graph: {}", path.string());
        servablesList[getPartialPath(path, depth)] = SERVABLE_TYPE_MEDIAPIPEGRAPH;
        return true;
    }
    return false;
}

static void listServables(const std::filesystem::path& directoryPath, std::unordered_map<std::string, ServableType_t>& servablesList, int depth) {
    SPDLOG_TRACE("Listing servables in directory: {}", directoryPath.string());
    if (!std::filesystem::is_directory(directoryPath)) {
        SPDLOG_TRACE("Path is not a directory: {}", directoryPath.string());
        return;
    }
    if (std::filesystem::is_empty(directoryPath)) {
        SPDLOG_TRACE("Directory is empty: {}", directoryPath.string());
        return;
    }
    SPDLOG_TRACE("Directory name: {}", directoryPath.filename().string());
    if (registeredGraphFromThisDirectory(directoryPath.string(), servablesList, depth)) {
        return;
    }
    if (registeredModelFromThisDirectory(directoryPath.string(), servablesList, depth)) {
        return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        if (entry.is_directory()) {
            listServables(entry.path(), servablesList, depth + 1);
        }
    }
    SPDLOG_TRACE("No servables found in directory: {}", directoryPath.string());
}
std::unordered_map<std::string, ServableType_t> listServables(const std::string directoryPath) {
    SPDLOG_TRACE("Listing servables in directory: {}", directoryPath);
    std::unordered_map<std::string, ServableType_t> servablesList;
    std::filesystem::path path(directoryPath);
    if (!std::filesystem::is_directory(path)) {
        SPDLOG_ERROR("Path is not a directory: {}", directoryPath);
        return servablesList;
    }
    if (std::filesystem::is_empty(path)) {
        SPDLOG_ERROR("Directory is empty: {}", directoryPath);
        return servablesList;
    }
    SPDLOG_TRACE("Directory name: {}", path.filename().string());
    for (const auto& file : std::filesystem::directory_iterator(path)) {
        listServables(file.path(), servablesList, 0);
    }
    return servablesList;
}
}  // namespace ovms

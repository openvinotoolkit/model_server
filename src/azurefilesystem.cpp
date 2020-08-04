//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "azurefilesystem.hpp"

#include <filesystem>
#include <fstream>
#include <set>
#include <string>

#include <spdlog/spdlog.h>

namespace ovms {

namespace fs = std::filesystem;

AzureFileSystem::AzureFileSystem() {
    SPDLOG_TRACE("AzureFileSystem default ctor");
    azure::storage::cloud_storage_account storage_account = azure::storage::cloud_storage_account::parse("");
}

AzureFileSystem::~AzureFileSystem() { SPDLOG_TRACE("AzureFileSystem dtor"); }

StatusCode AzureFileSystem::fileExists(const std::string& path, bool* exists) {
    return StatusCode::NOT_IMPLEMENTED;
}

StatusCode AzureFileSystem::isDirectory(const std::string& path,
    bool* is_directory) {
    return StatusCode::NOT_IMPLEMENTED;
}

StatusCode AzureFileSystem::fileModificationTime(const std::string& path,
    int64_t* mtime_ns) {
    return StatusCode::NOT_IMPLEMENTED;
}

StatusCode
AzureFileSystem::getDirectoryContents(const std::string& path,
    std::set<std::string>* contents) {
    return StatusCode::NOT_IMPLEMENTED;
}

StatusCode AzureFileSystem::getDirectorySubdirs(const std::string& path,
    std::set<std::string>* subdirs) {
    return StatusCode::NOT_IMPLEMENTED;
}

StatusCode AzureFileSystem::getDirectoryFiles(const std::string& path,
    std::set<std::string>* files) {
    return StatusCode::NOT_IMPLEMENTED;
}

StatusCode AzureFileSystem::readTextFile(const std::string& path,
    std::string* contents) {
    return StatusCode::NOT_IMPLEMENTED;
}

StatusCode AzureFileSystem::downloadFile(const std::string& remote_path,
    const std::string& local_path) {
    return StatusCode::NOT_IMPLEMENTED;
}

StatusCode AzureFileSystem::downloadFileFolder(const std::string& path,
    std::string* local_path) {
    return StatusCode::NOT_IMPLEMENTED;
}

StatusCode AzureFileSystem::downloadFileFolderTo(const std::string& path,
    const std::string& local_path) {
    return StatusCode::NOT_IMPLEMENTED;
}

StatusCode AzureFileSystem::deleteFileFolder(const std::string& path) {
    return StatusCode::NOT_IMPLEMENTED;
}

}  // namespace ovms

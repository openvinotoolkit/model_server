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
#pragma once

#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "model_version_policy.hpp"
#include "status.hpp"

namespace ovms {

using files_list_t = std::set<std::string>;

class FileSystem {
public:
    /**
     * @brief Destroy the File System object
     * 
     */
    virtual ~FileSystem() {}

    /**
     * @brief Check if given path or file exists
     * 
     * @param path 
     * @param exists 
     * @return StatusCode 
     */
    virtual StatusCode fileExists(const std::string& path, bool* exists) = 0;

    /**
     * @brief Check if given path is a directory
     * 
     * @param path 
     * @param is_dir 
     * @return StatusCode 
     */
    virtual StatusCode isDirectory(const std::string& path, bool* is_dir) = 0;

    /**
     * @brief Get path modification time
     * 
     * @param path 
     * @param mtime_ns 
     * @return StatusCode 
     */
    virtual StatusCode fileModificationTime(const std::string& path, int64_t* mtime_ns) = 0;

    /**
     * @brief Get the files and directories in given directory
     * 
     * @param path 
     * @param contents 
     * @return StatusCode 
     */
    virtual StatusCode getDirectoryContents(const std::string& path, files_list_t* contents) = 0;

    /**
     * @brief Get only directories in given directory
     * 
     * @param path 
     * @param subdirs 
     * @return StatusCode 
     */
    virtual StatusCode getDirectorySubdirs(const std::string& path, files_list_t* subdirs) = 0;

    /**
     * @brief Get only files in given directory
     * 
     * @param path 
     * @param files 
     * @return StatusCode 
     */
    virtual StatusCode getDirectoryFiles(const std::string& path, files_list_t* files) = 0;

    /**
     * @brief Read the content of the given file into a string
     * 
     * @param path 
     * @param contents 
     * @return StatusCode 
     */
    virtual StatusCode readTextFile(const std::string& path, std::string* contents) = 0;

    /**
     * @brief Download a remote directory
     * 
     * @param path 
     * @param local_path 
     * @return StatusCode 
     */
    virtual StatusCode downloadFileFolder(const std::string& path, const std::string& local_path) = 0;

    /**
     * @brief Create a Temp Path
     * 
     * @param local_path 
     * @return StatusCode 
     */
    static StatusCode createTempPath(std::string* local_path) {
        std::string file_template = "/tmp/fileXXXXXX";
        char* tmp_folder = mkdtemp(const_cast<char*>(file_template.c_str()));
        if (tmp_folder == nullptr) {
            spdlog::error("Failed to create local temp folder: {} {}", file_template, strerror(errno));
            return StatusCode::FILESYSTEM_ERROR;
        }
        std::filesystem::permissions(tmp_folder,
            std::filesystem::perms::others_all | std::filesystem::perms::group_all,
            std::filesystem::perm_options::remove);

        *local_path = std::string(tmp_folder);

        return StatusCode::OK;
    }

    /**
     * @brief Download model versions
     * 
     * @param path 
     * @param local_path 
     * @param versions 
     * @return StatusCode 
     */
    virtual StatusCode downloadModelVersions(const std::string& path, std::string* local_path, const std::vector<model_version_t>& versions) = 0;

    /**
     * @brief Delete a folder
     * 
     * @param path 
     * @return StatusCode 
     */
    virtual StatusCode deleteFileFolder(const std::string& path) = 0;

    static const std::vector<std::string> acceptedFiles;
};

}  // namespace ovms

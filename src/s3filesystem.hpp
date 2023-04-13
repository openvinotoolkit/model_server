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

#include <regex>
#include <string>
#include <vector>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>

#include "filesystem.hpp"
#include "status.hpp"

namespace ovms {

class S3FileSystem : public FileSystem {
public:
    /**
     * @brief Construct a new S3FileSystem object
     * 
     * @param options 
     * @param s3_path 
     */
    S3FileSystem(const Aws::SDKOptions& options, const std::string& s3_path);

    /**
     * @brief Destroy the S3FileSystem object
     * 
     */
    ~S3FileSystem();

    /**
     * @brief Check if given path or file exists
     * 
     * @param path 
     * @param exists 
     * @return StatusCode 
     */
    StatusCode fileExists(const std::string& path, bool* exists) override;

    /**
     * @brief Check if given path is a directory
     * 
     * @param path 
     * @param is_dir 
     * @return StatusCode 
     */
    StatusCode isDirectory(const std::string& path, bool* is_dir) override;

    /**
     * @brief Get the files and directories in given directory
     * 
     * @param path 
     * @param contents 
     * @return StatusCode 
     */
    StatusCode getDirectoryContents(const std::string& path, files_list_t* contents) override;

    /**
     * @brief Get only directories in given directory
     * 
     * @param path 
     * @param subdirs 
     * @return StatusCode 
     */
    StatusCode getDirectorySubdirs(const std::string& path, files_list_t* subdirs) override;

    /**
     * @brief Get only files in given directory
     * 
     * @param path 
     * @param files 
     * @return StatusCode 
     */
    StatusCode getDirectoryFiles(const std::string& path, files_list_t* files) override;

    /**
     * @brief Read the content of the given file into a string
     * 
     * @param path 
     * @param contents 
     * @return StatusCode 
     */
    StatusCode readTextFile(const std::string& path, std::string* contents) override;

    /**
     * @brief Download a remote directory
     * 
     * @param path 
     * @param local_path 
     * @return StatusCode 
     */
    StatusCode downloadFileFolder(const std::string& path, const std::string& local_path) override;

    /**
     * @brief Download selected model versions
     * 
     * @param path 
     * @param local_path 
     * @param versions 
     * @return StatusCode 
     */
    StatusCode downloadModelVersions(const std::string& path, std::string* local_path, const std::vector<model_version_t>& versions) override;

    /**
     * @brief Delete a folder
     * 
     * @param path 
     * @return StatusCode 
     */
    StatusCode deleteFileFolder(const std::string& path) override;

private:
    /**
     * @brief 
     * 
     * @param path 
     * @param bucket 
     * @param object 
     * @return StatusCode 
     */
    StatusCode parsePath(const std::string& path, std::string* bucket, std::string* object);

    /**
     * @brief 
     * 
     */
    Aws::SDKOptions options_;

    /**
     * @brief 
     * 
     */
    Aws::S3::S3Client client_;
    std::regex s3_regex_;
    std::regex proxy_regex_;
};

}  // namespace ovms

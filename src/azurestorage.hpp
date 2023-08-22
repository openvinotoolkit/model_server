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

#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "status.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wunused-value"
#include <was/file.h>

#include "was/blob.h"
#include "was/common.h"
#include "was/storage_account.h"
#pragma GCC diagnostic pop

namespace ovms {

namespace as = azure::storage;
using files_list_t = std::set<std::string>;

class AzureStorageAdapter {
public:
    AzureStorageAdapter() {}

    virtual StatusCode fileExists(bool* exists) = 0;
    virtual StatusCode isDirectory(bool* is_directory) = 0;
    virtual StatusCode fileModificationTime(int64_t* mtime_ns) = 0;
    virtual StatusCode getDirectoryContents(files_list_t* contents) = 0;
    virtual StatusCode getDirectorySubdirs(files_list_t* subdirs) = 0;
    virtual StatusCode getDirectoryFiles(files_list_t* files) = 0;
    virtual StatusCode readTextFile(std::string* contents) = 0;
    virtual StatusCode downloadFileFolder(const std::string& local_path) = 0;
    virtual StatusCode deleteFileFolder() = 0;
    virtual StatusCode downloadFile(const std::string& local_path) = 0;
    virtual StatusCode downloadFileFolderTo(const std::string& local_path) = 0;
    virtual StatusCode checkPath(const std::string& path) = 0;

    StatusCode CreateLocalDir(const std::string& path);
    bool isAbsolutePath(const std::string& path);
    std::vector<std::string> FindSubdirectories(std::string path);
    virtual ~AzureStorageAdapter() = default;

protected:
    const std::string extractAzureStorageExceptionMessage(const as::storage_exception& e);

private:
    virtual StatusCode parseFilePath(const std::string& path) = 0;
};

class AzureStorageBlob : public AzureStorageAdapter {
public:
    AzureStorageBlob(const std::string& path, as::cloud_storage_account& account);

    StatusCode checkPath(const std::string& path) override;

    StatusCode fileExists(bool* exists) override;

    StatusCode isDirectory(bool* is_directory) override;

    StatusCode fileModificationTime(int64_t* mtime_ns) override;

    StatusCode getDirectoryContents(files_list_t* contents) override;

    StatusCode getDirectorySubdirs(files_list_t* subdirs) override;

    StatusCode getDirectoryFiles(files_list_t* files) override;

    StatusCode readTextFile(std::string* contents) override;

    StatusCode downloadFileFolder(const std::string& local_path) override;

    StatusCode deleteFileFolder() override;

    StatusCode downloadFile(const std::string& local_path) override;

    StatusCode downloadFileFolderTo(const std::string& local_path) override;

private:
    std::string getLastPathPart(const std::string& path);

    StatusCode parseFilePath(const std::string& path) override;

    std::string getNameFromPath(std::string& path);

    bool isPathValidationOk_;

    std::string fullPath_;

    std::string fullUri_;

    std::string blockpath_;

    std::vector<std::string> subdirs_;

    std::string container_;

    as::cloud_blob_container as_container_;

    as::cloud_block_blob as_block_blob_;

    as::cloud_blob as_blob_;

    as::cloud_storage_account account_;

    as::cloud_blob_client as_blob_client_;
};

class AzureStorageFile : public AzureStorageAdapter {
public:
    AzureStorageFile(const std::string& path, as::cloud_storage_account& account);

    StatusCode checkPath(const std::string& path) override;

    StatusCode fileExists(bool* exists) override;

    StatusCode isDirectory(bool* is_directory) override;

    StatusCode fileModificationTime(int64_t* mtime_ns) override;

    StatusCode getDirectoryContents(files_list_t* contents) override;

    StatusCode getDirectorySubdirs(files_list_t* subdirs) override;

    StatusCode getDirectoryFiles(files_list_t* files) override;

    StatusCode readTextFile(std::string* contents) override;

    StatusCode downloadFileFolder(const std::string& local_path) override;

    StatusCode deleteFileFolder() override;

    StatusCode downloadFile(const std::string& local_path) override;

    StatusCode downloadFileFolderTo(const std::string& local_path) override;

private:
    StatusCode parseFilePath(const std::string& path) override;

    bool isPathValidationOk_;

    std::string fullPath_;

    std::string fullUri_;

    std::string file_;

    std::string directory_;

    std::vector<std::string> subdirs_;

    std::string share_;

    as::cloud_storage_account account_;

    as::cloud_file_client as_file_client_;

    as::cloud_file_share as_share_;

    as::cloud_file_directory as_directory_;

    as::cloud_file as_file1_;
};

class AzureStorageFactory {
public:
    std::shared_ptr<AzureStorageAdapter> getNewAzureStorageObject(const std::string& path, as::cloud_storage_account& account);

private:
    bool isBlobStoragePath(std::string path);
};

}  // namespace ovms

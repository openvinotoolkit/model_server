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

#include "src/status.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wunused-value"
#include <azure/storage/blobs.hpp>
#include <azure/storage/files/shares.hpp>
#pragma GCC diagnostic pop

namespace ovms {

namespace asblobs = Azure::Storage::Blobs;
namespace asfiles = Azure::Storage::Files::Shares;

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
    const std::string extractAzureStorageExceptionMessage(const Azure::Storage::StorageException& e);

private:
    virtual StatusCode parseFilePath(const std::string& path) = 0;
};

class AzureStorageBlob : public AzureStorageAdapter {
public:
    AzureStorageBlob(const std::string& path, const std::string& connection_string);

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

    std::string connection_string_;

    asblobs::BlobContainerClient as_container_;
};

class AzureStorageFile : public AzureStorageAdapter {
public:
    AzureStorageFile(const std::string& path, const std::string& connection_string);

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

    std::string connection_string_;

    asfiles::ShareClient as_share_;

    asfiles::ShareDirectoryClient as_directory_;

    asfiles::ShareFileClient as_file1_;
};

class AzureStorageFactory {
public:
    std::shared_ptr<AzureStorageAdapter> getNewAzureStorageObject(const std::string& path, const std::string& connection_string);

private:
    bool isBlobStoragePath(std::string path);
};

}  // namespace ovms

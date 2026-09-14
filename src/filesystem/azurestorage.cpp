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
#include "azurestorage.hpp"

#include <chrono>
#include <memory>
#include <utility>

#include "azurefilesystem.hpp"
#include "src/logging.hpp"

namespace ovms {

const std::string UNAVAILABLE_PATH_ERROR = "Unable to access path: {}";

const std::string AzureStorageAdapter::extractAzureStorageExceptionMessage(const Azure::Storage::StorageException& e) {
    if (!e.Message.empty()) {
        return e.Message;
    }
    return e.what();
}

StatusCode AzureStorageAdapter::CreateLocalDir(const std::string& path) {
    int status =
        mkdir(path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    if (status == -1) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Failed to create local folder: {} {} ", path,
            strerror(errno));
        return StatusCode::PATH_INVALID;
    }
    return StatusCode::OK;
}

bool AzureStorageAdapter::isAbsolutePath(const std::string& path) {
    return !path.empty() && (path[0] == '/');
}

// ========================================================================
// AzureStorageBlob
// ========================================================================

AzureStorageBlob::AzureStorageBlob(const std::string& path, const std::string& connection_string) :
    isPathValidationOk_(false),
    connection_string_(connection_string),
    as_container_(asblobs::BlobContainerClient("https://placeholder.blob.core.windows.net/placeholder")) {
}

StatusCode AzureStorageBlob::checkPath(const std::string& path) {
    try {
        if (FileSystem::isPathEscaped(path)) {
            SPDLOG_LOGGER_ERROR(azurestorage_logger, "Path {} escape with .. is forbidden.", path);
            return StatusCode::PATH_INVALID;
        }

        auto status = this->parseFilePath(path);
        if (status != StatusCode::OK) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to parse path: {} -> {}", fullPath_,
                ovms::Status(status).string());
            return status;
        }

        as_container_ = asblobs::BlobContainerClient::CreateFromConnectionString(connection_string_, container_);

        try {
            as_container_.GetProperties();
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Container does not exist: {} -> {}", fullPath_, container_);
                return StatusCode::AS_CONTAINER_NOT_FOUND;
            }
            throw;
        }

        isPathValidationOk_ = true;
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageBlob::fileExists(bool* exists) {
    try {
        *exists = false;
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto blob_client = as_container_.GetBlobClient(blockpath_);
        try {
            blob_client.GetProperties();
            *exists = true;
            return StatusCode::OK;
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Block blob does not exist: {} -> {}", fullPath_, blockpath_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageBlob::isDirectory(bool* is_directory) {
    try {
        *is_directory = false;
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        if (blockpath_.empty()) {
            // Container root is always a directory when the container exists
            *is_directory = true;
            return StatusCode::OK;
        }

        // A virtual directory exists if any blobs share the prefix
        Azure::Storage::Blobs::ListBlobsOptions options;
        options.Prefix = blockpath_ + "/";
        options.PageSizeHint = 1;
        auto page = as_container_.ListBlobsByHierarchy("/", options);
        *is_directory = !page.Blobs.empty() || !page.BlobPrefixes.empty();
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageBlob::fileModificationTime(int64_t* mtime_ns) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto blob_client = as_container_.GetBlobClient(blockpath_);
        try {
            auto props = blob_client.GetProperties();
            auto tp = static_cast<std::chrono::system_clock::time_point>(props.Value.LastModified);
            auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
            SPDLOG_LOGGER_TRACE(azurestorage_logger, "Modification time for {} is {}", fullPath_, nanoseconds);
            *mtime_ns = nanoseconds;
            return StatusCode::OK;
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Block blob does not exist: {} -> {}", fullPath_, blockpath_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageBlob::getDirectoryContents(files_list_t* contents) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        Azure::Storage::Blobs::ListBlobsOptions options;
        if (!blockpath_.empty()) {
            options.Prefix = blockpath_ + "/";
        }
        for (auto page = as_container_.ListBlobsByHierarchy("/", options); page.HasPage(); page.MoveToNextPage()) {
            for (const auto& blob : page.Blobs) {
                contents->insert(getLastPathPart(blob.Name));
            }
            for (const auto& prefix : page.BlobPrefixes) {
                contents->insert(getLastPathPart(prefix));
            }
        }
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageBlob::getDirectorySubdirs(files_list_t* subdirs) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        Azure::Storage::Blobs::ListBlobsOptions options;
        if (!blockpath_.empty()) {
            options.Prefix = blockpath_ + "/";
        }
        for (auto page = as_container_.ListBlobsByHierarchy("/", options); page.HasPage(); page.MoveToNextPage()) {
            for (const auto& prefix : page.BlobPrefixes) {
                subdirs->insert(getLastPathPart(prefix));
            }
        }
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageBlob::getDirectoryFiles(files_list_t* files) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        Azure::Storage::Blobs::ListBlobsOptions options;
        if (!blockpath_.empty()) {
            options.Prefix = blockpath_ + "/";
        }
        for (auto page = as_container_.ListBlobsByHierarchy("/", options); page.HasPage(); page.MoveToNextPage()) {
            for (const auto& blob : page.Blobs) {
                files->insert(getLastPathPart(blob.Name));
            }
        }
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageBlob::readTextFile(std::string* contents) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto blob_client = as_container_.GetBlockBlobClient(blockpath_);
        try {
            auto result = blob_client.Download();
            auto body_bytes = result.Value.BodyStream->ReadToEnd();
            *contents = std::string(body_bytes.begin(), body_bytes.end());
            return StatusCode::OK;
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Block blob does not exist: {} -> {}", fullPath_, blockpath_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageBlob::downloadFileFolder(const std::string& local_path) {
    if (!isPathValidationOk_) {
        auto status = checkPath(fullUri_);
        if (status != StatusCode::OK)
            return status;
    }

    SPDLOG_LOGGER_DEBUG(azurestorage_logger,
        "Downloading dir {} (recursive) and saving a new local path: {}",
        fullUri_, local_path);
    return downloadFileFolderTo(local_path);
}

StatusCode AzureStorageBlob::deleteFileFolder() {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto blob_client = as_container_.GetBlobClient(blockpath_);
        try {
            blob_client.Delete();
            return StatusCode::OK;
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "block blob does not exist: {} -> {}", fullPath_, blockpath_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

std::string AzureStorageBlob::getLastPathPart(const std::string& path) {
    std::string proper_path = path;
    if (path.back() == '/')
        proper_path = path.substr(0, path.size() - 1);

    int part_start = proper_path.find_last_of("/");
    int part_end = proper_path.length();

    return proper_path.substr(part_start + 1, part_end - part_start - 1);
}

StatusCode AzureStorageBlob::downloadFile(const std::string& local_path) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto blob_client = as_container_.GetBlobClient(blockpath_);
        try {
            blob_client.DownloadTo(local_path);
            return StatusCode::OK;
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Block blob does not exist: {} -> {}", fullPath_, blockpath_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageBlob::downloadFileFolderTo(const std::string& local_path) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        SPDLOG_LOGGER_TRACE(azurestorage_logger, "Downloading dir {} and saving to {}", fullPath_, local_path);
        bool is_dir;
        auto status = this->isDirectory(&is_dir);
        if (status != StatusCode::OK) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "File/folder does not exist at {}", fullPath_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        if (!is_dir) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Path is not a directory: {}", fullPath_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        std::set<std::string> dirs;
        status = getDirectorySubdirs(&dirs);
        if (status != StatusCode::OK) {
            return status;
        }

        std::set<std::string> files;
        status = getDirectoryFiles(&files);
        if (status != StatusCode::OK) {
            return status;
        }

        for (auto&& d : dirs) {
            std::string remote_dir_path = FileSystem::joinPath({fullUri_, d});
            std::string local_dir_path = FileSystem::joinPath({local_path, d});
            SPDLOG_LOGGER_TRACE(azurestorage_logger, "Processing directory {} from {} -> {}", d, remote_dir_path,
                local_dir_path);

            auto factory = std::make_shared<ovms::AzureStorageFactory>();
            auto azureSubdirStorageObj = factory.get()->getNewAzureStorageObject(remote_dir_path, connection_string_);
            status = azureSubdirStorageObj->checkPath(remote_dir_path);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", remote_dir_path,
                    ovms::Status(status).string());
                return status;
            }

            auto mkdir_status = CreateLocalDir(local_dir_path);
            if (mkdir_status != StatusCode::OK) {
                return status;
            }
            auto download_dir_status = azureSubdirStorageObj->downloadFileFolderTo(local_dir_path);
            if (download_dir_status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to download directory from {} to {}",
                    remote_dir_path, local_dir_path);
                return download_dir_status;
            }
        }

        for (auto&& f : files) {
            std::string remote_file_path = FileSystem::joinPath({fullUri_, f});
            std::string local_file_path = FileSystem::joinPath({local_path, f});
            SPDLOG_LOGGER_TRACE(azurestorage_logger, "Processing file {} from {} -> {}", f, remote_file_path,
                local_file_path);

            auto factory = std::make_shared<ovms::AzureStorageFactory>();
            auto azureFiledirStorageObj = factory.get()->getNewAzureStorageObject(remote_file_path, connection_string_);
            status = azureFiledirStorageObj->checkPath(remote_file_path);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to download directory from {} to {}",
                    remote_file_path, local_file_path);
                return status;
            }

            auto download_status = azureFiledirStorageObj->downloadFile(local_file_path);
            if (download_status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to save file from {} to {}", remote_file_path,
                    local_file_path);
                return download_status;
            }
        }
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

std::string AzureStorageBlob::getNameFromPath(std::string& path) {
    int name_start = path.find_last_of("/");
    int name_end = path.length();

    if (name_start == name_end)
        path = path.substr(0, path.size() - 1);

    name_start = path.find_last_of("/");
    name_end = path.length();

    return path.substr(name_start, name_end - name_start);
}

StatusCode AzureStorageBlob::parseFilePath(const std::string& path) {
    // az://share/blockpath/file
    // az://share/blockpath
    // az://share/
    if (path.back() == '/') {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Path can not end with '/'", path);
        return StatusCode::AS_INVALID_PATH;
    }

    fullUri_ = path;
    int share_start = 0;
    // Blob path
    if (path.find(FileSystem::AZURE_URL_BLOB_PREFIX) != std::string::npos) {
        share_start = path.find(FileSystem::AZURE_URL_BLOB_PREFIX) + FileSystem::AZURE_URL_BLOB_PREFIX.size();
    } else if (path.find(FileSystem::AZURE_URL_FILE_PREFIX) != std::string::npos) {
        // File path
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Wrong object type - az:// prefix in path required, azure:// found:", path);
        return StatusCode::AS_INVALID_PATH;
    } else {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Missing az:// prefix in path:", path);
        return StatusCode::AS_INVALID_PATH;
    }

    int share_end = path.find_first_of("/", share_start);
    int file_end = path.length();

    fullPath_ = path.substr(share_end + 1, file_end - share_end - 1);

    subdirs_ = FindSubdirectories(fullPath_);

    if (share_end > share_start) {
        container_ = path.substr(share_start, share_end - share_start);
        blockpath_ = path.substr(share_end + 1, file_end - share_end - 1);
    } else {
        // No directory and no file
        container_ = path.substr(share_start);
        blockpath_ = "";
    }

    // No container
    if (container_.empty()) {
        return StatusCode::AS_INVALID_PATH;
    }

    return StatusCode::OK;
}

// ========================================================================
// AzureStorageFile
// ========================================================================

AzureStorageFile::AzureStorageFile(const std::string& path, const std::string& connection_string) :
    isPathValidationOk_(false),
    connection_string_(connection_string),
    as_share_(asfiles::ShareClient("https://placeholder.file.core.windows.net/placeholder")),
    as_directory_(asfiles::ShareDirectoryClient("https://placeholder.file.core.windows.net/placeholder/dir")),
    as_file1_(asfiles::ShareFileClient("https://placeholder.file.core.windows.net/placeholder/dir/file")) {
}

StatusCode AzureStorageFile::checkPath(const std::string& path) {
    try {
        if (FileSystem::isPathEscaped(path)) {
            SPDLOG_LOGGER_ERROR(azurestorage_logger, "Path {} escape with .. is forbidden.", path);
            return StatusCode::PATH_INVALID;
        }

        auto status = this->parseFilePath(path);
        if (status != StatusCode::OK) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to parse path: {} -> {}", path,
                ovms::Status(status).string());
            return status;
        }

        as_share_ = asfiles::ShareClient::CreateFromConnectionString(connection_string_, share_);

        try {
            as_share_.GetProperties();
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Share does not exist: {} -> {}", path, share_);
                return StatusCode::AS_SHARE_NOT_FOUND;
            }
            throw;
        }

        if (directory_.empty()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Directory required in path: {} -> {}", path, directory_);
            return StatusCode::AS_INVALID_PATH;
        }

        isPathValidationOk_ = true;
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

static asfiles::ShareDirectoryClient getDeepestExistingDirectory(
    const asfiles::ShareClient& share,
    const std::vector<std::string>& subdirs) {
    auto dir_client = share.GetRootDirectoryClient();
    std::string tmp_dir = "";
    for (const auto& segment : subdirs) {
        if (segment.empty())
            continue;
        tmp_dir = tmp_dir.empty() ? segment : (tmp_dir + "/" + segment);
        try {
            auto sub = share.GetRootDirectoryClient().GetSubdirectoryClient(tmp_dir);
            sub.GetProperties();
            dir_client = sub;
        } catch (const Azure::Storage::StorageException&) {
            break;
        }
    }
    return dir_client;
}

StatusCode AzureStorageFile::fileExists(bool* exists) {
    try {
        *exists = false;
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto dir_client = getDeepestExistingDirectory(as_share_, subdirs_);
        auto file_client = dir_client.GetFileClient(file_);
        try {
            file_client.GetProperties();
            *exists = true;
            return StatusCode::OK;
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "File does not exist: {} -> {}", fullPath_, file_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageFile::isDirectory(bool* is_directory) {
    try {
        *is_directory = false;
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        std::string tmp_dir = "";
        for (const auto& segment : subdirs_) {
            if (segment.empty())
                continue;
            tmp_dir = tmp_dir.empty() ? segment : (tmp_dir + "/" + segment);
            try {
                auto sub = as_share_.GetRootDirectoryClient().GetSubdirectoryClient(tmp_dir);
                sub.GetProperties();
            } catch (const Azure::Storage::StorageException&) {
                return StatusCode::OK;
            }
        }

        *is_directory = true;
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageFile::fileModificationTime(int64_t* mtime_ns) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        as_directory_ = as_share_.GetRootDirectoryClient().GetSubdirectoryClient(directory_);
        try {
            as_directory_.GetProperties();
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Directory does not exist: {} -> {}", fullPath_, directory_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }

        as_file1_ = as_directory_.GetFileClient(file_);
        try {
            auto props = as_file1_.GetProperties();
            auto tp = static_cast<std::chrono::system_clock::time_point>(props.Value.LastModified);
            auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
            SPDLOG_LOGGER_TRACE(azurestorage_logger, "Modification time for {} is {}", fullPath_, nanoseconds);
            *mtime_ns = nanoseconds;
            return StatusCode::OK;
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "File does not exist: {} -> {}", fullPath_, file_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageFile::getDirectoryContents(files_list_t* contents) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto dir_client = getDeepestExistingDirectory(as_share_, subdirs_);

        for (auto page = dir_client.ListFilesAndDirectories(); page.HasPage(); page.MoveToNextPage()) {
            for (const auto& file : page.Files) {
                contents->insert(file.Name);
            }
            for (const auto& dir : page.Directories) {
                contents->insert(dir.Name);
            }
        }
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageFile::getDirectorySubdirs(files_list_t* subdirs) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto dir_client = getDeepestExistingDirectory(as_share_, subdirs_);

        for (auto page = dir_client.ListFilesAndDirectories(); page.HasPage(); page.MoveToNextPage()) {
            for (const auto& dir : page.Directories) {
                subdirs->insert(dir.Name);
            }
        }
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageFile::getDirectoryFiles(files_list_t* files) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto dir_client = getDeepestExistingDirectory(as_share_, subdirs_);

        for (auto page = dir_client.ListFilesAndDirectories(); page.HasPage(); page.MoveToNextPage()) {
            for (const auto& file : page.Files) {
                files->insert(file.Name);
            }
        }
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageFile::readTextFile(std::string* contents) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto dir_client = getDeepestExistingDirectory(as_share_, subdirs_);
        as_file1_ = dir_client.GetFileClient(file_);
        try {
            auto result = as_file1_.Download();
            auto body_bytes = result.Value.BodyStream->ReadToEnd();
            *contents = std::string(body_bytes.begin(), body_bytes.end());
            return StatusCode::OK;
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "File does not exist: {} -> {}", fullPath_, file_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageFile::downloadFileFolder(const std::string& local_path) {
    if (!isPathValidationOk_) {
        auto status = checkPath(fullUri_);
        if (status != StatusCode::OK)
            return status;
    }

    SPDLOG_LOGGER_DEBUG(azurestorage_logger,
        "Downloading dir {} (recursive) and saving a new local path: {}",
        fullPath_, local_path);
    return downloadFileFolderTo(local_path);
}

StatusCode AzureStorageFile::deleteFileFolder() {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto dir_client = getDeepestExistingDirectory(as_share_, subdirs_);
        as_file1_ = dir_client.GetFileClient(file_);
        try {
            as_file1_.Delete();
            return StatusCode::OK;
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "File does not exist: {} -> {}", fullPath_, file_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageFile::downloadFile(const std::string& local_path) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        auto dir_client = getDeepestExistingDirectory(as_share_, subdirs_);
        as_file1_ = dir_client.GetFileClient(file_);
        try {
            as_file1_.DownloadTo(local_path);
            return StatusCode::OK;
        } catch (const Azure::Storage::StorageException& e) {
            if (e.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "File does not exist: {} -> {}", fullPath_, file_);
                return StatusCode::AS_FILE_NOT_FOUND;
            }
            throw;
        }
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageFile::downloadFileFolderTo(const std::string& local_path) {
    try {
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        SPDLOG_LOGGER_TRACE(azurestorage_logger, "Downloading dir {} and saving to {}", fullPath_, local_path);
        bool is_dir;
        auto status = this->isDirectory(&is_dir);
        if (status != StatusCode::OK) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Folder does not exist at {}", fullPath_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        if (!is_dir) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Path is not a directory: {}", fullPath_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        std::set<std::string> dirs;
        status = getDirectorySubdirs(&dirs);
        if (status != StatusCode::OK) {
            return status;
        }

        std::set<std::string> files;
        status = getDirectoryFiles(&files);
        if (status != StatusCode::OK) {
            return status;
        }

        for (auto&& d : dirs) {
            std::string remote_dir_path = FileSystem::joinPath({fullUri_, d});
            std::string local_dir_path = FileSystem::joinPath({local_path, d});
            SPDLOG_LOGGER_TRACE(azurestorage_logger, "Processing directory {} from {} -> {}", d, remote_dir_path,
                local_dir_path);

            auto factory = std::make_shared<ovms::AzureStorageFactory>();
            auto azureSubdirStorageObj = factory.get()->getNewAzureStorageObject(remote_dir_path, connection_string_);
            status = azureSubdirStorageObj->checkPath(remote_dir_path);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", remote_dir_path,
                    ovms::Status(status).string());
                return status;
            }

            auto mkdir_status = CreateLocalDir(local_dir_path);
            if (mkdir_status != StatusCode::OK) {
                return status;
            }
            auto download_dir_status = azureSubdirStorageObj->downloadFileFolderTo(local_dir_path);
            if (download_dir_status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to download directory from {} to {}",
                    remote_dir_path, local_dir_path);
                return download_dir_status;
            }
        }

        for (auto&& f : files) {
            std::string remote_file_path = FileSystem::joinPath({fullUri_, f});
            std::string local_file_path = FileSystem::joinPath({local_path, f});
            SPDLOG_LOGGER_TRACE(azurestorage_logger, "Processing file {} from {} -> {}", f, remote_file_path,
                local_file_path);

            auto factory = std::make_shared<ovms::AzureStorageFactory>();
            auto azureFileStorageObj = factory.get()->getNewAzureStorageObject(remote_file_path, connection_string_);
            status = azureFileStorageObj->checkPath(remote_file_path);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", remote_file_path,
                    ovms::Status(status).string());
                return status;
            }

            auto download_status = azureFileStorageObj->downloadFile(local_file_path);
            if (download_status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to save file from {} to {}", remote_file_path,
                    local_file_path);
                return download_status;
            }
        }
        return StatusCode::OK;
    } catch (const Azure::Storage::StorageException& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }
    return StatusCode::AS_FILE_NOT_FOUND;
}

std::vector<std::string> AzureStorageAdapter::FindSubdirectories(std::string path) {
    std::vector<std::string> output;

    std::string::size_type prev_pos = 0, pos = 0;

    while ((pos = path.find('/', pos)) != std::string::npos) {
        std::string substring(path.substr(prev_pos, pos - prev_pos));
        output.push_back(substring);
        prev_pos = ++pos;
    }

    output.push_back(path.substr(prev_pos, pos - prev_pos));

    return output;
}

StatusCode AzureStorageFile::parseFilePath(const std::string& path) {
    // azure://share/directory/file
    // azure://share/directory
    // azure://share/
    if (path.back() == '/') {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Path can not end with '/'", path);
        return StatusCode::AS_INVALID_PATH;
    }

    fullUri_ = path;
    int share_start = 0;
    // File or directory path
    if (path.find(FileSystem::AZURE_URL_FILE_PREFIX) != std::string::npos) {
        share_start = path.find(FileSystem::AZURE_URL_FILE_PREFIX) + FileSystem::AZURE_URL_FILE_PREFIX.size();
    } else if (path.find(FileSystem::AZURE_URL_BLOB_PREFIX) != std::string::npos) {
        // Blob path
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Wrong object type. azfs:// prefix in path required, found az://:", path);
        return StatusCode::AS_INVALID_PATH;
    } else {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Missing azfs:// prefix in path:", path);
        return StatusCode::AS_INVALID_PATH;
    }

    int share_end = path.find_first_of("/", share_start);
    int file_start = path.find_last_of("/");
    int file_end = path.length();

    fullPath_ = path.substr(share_end + 1, file_end - share_end - 1);

    subdirs_ = FindSubdirectories(fullPath_);

    if (share_end > share_start) {
        share_ = path.substr(share_start, share_end - share_start);
        directory_ = path.substr(share_end + 1, file_start - share_end - 1);

        // No file or no directory
        if (share_end == file_start) {
            file_ = "";
        } else {
            // No file
            if (file_start == file_end) {
                file_ = "";
            } else {
                file_ = path.substr(file_start + 1, file_end - file_start);
            }
        }
    } else {
        // No directory and no file
        share_ = path.substr(share_start);
        directory_ = "";
        file_ = "";
    }

    // No share
    if (share_.empty()) {
        return StatusCode::AS_INVALID_PATH;
    }

    return StatusCode::OK;
}

std::shared_ptr<AzureStorageAdapter> AzureStorageFactory::getNewAzureStorageObject(const std::string& path, const std::string& connection_string) {
    if (isBlobStoragePath(path))
        return std::make_shared<AzureStorageBlob>(path, connection_string);

    return std::make_shared<AzureStorageFile>(path, connection_string);
}

bool AzureStorageFactory::isBlobStoragePath(std::string path) {
    return (path.find(FileSystem::AZURE_URL_BLOB_PREFIX) != std::string::npos);
}

}  // namespace ovms

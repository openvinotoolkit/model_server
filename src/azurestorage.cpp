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

#include <memory>
#include <utility>

#include "azurefilesystem.hpp"
#include "logging.hpp"

namespace ovms {

using namespace utility;

const std::string UNAVAILABLE_PATH_ERROR = "Unable to access path: {}";

const std::string AzureStorageAdapter::extractAzureStorageExceptionMessage(const as::storage_exception& e) {
    as::request_result result = e.result();
    as::storage_extended_error extended_error = result.extended_error();
    if (!extended_error.message().empty()) {
        return extended_error.message();
    } else {
        return e.what();
    }
}

StatusCode AzureStorageAdapter::CreateLocalDir(const std::string& path) {
    int status =
        mkdir(const_cast<char*>(path.c_str()), S_IRUSR | S_IWUSR | S_IXUSR);
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

AzureStorageBlob::AzureStorageBlob(const std::string& path, as::cloud_storage_account& account) {
    account_ = account;
    as_blob_client_ = account_.create_cloud_blob_client();
    isPathValidationOk_ = false;
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

        as_container_ = as_blob_client_.get_container_reference(container_);

        if (!as_container_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Container does not exist: {} -> {}", fullPath_, container_);
            return StatusCode::AS_CONTAINER_NOT_FOUND;
        }

        isPathValidationOk_ = true;

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as_blob_ = as_container_.get_blob_reference(blockpath_);
        if (!as_blob_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Block blob does not exist: {} -> {}", fullPath_, blockpath_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        *exists = true;
        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as::cloud_blob_directory temp_directory = as_container_.get_directory_reference(blockpath_);
        as::cloud_blob_directory parent_directory = temp_directory.get_parent_reference();

        // List blobs in the blob container
        as::continuation_token token;
        do {
            as::list_blob_item_segment result;
            // Check if we are at container root
            if (parent_directory.prefix() == "")
                result = as_container_.list_blobs_segmented(token);
            else
                result = parent_directory.list_blobs_segmented(token);

            for (auto& item : result.results()) {
                if (!item.is_blob()) {
                    std::string prefix = item.as_directory().prefix();
                    if (prefix.back() == '/')
                        prefix.pop_back();
                    if (prefix == blockpath_) {
                        *is_directory = true;
                    }
                }
            }

            token = result.continuation_token();
        } while (!token.empty());

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as_blob_ = as_container_.get_blob_reference(blockpath_);
        if (!as_blob_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Block blob does not exist: {} -> {}", fullPath_, blockpath_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        as::cloud_blob_properties& prop = as_blob_.properties();
        utility::datetime time = prop.last_modified();
        std::string date = time.to_string();

        auto nanoseconds = time.to_interval();

        SPDLOG_LOGGER_TRACE(azurestorage_logger, "Modification time for {} is {}", fullPath_, nanoseconds);
        *mtime_ns = nanoseconds;
        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as::cloud_blob_directory parent_directory = as_container_.get_directory_reference(blockpath_);

        // List blobs in the blob container
        as::continuation_token token;
        do {
            as::list_blob_item_segment result;
            // Check if we are at container root
            if (parent_directory.prefix() == "")
                result = as_container_.list_blobs_segmented(token);
            else
                result = parent_directory.list_blobs_segmented(token);

            for (auto& item : result.results()) {
                if (item.is_blob()) {
                    contents->insert(getLastPathPart(item.as_blob().name()));
                } else {
                    contents->insert(getLastPathPart(item.as_directory().prefix()));
                }
            }
        } while (!token.empty());

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as::cloud_blob_directory parent_directory = as_container_.get_directory_reference(blockpath_);

        // List blobs in the blob container
        as::continuation_token token;
        do {
            as::list_blob_item_segment result;
            // Check if we are at container root
            if (parent_directory.prefix() == "")
                result = as_container_.list_blobs_segmented(token);
            else
                result = parent_directory.list_blobs_segmented(token);

            for (auto& item : result.results()) {
                if (!item.is_blob()) {
                    subdirs->insert(getLastPathPart(item.as_directory().prefix()));
                }
            }
        } while (!token.empty());

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as::cloud_blob_directory parent_directory = as_container_.get_directory_reference(blockpath_);

        // List blobs in the blob container
        as::continuation_token token;
        do {
            as::list_blob_item_segment result;
            // Check if we are at container root
            if (parent_directory.prefix() == "")
                result = as_container_.list_blobs_segmented(token);
            else
                result = parent_directory.list_blobs_segmented(token);

            for (auto& item : result.results()) {
                if (item.is_blob()) {
                    files->insert(getLastPathPart(item.as_blob().name()));
                }
            }
        } while (!token.empty());

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as_blob_ = as_container_.get_blob_reference(blockpath_);
        if (!as_blob_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Nlock blob does not exist: {} -> {}", fullPath_, blockpath_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        as_block_blob_ = as_container_.get_block_blob_reference(blockpath_);

        concurrency::streams::container_buffer<std::vector<uint8_t>> buffer;
        concurrency::streams::ostream output_stream(buffer);
        as_block_blob_.download_to_stream(output_stream);
        *contents = utility::string_t(buffer.collection().begin(), buffer.collection().end());

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as_blob_ = as_container_.get_blob_reference(blockpath_);
        if (!as_blob_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "block blob does not exist: {} -> {}", fullPath_, blockpath_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        as_blob_.delete_blob();
        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as_blob_ = as_container_.get_blob_reference(blockpath_);
        if (!as_blob_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Block blob does not exist: {} -> {}", fullPath_, blockpath_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        as_blob_.download_to_file(local_path);
        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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
            auto azureSubdirStorageObj = factory.get()->getNewAzureStorageObject(remote_dir_path, account_);
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
            auto download_dir_status =
                azureSubdirStorageObj->downloadFileFolderTo(local_dir_path);
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
            auto azureFiledirStorageObj = factory.get()->getNewAzureStorageObject(remote_file_path, account_);
            status = azureFiledirStorageObj->checkPath(remote_file_path);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to download directory from {} to {}",
                    remote_file_path, local_file_path);
                return status;
            }

            auto download_status =
                azureFiledirStorageObj->downloadFile(local_file_path);
            if (download_status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to save file from {} to {}", remote_file_path,
                    local_file_path);
                return download_status;
            }
        }
        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

    std::set<std::string> subdirs;

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

AzureStorageFile::AzureStorageFile(const std::string& path, as::cloud_storage_account& account) {
    account_ = account;
    as_file_client_ = account_.create_cloud_file_client();
    isPathValidationOk_ = false;
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

        as_file_client_ = account_.create_cloud_file_client();
        as_share_ = as_file_client_.get_share_reference(share_);

        if (!as_share_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Share does not exist: {} -> {}", path, share_);
            return StatusCode::AS_SHARE_NOT_FOUND;
        }

        if (directory_.empty()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Directory required in path: {} -> {}", path, directory_);
            return StatusCode::AS_INVALID_PATH;
        }

        isPathValidationOk_ = true;

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to access path: {}", extractAzureStorageExceptionMessage(e));
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, UNAVAILABLE_PATH_ERROR, e.what());
    }

    return StatusCode::AS_FILE_NOT_FOUND;
}

StatusCode AzureStorageFile::fileExists(bool* exists) {
    try {
        *exists = false;
        if (!isPathValidationOk_) {
            auto status = checkPath(fullUri_);
            if (status != StatusCode::OK)
                return status;
        }

        as::cloud_file_directory as_last_working_subdir;

        std::string tmp_dir = "";

        try {
            for (std::vector<std::string>::size_type i = 0; i != subdirs_.size(); i++) {
                tmp_dir = tmp_dir + (i == 0 ? "" : "/") + subdirs_[i];
                as::cloud_file_directory as_tmp_subdir = as_share_.get_directory_reference(tmp_dir);
                if (!as_tmp_subdir.exists()) {
                    break;
                }

                as_last_working_subdir = as_tmp_subdir;
            }
        } catch (const as::storage_exception& e) {
        }

        as_file1_ = as_last_working_subdir.get_file_reference(file_);
        if (!as_file1_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "File does not exist: {} -> {}", fullPath_, file_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        *exists = true;
        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        try {
            for (std::vector<std::string>::size_type i = 0; i != subdirs_.size(); i++) {
                tmp_dir = tmp_dir + (i == 0 ? "" : "/") + subdirs_[i];
                as::cloud_file_directory as_tmp_subdir = as_share_.get_directory_reference(tmp_dir);
                if (!as_tmp_subdir.exists()) {
                    return StatusCode::OK;
                }
            }

            *is_directory = true;
            return StatusCode::OK;
        } catch (const as::storage_exception& e) {
        }
    } catch (const as::storage_exception& e) {
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

        as_directory_ = as_share_.get_directory_reference(_XPLATSTR(directory_));
        if (!as_directory_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Directory does not exist: {} -> {}", fullPath_, directory_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        as_file1_ = as_directory_.get_file_reference(_XPLATSTR(file_));
        if (!as_file1_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "File does not exist: {} -> {}", fullPath_, file_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        as::cloud_file_properties& prop = as_file1_.properties();
        utility::datetime time = prop.last_modified();
        std::string date = time.to_string();

        auto nanoseconds = time.to_interval();

        SPDLOG_LOGGER_TRACE(azurestorage_logger, "Modification time for {} is {}", fullPath_, nanoseconds);
        *mtime_ns = nanoseconds;
        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as::cloud_file_directory as_last_working_subdir;
        std::string tmp_dir = "";

        try {
            for (std::vector<std::string>::size_type i = 0; i != subdirs_.size(); i++) {
                tmp_dir = tmp_dir + (i == 0 ? "" : "/") + subdirs_[i];
                as::cloud_file_directory as_tmp_subdir = as_share_.get_directory_reference(tmp_dir);
                if (!as_tmp_subdir.exists()) {
                    break;
                }

                as_last_working_subdir = as_tmp_subdir;
            }
        } catch (const as::storage_exception& e) {
        }

        // List files and directories in the directory
        as::continuation_token token;
        do {
            as::list_file_and_directory_result_segment result = as_last_working_subdir.list_files_and_directories_segmented(token);
            for (auto& item : result.results()) {
                if (item.is_file()) {
                    contents->insert(item.as_file().name());
                }
                if (item.is_directory()) {
                    contents->insert(item.as_directory().name());
                }
            }
        } while (!token.empty());

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as::cloud_file_directory as_last_working_subdir;
        std::string tmp_dir = "";

        try {
            for (std::vector<std::string>::size_type i = 0; i != subdirs_.size(); i++) {
                tmp_dir = tmp_dir + (i == 0 ? "" : "/") + subdirs_[i];
                as::cloud_file_directory as_tmp_subdir = as_share_.get_directory_reference(tmp_dir);
                if (!as_tmp_subdir.exists()) {
                    break;
                }

                as_last_working_subdir = as_tmp_subdir;
            }
        } catch (const as::storage_exception& e) {
        }

        // List files and directories in the directory
        as::continuation_token token;
        do {
            as::list_file_and_directory_result_segment result = as_last_working_subdir.list_files_and_directories_segmented(token);
            for (auto& item : result.results()) {
                if (item.is_directory()) {
                    subdirs->insert(item.as_directory().name());
                }
            }
        } while (!token.empty());

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as::cloud_file_directory as_last_working_subdir;
        std::string tmp_dir = "";

        try {
            for (std::vector<std::string>::size_type i = 0; i != subdirs_.size(); i++) {
                tmp_dir = tmp_dir + (i == 0 ? "" : "/") + subdirs_[i];
                as::cloud_file_directory as_tmp_subdir = as_share_.get_directory_reference(tmp_dir);
                if (!as_tmp_subdir.exists()) {
                    break;
                }

                as_last_working_subdir = as_tmp_subdir;
            }
        } catch (const as::storage_exception& e) {
        }

        // List files and directories in the directory
        as::continuation_token token;
        do {
            as::list_file_and_directory_result_segment result = as_last_working_subdir.list_files_and_directories_segmented(token);
            for (auto& item : result.results()) {
                if (item.is_file()) {
                    files->insert(item.as_file().name());
                }
            }
        } while (!token.empty());

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as::cloud_file_directory as_last_working_subdir;
        std::string tmp_dir = "";

        try {
            for (std::vector<std::string>::size_type i = 0; i != subdirs_.size(); i++) {
                tmp_dir = tmp_dir + (i == 0 ? "" : "/") + subdirs_[i];
                as::cloud_file_directory as_tmp_subdir = as_share_.get_directory_reference(tmp_dir);
                if (!as_tmp_subdir.exists()) {
                    break;
                }

                as_last_working_subdir = as_tmp_subdir;
            }
        } catch (const as::storage_exception& e) {
        }

        as_file1_ = as_last_working_subdir.get_file_reference(_XPLATSTR(file_));
        if (!as_file1_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "File does not exist: {} -> {}", fullPath_, file_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        concurrency::streams::container_buffer<std::vector<uint8_t>> buffer;
        concurrency::streams::ostream output_stream(buffer);
        as_file1_.download_to_stream(output_stream);
        *contents = utility::string_t(buffer.collection().begin(), buffer.collection().end());

        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as::cloud_file_directory as_last_working_subdir;
        std::string tmp_dir = "";

        try {
            for (std::vector<std::string>::size_type i = 0; i != subdirs_.size(); i++) {
                tmp_dir = tmp_dir + (i == 0 ? "" : "/") + subdirs_[i];
                as::cloud_file_directory as_tmp_subdir = as_share_.get_directory_reference(tmp_dir);
                if (!as_tmp_subdir.exists()) {
                    break;
                }

                as_last_working_subdir = as_tmp_subdir;
            }
        } catch (const as::storage_exception& e) {
        }

        as_file1_ = as_last_working_subdir.get_file_reference(_XPLATSTR(file_));
        if (!as_file1_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "File does not exist: {} -> {}", fullPath_, file_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        as_file1_.delete_file();
        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

        as::cloud_file_directory as_last_working_subdir;
        std::string tmp_dir = "";

        try {
            for (std::vector<std::string>::size_type i = 0; i != subdirs_.size(); i++) {
                tmp_dir = tmp_dir + (i == 0 ? "" : "/") + subdirs_[i];
                as::cloud_file_directory as_tmp_subdir = as_share_.get_directory_reference(tmp_dir);
                if (!as_tmp_subdir.exists()) {
                    break;
                }

                as_last_working_subdir = as_tmp_subdir;
            }
        } catch (const as::storage_exception& e) {
        }

        as_file1_ = as_last_working_subdir.get_file_reference(_XPLATSTR(file_));
        if (!as_file1_.exists()) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "File does not exist: {} -> {}", fullPath_, file_);
            return StatusCode::AS_FILE_NOT_FOUND;
        }

        as_file1_.download_to_file(local_path);
        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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
            auto azureSubdirStorageObj = factory.get()->getNewAzureStorageObject(remote_dir_path, account_);
            auto status = azureSubdirStorageObj->checkPath(remote_dir_path);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", remote_dir_path,
                    ovms::Status(status).string());
                return status;
            }

            auto mkdir_status = CreateLocalDir(local_dir_path);
            if (mkdir_status != StatusCode::OK) {
                return status;
            }
            auto download_dir_status =
                azureSubdirStorageObj->downloadFileFolderTo(local_dir_path);
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
            auto azureFileStorageObj = factory.get()->getNewAzureStorageObject(remote_file_path, account_);
            auto status = azureFileStorageObj->checkPath(remote_file_path);
            if (status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", remote_file_path,
                    ovms::Status(status).string());
                return status;
            }

            auto download_status =
                azureFileStorageObj->downloadFile(local_file_path);
            if (download_status != StatusCode::OK) {
                SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to save file from {} to {}", remote_file_path,
                    local_file_path);
                return download_status;
            }
        }
        return StatusCode::OK;
    } catch (const as::storage_exception& e) {
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

std::shared_ptr<AzureStorageAdapter> AzureStorageFactory::getNewAzureStorageObject(const std::string& path, as::cloud_storage_account& account) {
    if (isBlobStoragePath(path))
        return std::make_shared<AzureStorageBlob>(path, account);

    return std::make_shared<AzureStorageFile>(path, account);
}

bool AzureStorageFactory::isBlobStoragePath(std::string path) {
    return (path.find(FileSystem::AZURE_URL_BLOB_PREFIX) != std::string::npos);
}

}  // namespace ovms

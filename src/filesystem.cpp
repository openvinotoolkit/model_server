//*****************************************************************************
// Copyright 2020-2023 Intel Corporation
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
#include "filesystem.hpp"

#include "stringutils.hpp"

namespace ovms {

const std::string FileSystem::S3_URL_PREFIX = "s3://";

const std::string FileSystem::GCS_URL_PREFIX = "gs://";

const std::string FileSystem::AZURE_URL_FILE_PREFIX = "azfs://";

const std::string FileSystem::AZURE_URL_BLOB_PREFIX = "az://";

#ifdef __linux__
StatusCode FileSystem::createTempPath(std::string* local_path) {
    if (!local_path) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Target path variable for createTempPAth not set.");
        return StatusCode::FILESYSTEM_ERROR;
    }
    std::string file_template = "/tmp/fileXXXXXX";
    char* tmp_folder = mkdtemp(const_cast<char*>(file_template.c_str()));
    if (tmp_folder == nullptr) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create local temp folder: {} {}", file_template, strerror(errno));
        return StatusCode::FILESYSTEM_ERROR;
    }
    fs::permissions(tmp_folder,
        fs::perms::others_all | fs::perms::group_all,
        fs::perm_options::remove);

    *local_path = std::string(tmp_folder);

    return StatusCode::OK;
}
#elif _WIN32
StatusCode FileSystem::createTempPath(std::string* local_path) {
    if (!local_path) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Target path variable for createTempPAth not set.");
        return StatusCode::FILESYSTEM_ERROR;
    }

    wchar_t temp_path[MAX_PATH];
    wchar_t temp_file[MAX_PATH];

    DWORD path_len = GetTempPathW(MAX_PATH, temp_path);
    if (path_len == 0 || path_len > MAX_PATH) {
        DWORD error = GetLastError();
        std::string message = std::system_category().message(error);
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get temp path: {}", message);
        return StatusCode::FILESYSTEM_ERROR;
    }

    UINT unique_num = GetTempFileNameW(temp_path, L"file", 0, temp_file);
    if (unique_num == 0) {
        DWORD error = GetLastError();
        std::string message = std::system_category().message(error);
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create temp file: {}", message);
        return StatusCode::FILESYSTEM_ERROR;
    }

    if (!DeleteFileW(temp_file)) {
        SetLastError(0);
        DeleteFileW(temp_file);
        DWORD error = GetLastError();
        std::string message = std::system_category().message(error);
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to delete temp file: {}", message);
        return StatusCode::FILESYSTEM_ERROR;
    }

    if (!CreateDirectoryW(temp_file, NULL)) {
        SetLastError(0);
        DeleteFileW(temp_file);
        DWORD error = GetLastError();
        std::string message = std::system_category().message(error);
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create temp directory: {}", message);
        return StatusCode::FILESYSTEM_ERROR;
    }

    *local_path = fs::path(temp_file).generic_string();

    return StatusCode::OK;
}
#endif

std::string FileSystem::findFilePathWithExtension(const std::string& path, const std::string& extension) {
    if (isPathEscaped(path)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", path);
        return std::string();
    }

    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (!std::filesystem::is_directory(entry.status())) {
            auto name = entry.path().string();
            if (endsWith(name, extension)) {
                return name;
            }
        }
    }

    return std::string();
}

}  // namespace ovms

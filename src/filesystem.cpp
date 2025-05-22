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

#include <memory>
#ifdef _WIN32
#include <windows.h>
#include <aclapi.h>
#include <sddl.h>
#endif
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

    // Setup proper access rights
    // Get the current user
    DWORD dwSize = 0;
    GetUserNameW(NULL, &dwSize);  // First call to get the required buffer size
    LPWSTR userName = new WCHAR[dwSize];
    auto userNameDeleter = std::shared_ptr<int>(new int, [userName](int* p) { delete[] userName; delete p; });
    if (!GetUserNameW(userName, &dwSize)) {
        DWORD error = GetLastError();
        std::string message = std::system_category().message(error);
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to get username: {}", message);
        return StatusCode::FILESYSTEM_ERROR;
    }
    // Set up the ACL
    EXPLICIT_ACCESSW ea = {0};
    ea.grfAccessPermissions = GENERIC_READ | GENERIC_WRITE;  // Allow only read & write
    ea.grfAccessMode = SET_ACCESS;
    ea.grfInheritance = SUB_CONTAINERS_AND_OBJECTS_INHERIT;  // Apply to subfolders/files
    ea.Trustee.TrusteeForm = TRUSTEE_IS_NAME;
    ea.Trustee.TrusteeType = TRUSTEE_IS_USER;
    ea.Trustee.ptstrName = userName;
    PACL pACL = NULL;
    if (SetEntriesInAclW(1, &ea, NULL, &pACL) != ERROR_SUCCESS) {
        DWORD error = GetLastError();
        std::string message = std::system_category().message(error);
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create ACL: {}", message);
        return StatusCode::FILESYSTEM_ERROR;
    }

    auto pACLDeleter = std::shared_ptr<int>(new int, [pACL](int* p) { LocalFree(pACL); delete p; });
    // Create a Security Descriptor
    PSECURITY_DESCRIPTOR pSD = (PSECURITY_DESCRIPTOR)LocalAlloc(LPTR, SECURITY_DESCRIPTOR_MIN_LENGTH);
    if (pSD == NULL) {
        DWORD error = GetLastError();
        std::string message = std::system_category().message(error);
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to initialize security descriptor: {}", message);
        return StatusCode::FILESYSTEM_ERROR;
    }

    auto pSDDeleter = std::shared_ptr<int>(new int, [pSD](int* p) { LocalFree(pSD); delete p; });
    if (!InitializeSecurityDescriptor(pSD, SECURITY_DESCRIPTOR_REVISION)) {
        DWORD error = GetLastError();
        std::string message = std::system_category().message(error);
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to initialize security descriptor: {}", message);
        return StatusCode::FILESYSTEM_ERROR;
    }
    // Apply the ACL to the security descriptor
    if (!SetSecurityDescriptorDacl(pSD, TRUE, pACL, FALSE)) {
        DWORD error = GetLastError();
        std::string message = std::system_category().message(error);
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to set DACL: {}", message);
        return StatusCode::FILESYSTEM_ERROR;
    }

    // Assign security attributes
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.lpSecurityDescriptor = pSD;
    sa.bInheritHandle = FALSE;  // No handle inheritance

    // Create the directory with the security attributes
    if (!CreateDirectoryW(temp_file, &sa)) {
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

const std::string& FileSystem::getOsSeparator() {
    static std::string separator = std::string(1, std::filesystem::path::preferred_separator);
    return separator;
}

Status FileSystem::createFileOverwrite(const std::string& filePath, const std::string& contents) {
    SPDLOG_DEBUG("Creating file {}", filePath);
    // Always overwrite
    {
        std::ofstream graphFile(filePath, std::ios::trunc | std::ofstream::binary);
        if (graphFile.is_open()) {
            graphFile << contents << std::endl;
        } else {
            SPDLOG_ERROR("Unable to open file: ", filePath);
            return StatusCode::FILE_INVALID;
        }
    }
    return StatusCode::OK;
}

}  // namespace ovms

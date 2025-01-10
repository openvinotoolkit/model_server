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
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "logging.hpp"
#include "model_version_policy.hpp"
#include "openssl/md5.h"
#include "status.hpp"

namespace ovms {

namespace fs = std::filesystem;

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

// TODO: Implement Windows version
#ifdef __linux__
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
    static StatusCode createTempPath(std::string* local_path) {
        if (!local_path){
            DWORD error = GetLastError();
            std::string message = std::system_category().message(error);
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Target path variable for createTempPAth not set: {}, error: {}",local_path, message);
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

    static bool isPathEscaped(const std::string& path) {
        std::size_t lhs = path.find("../");
        std::size_t rhs = path.find("/..");
        return (std::string::npos != lhs && lhs == 0) || (std::string::npos != rhs && rhs == path.length() - 3) || std::string::npos != path.find("/../");
    }

    static const std::string S3_URL_PREFIX;

    static const std::string GCS_URL_PREFIX;

    static const std::string AZURE_URL_FILE_PREFIX;

    static const std::string AZURE_URL_BLOB_PREFIX;

    static bool isLocalFilesystem(const std::string& basePath) {
        if (basePath.rfind(S3_URL_PREFIX, 0) == 0) {
            return false;
        }
        if (basePath.rfind(GCS_URL_PREFIX, 0) == 0) {
            return false;
        }
        if (basePath.rfind(AZURE_URL_FILE_PREFIX, 0) == 0) {
            return false;
        }
        if (basePath.rfind(AZURE_URL_BLOB_PREFIX, 0) == 0) {
            return false;
        }
        return true;
    }

    static void setPath(std::string& path, const std::string& givenPath, const std::string& rootDirectoryPath) {
        if (givenPath.size() == 0) {
            path = rootDirectoryPath;
        } else if (!FileSystem::isLocalFilesystem(givenPath)) {
            // Cloud filesystem
            path = givenPath;
        } else if (givenPath.size() > 0 && isFullPath(givenPath)) {
            // Full path case
            path = givenPath;
        } else {
            // Relative path case
            if (rootDirectoryPath.empty())
                throw std::logic_error("Using relative path without setting graph directory path.");
            path = rootDirectoryPath + givenPath;
        }
    }

    static bool isFullPath(const std::string& inputPath) {
        std::filesystem::path filePath(inputPath);
        try {
            std::filesystem::path absolutePath = std::filesystem::absolute(filePath);
            return absolutePath == filePath;
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Exception during path absolute check for path:", inputPath, e.what());
            return false;
        } catch (...) {
            SPDLOG_ERROR("Exception during path absolute check for path:", inputPath);
            return false;
        }
    }

    static void setRootDirectoryPath(std::string& rootDirectoryPath, const std::string& givenPath) {
        std::string currentWorkingDir = std::filesystem::current_path().string();
        if (givenPath.size() > 1 && givenPath.find_last_of("/\\") != std::string::npos) {
            auto configDirectory = givenPath.substr(0, givenPath.find_last_of("/\\") + 1);
            configDirectory.empty() ? rootDirectoryPath = currentWorkingDir + "/" : rootDirectoryPath = std::move(configDirectory);
        } else {
            rootDirectoryPath = currentWorkingDir + "/";
        }
    }

    static std::string appendSlash(const std::string& name) {
        if (name.empty() || (name.back() == '/')) {
            return name;
        }

        return (name + "/");
    }

    static bool isAbsolutePath(const std::string& path) {
        return !path.empty() && (path[0] == '/');
    }

    static std::string joinPath(std::initializer_list<std::string> segments) {
        std::string joined;

        for (const auto& seg : segments) {
            if (joined.empty()) {
                joined = seg;
            } else if (isAbsolutePath(seg)) {
                if (joined[joined.size() - 1] == '/') {
                    joined.append(seg.substr(1));
                } else {
                    joined.append(seg);
                }
            } else {
                if (joined[joined.size() - 1] != '/') {
                    joined.append("/");
                }
                joined.append(seg);
            }
        }

        return joined;
    }

    static std::string getFileMD5(const std::string& filename) {
        std::ifstream ifs;
        ifs.open(filename);
        std::stringstream strStream;
        strStream << ifs.rdbuf();
        std::string str = strStream.str();
        ifs.close();
        return getStringMD5(str);
    }

    static std::string getStringMD5(const std::string& str) {
        unsigned char result[MD5_DIGEST_LENGTH];

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        MD5((unsigned char*)str.c_str(), str.size(), result);
        std::string md5sum(reinterpret_cast<char*>(result), MD5_DIGEST_LENGTH);
#pragma GCC diagnostic pop
        return (md5sum);
    }

    StatusCode CreateLocalDir(const std::string& path) {
        try {
            fs::create_directory(path);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to create local folder: {} {} ", path, e.what());
            return StatusCode::PATH_INVALID;
        }
        return StatusCode::OK;
    }

    static const std::vector<std::string> acceptedFiles;
};

}  // namespace ovms

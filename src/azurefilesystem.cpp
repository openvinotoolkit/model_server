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

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "logging.hpp"
#include "stringutils.hpp"

namespace ovms {

static as::cloud_storage_account createDefaultOrAnonymousAccount() {
    try {
        const char* env_cred = std::getenv("AZURE_STORAGE_CONNECTION_STRING");

        std::string credentials = std::string(_XPLATSTR("DefaultEndpointsProtocol = https;"));

        if (!env_cred) {
            SPDLOG_LOGGER_TRACE(azurestorage_logger, "Creating AzureFileSystem anonymous connection string.");
        } else {
            credentials = std::string(_XPLATSTR(env_cred));
        }

        as::cloud_storage_account storage_account = as::cloud_storage_account::parse(credentials);
        if (!storage_account.is_initialized()) {
            SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to create default azure storage account");
            throw std::runtime_error("Unable to create default azure storage account");
        }

        const char* use_http = std::getenv("AZURE_STORAGE_USE_HTTP_PROXY");

        const char* proxy_env;

        std::string https_proxy = std::string("");
        if (!use_http) {
            proxy_env = std::getenv("https_proxy");
        } else {
            proxy_env = std::getenv("http_proxy");
        }

        if (!proxy_env) {
            SPDLOG_LOGGER_DEBUG(azurestorage_logger, "No proxy detected.");
        } else {
            https_proxy = std::string(proxy_env);
            web::web_proxy wproxy(https_proxy);
            as::operation_context::set_default_proxy(std::move(wproxy));

            SPDLOG_LOGGER_DEBUG(azurestorage_logger, "Proxy detected: {}" + https_proxy);
        }

        return storage_account;
    } catch (const as::storage_exception& e) {
        as::request_result result = e.result();
        as::storage_extended_error extended_error = result.extended_error();
        if (!extended_error.message().empty()) {
            SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to create default azure storage account: {}", extended_error.message());
        } else {
            SPDLOG_LOGGER_ERROR(azurestorage_logger, "Unable to create default azure storage account: {}", e.what());
        }
        throw e;
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Unable to create default azure storage account: {}", e.what());
        throw e;
    }
}

AzureFileSystem::AzureFileSystem() :
    account_{createDefaultOrAnonymousAccount()} {
    SPDLOG_LOGGER_TRACE(azurestorage_logger, "AzureFileSystem default ctor");
}

AzureFileSystem::~AzureFileSystem() { SPDLOG_LOGGER_TRACE(azurestorage_logger, "AzureFileSystem dtor"); }

StatusCode AzureFileSystem::createAndCheckAzureStorageObject(const std::string& path, std::shared_ptr<AzureStorageAdapter>& azureStorageObj) {
    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
    }
    return status;
}

StatusCode AzureFileSystem::fileExists(const std::string& path, bool* exists) {
    *exists = false;

    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->fileExists(exists);
}

StatusCode AzureFileSystem::isDirectory(const std::string& path,
    bool* is_directory) {
    *is_directory = false;

    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->isDirectory(is_directory);
}

StatusCode AzureFileSystem::fileModificationTime(const std::string& path,
    int64_t* mtime_ns) {

    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->fileModificationTime(mtime_ns);
}

StatusCode
AzureFileSystem::getDirectoryContents(const std::string& path,
    std::set<std::string>* contents) {

    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->getDirectoryContents(contents);
}

StatusCode AzureFileSystem::getDirectorySubdirs(const std::string& path,
    std::set<std::string>* subdirs) {

    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->getDirectorySubdirs(subdirs);
}

StatusCode AzureFileSystem::getDirectoryFiles(const std::string& path,
    std::set<std::string>* files) {

    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->getDirectoryFiles(files);
}

StatusCode AzureFileSystem::readTextFile(const std::string& path,
    std::string* contents) {

    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->readTextFile(contents);
}

StatusCode AzureFileSystem::downloadModelVersions(const std::string& path,
    std::string* local_path,
    const std::vector<model_version_t>& versions) {

    auto sc = createTempPath(local_path);
    if (sc != StatusCode::OK) {
        SPDLOG_LOGGER_ERROR(azurestorage_logger, "Failed to create a temporary path {}", sc);
        return sc;
    }

    for (auto& ver : versions) {
        std::string versionpath = path;
        if (!endsWith(versionpath, "/")) {
            versionpath.append("/");
        }
        versionpath.append(std::to_string(ver));
        std::string lpath = *local_path;
        if (!endsWith(lpath, "/")) {
            lpath.append("/");
        }
        lpath.append(std::to_string(ver));
        fs::create_directory(lpath);

        std::shared_ptr<AzureStorageAdapter> azureStorageObj;
        auto status = createAndCheckAzureStorageObject(versionpath, azureStorageObj);
        if (status != StatusCode::OK) {
            return status;
        }

        status = azureStorageObj->downloadFileFolderTo(lpath);
        if (status != StatusCode::OK) {
            SPDLOG_LOGGER_ERROR(azurestorage_logger, "Failed to download model version {}", versionpath);
            return status;
        }
    }

    return StatusCode::OK;
}

StatusCode AzureFileSystem::downloadFile(const std::string& remote_path,
    const std::string& local_path) {

    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(remote_path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->downloadFile(local_path);
}

StatusCode AzureFileSystem::downloadFileFolder(const std::string& path,
    const std::string& local_path) {
    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->downloadFileFolder(local_path);
}

StatusCode AzureFileSystem::downloadFileFolderTo(const std::string& path,
    const std::string& local_path) {

    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->downloadFileFolderTo(local_path);
}

StatusCode AzureFileSystem::deleteFileFolder(const std::string& path) {
    std::shared_ptr<AzureStorageAdapter> azureStorageObj;
    auto status = createAndCheckAzureStorageObject(path, azureStorageObj);
    if (status != StatusCode::OK) {
        return status;
    }

    return azureStorageObj->deleteFileFolder();
}

}  // namespace ovms

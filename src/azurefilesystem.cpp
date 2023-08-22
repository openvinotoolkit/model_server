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
            as::operation_context::set_default_proxy(wproxy);

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

StatusCode AzureFileSystem::fileExists(const std::string& path, bool* exists) {
    *exists = false;

    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->fileExists(exists);

    return status;
}

StatusCode AzureFileSystem::isDirectory(const std::string& path,
    bool* is_directory) {
    *is_directory = false;

    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->isDirectory(is_directory);

    return status;
}

StatusCode AzureFileSystem::fileModificationTime(const std::string& path,
    int64_t* mtime_ns) {

    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->fileModificationTime(mtime_ns);

    return status;
}

StatusCode
AzureFileSystem::getDirectoryContents(const std::string& path,
    std::set<std::string>* contents) {

    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->getDirectoryContents(contents);

    return status;
}

StatusCode AzureFileSystem::getDirectorySubdirs(const std::string& path,
    std::set<std::string>* subdirs) {

    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->getDirectorySubdirs(subdirs);

    return status;
}

StatusCode AzureFileSystem::getDirectoryFiles(const std::string& path,
    std::set<std::string>* files) {

    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->getDirectoryFiles(files);

    return status;
}

StatusCode AzureFileSystem::readTextFile(const std::string& path,
    std::string* contents) {

    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->readTextFile(contents);

    return status;
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

        auto factory = std::make_shared<ovms::AzureStorageFactory>();
        auto azureStorageObj = factory.get()->getNewAzureStorageObject(versionpath, account_);
        auto status = azureStorageObj->checkPath(versionpath);
        if (status != StatusCode::OK) {
            SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", versionpath,
                ovms::Status(status).string());
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

    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(remote_path, account_);
    auto status = azureStorageObj->checkPath(remote_path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", remote_path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->downloadFile(local_path);

    return status;
}

StatusCode AzureFileSystem::downloadFileFolder(const std::string& path,
    const std::string& local_path) {
    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->downloadFileFolder(local_path);

    return status;
}

StatusCode AzureFileSystem::downloadFileFolderTo(const std::string& path,
    const std::string& local_path) {

    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->downloadFileFolderTo(local_path);

    return status;
}

StatusCode AzureFileSystem::deleteFileFolder(const std::string& path) {
    auto factory = std::make_shared<ovms::AzureStorageFactory>();
    auto azureStorageObj = factory.get()->getNewAzureStorageObject(path, account_);
    auto status = azureStorageObj->checkPath(path);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_WARN(azurestorage_logger, "Check path failed: {} -> {}", path,
            ovms::Status(status).string());
        return status;
    }

    status = azureStorageObj->deleteFileFolder();

    return status;
}

}  // namespace ovms

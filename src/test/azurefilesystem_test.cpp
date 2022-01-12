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

#include <fstream>

#include "spdlog/spdlog.h"

#include "../azurefilesystem.hpp"
#include "gtest/gtest.h"

using namespace ovms;

namespace {

std::string getEnvOrThrow(const std::string& name) {
    const char* p = std::getenv(name.c_str());
    if (!p) {
        spdlog::error("Missing required environment variable: {}", name);
        throw std::runtime_error("Missing required environment variable");
    }
    spdlog::trace("Value of env {} is {}", name, std::string(p));
    return std::string(p);
}

std::string getCredentials() {
    const char* path = "/tmp/ovms_test_azure_credentials.json";
    std::string creds = getEnvOrThrow("AZURE_STORAGE_CONNECTION_STRING");
    std::ofstream f;
    f.open(path);
    f << creds;
    f.close();
    return path;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
std::string getAnonFilePath() { return std::string("azfs://my-sample-share/my-sample-directory/my_test_file.txt"); }

std::string getAnonDirPath() { return std::string("azfs://my-sample-share/my-sample-directory"); }

std::string getAnonBlobFilePath() { return std::string("az://my-sample-container/my-sample-directory/my_test_file.txt"); }

std::string getBlobDirPath() { return std::string("az://my-sample-container/my-sample-directory"); }

std::string getPrivateFilePath() { return std::string("azfs://my-sample-share/my-sample-directory/my-sample-subdirectory/my_test_file2.txt"); }

std::string getPrivateDirPath() { return std::string("azfs://my-sample-share/my-sample-directory/my-sample-subdirectory"); }

std::string getPrivateBlobFilePath() { return std::string("az://my-sample-container/my-sample-directory/my-sample-subdirectory/my_test_file2.txt"); }

std::string getPrivateBlobDirPath() { return std::string("az://my-sample-container/my-sample-directory/my-sample-subdirectory"); }

void check_file_access(const std::string& path, ovms::FileSystem* fs) {
    spdlog::trace("Checking file AS access for {}", path);
    ovms::StatusCode status;
    bool exists;
    status = fs->fileExists(path, &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, true);

    bool is_dir;
    status = fs->isDirectory(path, &is_dir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(is_dir, false);

    /*int64_t time_ns;
    status = fs->fileModificationTime(path, &time_ns);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_GT(time_ns, 0);*/

    std::string file_contents;
    status = fs->readTextFile(
        path, &file_contents);  // make sure file provided in env is small, or
                                // tests will timeout on downloading.
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_GT(file_contents.size(), 0);
}

void check_dir_recursive_download(const std::string& path,
    ovms::FileSystem* fs) {
    spdlog::trace("Checking AS directory in a recursive way for {}", path);
    ovms::StatusCode status;
    std::string local_path_out = "/tmp/ovms_model_dir_123";
    status = fs->downloadFileFolder(path, local_path_out);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    spdlog::trace("Directory saved to {}", local_path_out);
}

void check_dir_access(const std::string& path, ovms::FileSystem* fs) {
    spdlog::trace("Checking directory AS access for {}", path);
    ovms::StatusCode status;
    bool exists;
    status = fs->fileExists(path, &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, true);

    bool is_dir;
    status = fs->isDirectory(path, &is_dir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(is_dir, true);

    std::set<std::string> dir_contents;
    status = fs->getDirectoryContents(path, &dir_contents);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_GT(dir_contents.size(), 0);
    spdlog::trace("DIR CONTENTS:");
    for (auto& d : dir_contents) {
        (void)d;
        spdlog::trace(" -> {}", d);
    }

    std::set<std::string> subdirs;
    status = fs->getDirectorySubdirs(path, &subdirs);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_GT(subdirs.size(), 0);
    spdlog::trace("SUBDIRS:");
    for (auto& sd : subdirs) {
        (void)sd;
        spdlog::trace(" -> {}", sd);
    }
}

}  // namespace

TEST(DISABLED_AzureFileSystem, file_details) {
    spdlog::set_level(spdlog::level::trace);  // uncomment me for debugging.

    auto anon_fs = std::make_shared<ovms::AzureFileSystem>();
    check_file_access(getAnonFilePath(), anon_fs.get());
    check_dir_access(getAnonDirPath(), anon_fs.get());

    auto creds_path = getCredentials();
    ::setenv("AZURE_STORAGE_CONNECTION_STRING", creds_path.c_str(), 1);

    auto fs = std::make_shared<ovms::AzureFileSystem>();
    check_file_access(getAnonFilePath(), fs.get());
    check_dir_access(getAnonDirPath(), fs.get());

    check_dir_recursive_download(getAnonDirPath(), fs.get());

    check_file_access(getPrivateFilePath(), fs.get());
    check_dir_access(getPrivateDirPath(), fs.get());
}
#pragma GCC diagnostic pop

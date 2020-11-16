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

#include "../gcsfilesystem.hpp"
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
    const char* path = "/tmp/ovms_test_gcs_credentials.json";
    std::string creds = getEnvOrThrow("GCS_CREDENTIALS_JSON");
    std::ofstream f;
    f.open(path);
    f << creds;
    f.close();
    return path;
}

std::string getAnonFilePath() { return getEnvOrThrow("GCS_ANON_FILE_PATH"); }

std::string getAnonDirPath() { return getEnvOrThrow("GCS_ANON_DIR_PATH"); }

std::string getPrivateFilePath() { return getEnvOrThrow("GCS_PRIV_FILE_PATH"); }

std::string getPrivateDirPath() { return getEnvOrThrow("GCS_PRIV_DIR_PATH"); }

void check_file_access(const std::string& path, ovms::FileSystem* fs) {
    spdlog::trace("Checking file GCS access for {}", path);
    ovms::StatusCode status;
    bool exists;
    status = fs->fileExists(path, &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, true);

    bool is_dir;
    status = fs->isDirectory(path, &is_dir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(is_dir, false);

    std::string file_contents;
    status = fs->readTextFile(
        path, &file_contents);  // make sure file provided in env is small, or
                                // tests will timeout on downloading.
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_GT(file_contents.size(), 0);
}

void check_dir_recursive_download(const std::string& path,
    ovms::FileSystem* fs) {
    spdlog::trace("Checking GCS directory in a recursive way for {}", path);
    ovms::StatusCode status;
    std::string local_path_out;
    FileSystem::createTempPath(&local_path_out);
    status = fs->downloadFileFolder(path, local_path_out);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    spdlog::trace("Directory saved to {}", local_path_out);
}

void check_dir_access(const std::string& path, ovms::FileSystem* fs) {
    spdlog::trace("Checking directory GCS access for {}", path);
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

TEST(DISABLED_GCSFileSystem, file_details) {
    spdlog::set_level(spdlog::level::trace);  // uncomment me for debugging.

    auto anon_fs = std::make_shared<ovms::GCSFileSystem>();
    check_file_access(getAnonFilePath(), anon_fs.get());
    check_dir_access(getAnonDirPath(), anon_fs.get());

    auto creds_path = getCredentials();
    ::setenv("GOOGLE_APPLICATION_CREDENTIALS", creds_path.c_str(), 1);

    auto fs = std::make_shared<ovms::GCSFileSystem>();
    check_file_access(getAnonFilePath(), fs.get());
    check_dir_access(getAnonDirPath(), fs.get());

    check_dir_recursive_download(getAnonDirPath(), fs.get());

    check_file_access(getPrivateFilePath(), fs.get());
    check_dir_access(getPrivateDirPath(), fs.get());
}

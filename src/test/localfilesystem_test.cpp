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

#include <filesystem>
#include <fstream>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../filesystem.hpp"
#include "../localfilesystem.hpp"

using namespace testing;
using ::testing::UnorderedElementsAre;

const std::string TMP_PATH = "/tmp/structure/";
const std::string TMP_FILE = "file1.txt";
const std::string TMP_CONTENT = "filecontent123\r\n";
const std::string TMP_DIR1 = "dir1";
const std::string TMP_DIR2 = "dir2";

void createTmpFiles() {
    std::ofstream configFile(TMP_PATH + TMP_FILE);
    configFile << TMP_CONTENT << std::endl;
    configFile.close();

    std::filesystem::create_directories(TMP_PATH + TMP_DIR1);
    std::filesystem::create_directories(TMP_PATH + TMP_DIR2);
}

TEST(LocalFileSystem, FileExists) {
    ovms::LocalFileSystem lfs;
    bool exists = false;
    createTmpFiles();

    auto status = lfs.fileExists("/tmp/structure/file.txt", &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, false);

    status = lfs.fileExists("/tmp/structure/dir1", &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, true);
}

TEST(LocalFileSystem, IsDirectory) {
    ovms::LocalFileSystem lfs;
    bool isDir = false;
    createTmpFiles();

    auto status = lfs.isDirectory("/tmp/structure/file.txt", &isDir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(isDir, false);

    status = lfs.isDirectory("/tmp/structure/dir1", &isDir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(isDir, true);

    status = lfs.isDirectory("/tmp/structure/dir5345", &isDir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(isDir, false);
}

TEST(LocalFileSystem, GetDirectoryContents) {
    ovms::LocalFileSystem lfs;
    ovms::files_list_t files;
    createTmpFiles();

    auto status = lfs.getDirectoryContents("/tmp/structure/file.txt", &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryContents("/tmp/structure/file1.txt", &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryContents("/tmp/structure/", &files);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(files.size(), 3);
}

TEST(LocalFileSystem, GetDirectorySubdirs) {
    ovms::LocalFileSystem lfs;
    ovms::files_list_t files;
    createTmpFiles();

    auto status = lfs.getDirectorySubdirs("/tmp/structure/file.txt", &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectorySubdirs("/tmp/structure/file1.txt", &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectorySubdirs("/tmp/structure/", &files);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(files.size(), 2);
}

TEST(LocalFileSystem, GetDirectoryFiles) {
    ovms::LocalFileSystem lfs;
    ovms::files_list_t files;
    createTmpFiles();

    auto status = lfs.getDirectoryFiles("/tmp/structure/file.txt", &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryFiles("/tmp/structure/file1.txt", &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryFiles("/tmp/structure/", &files);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(files.size(), 1);
}

TEST(LocalFileSystem, DownloadFileFolder) {
    ovms::LocalFileSystem lfs;
    std::string location;
    auto status = lfs.downloadFileFolder("/path/to/download", location);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST(LocalFileSystem, DestroyFileFolder) {
    ovms::LocalFileSystem lfs;
    bool exists = false;
    auto status = lfs.fileExists("/tmp/structure/dir1", &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, true);
    status = lfs.deleteFileFolder("/tmp/structure/dir1");
    EXPECT_EQ(status, ovms::StatusCode::OK);
    status = lfs.fileExists("/tmp/structure/dir1", &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, false);
    status = lfs.deleteFileFolder("/tmp/structure/dir1");
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

TEST(FileSystem, CreateTempFolder) {
    std::string local_path;
    namespace fs = std::filesystem;
    auto sc = ovms::FileSystem::createTempPath(&local_path);
    EXPECT_EQ(sc, ovms::StatusCode::OK);
    std::cout << "Temp path:" << local_path << "\n";
    bool status = fs::exists(local_path);
    EXPECT_TRUE(status);
    EXPECT_EQ(sc, ovms::StatusCode::OK);
    fs::perms p = fs::status(local_path).permissions();
    EXPECT_TRUE((p & fs::perms::group_read) == fs::perms::none);
    EXPECT_TRUE((p & fs::perms::others_read) == fs::perms::none);
    EXPECT_TRUE((p & fs::perms::owner_read) != fs::perms::none);
}

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

#ifdef __linux__
    const std::string TMP_PATH = "/tmp/structure/";
    const std::string TMP_CONTENT = "filecontent123\n";
#elif _WIN32
    const std::filesystem::path TMP_PATH = std::filesystem::temp_directory_path() / "structure";
    const std::string TMP_CONTENT = "filecontent123\r\n";
#endif

const std::string TMP_FILE = "file1.txt";
const std::string TMP_DIR1 = "dir1";
const std::string TMP_DIR2 = "dir2";

static void createTmpFiles() {
    std::ofstream configFile((TMP_PATH / TMP_FILE).string());
    configFile << TMP_CONTENT << std::endl;
    configFile.close();

    std::filesystem::create_directories(TMP_PATH / TMP_DIR1);
    std::filesystem::create_directories(TMP_PATH / TMP_DIR2);
}

TEST(LocalFileSystem, FileExists) {
    ovms::LocalFileSystem lfs;
    bool exists = false;
    createTmpFiles();

    auto status = lfs.fileExists((TMP_PATH / "file.txt").string(), &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, false);

    status = lfs.fileExists((TMP_PATH / "dir1").string(), &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, true);
}

TEST(LocalFileSystem, IsDirectory) {
    ovms::LocalFileSystem lfs;
    bool isDir = false;
    createTmpFiles();

    auto status = lfs.isDirectory((TMP_PATH / "file.txt").string(), &isDir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(isDir, false);

    status = lfs.isDirectory((TMP_PATH / "dir1").string(), &isDir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(isDir, true);

    status = lfs.isDirectory((TMP_PATH / "dir5345").string(), &isDir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(isDir, false);
}

TEST(LocalFileSystem, GetDirectoryContents) {
    ovms::LocalFileSystem lfs;
    ovms::files_list_t files;
    createTmpFiles();

    auto status = lfs.getDirectoryContents((TMP_PATH / "file.txt").string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryContents((TMP_PATH / "file1.txt").string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryContents((TMP_PATH).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(files.size(), 3);
}

TEST(LocalFileSystem, GetDirectorySubdirs) {
    ovms::LocalFileSystem lfs;
    ovms::files_list_t files;
    createTmpFiles();

    auto status = lfs.getDirectorySubdirs((TMP_PATH / "file.txt").string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectorySubdirs((TMP_PATH / "file1.txt").string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectorySubdirs((TMP_PATH).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(files.size(), 2);
}

TEST(LocalFileSystem, GetDirectoryFiles) {
    ovms::LocalFileSystem lfs;
    ovms::files_list_t files;
    createTmpFiles();

    auto status = lfs.getDirectoryFiles((TMP_PATH / "file.txt").string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryFiles((TMP_PATH / "file1.txt").string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryFiles((TMP_PATH).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(files.size(), 1);
}

TEST(LocalFileSystem, DownloadFileFolder) {
    ovms::LocalFileSystem lfs;
    std::string location;
    auto status = lfs.downloadFileFolder((TMP_PATH / "download").string(), location);
    // auto status = lfs.downloadFileFolder("/path/to/download", location);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST(LocalFileSystem, DestroyFileFolder) {
    ovms::LocalFileSystem lfs;
    bool exists = false;
    auto status = lfs.fileExists((TMP_PATH / "dir1").string(), &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, true);
    status = lfs.deleteFileFolder((TMP_PATH / "dir1").string());
    EXPECT_EQ(status, ovms::StatusCode::OK);
    status = lfs.fileExists((TMP_PATH / "dir1").string(), &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, false);
    status = lfs.deleteFileFolder((TMP_PATH / "dir1").string());
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

    #ifdef __linux__
        fs::perms p = fs::status(local_path).permissions();
        EXPECT_TRUE((p & fs::perms::group_read) == fs::perms::none);
        EXPECT_TRUE((p & fs::perms::others_read) == fs::perms::none);
        EXPECT_TRUE((p & fs::perms::owner_read) != fs::perms::none);
    #endif
}

TEST(FileSystem, CheckIfPathIsEscaped) {
    ASSERT_TRUE(ovms::FileSystem::isPathEscaped("/../"));
    ASSERT_TRUE(ovms::FileSystem::isPathEscaped("/.."));
    ASSERT_TRUE(ovms::FileSystem::isPathEscaped("../"));
    ASSERT_TRUE(!ovms::FileSystem::isPathEscaped("/path/..resnet/"));
    ASSERT_TRUE(!ovms::FileSystem::isPathEscaped("/path/resnet../"));
}

TEST(FileSystem, IsLocalFilesystem) {
    ASSERT_TRUE(ovms::FileSystem::isLocalFilesystem(""));
    ASSERT_FALSE(ovms::FileSystem::isLocalFilesystem("s3://"));
    ASSERT_FALSE(ovms::FileSystem::isLocalFilesystem("gs://"));
    ASSERT_FALSE(ovms::FileSystem::isLocalFilesystem("azfs://"));
    ASSERT_FALSE(ovms::FileSystem::isLocalFilesystem("az://"));
    ASSERT_TRUE(ovms::FileSystem::isLocalFilesystem("nanas3://"));
    ASSERT_TRUE(ovms::FileSystem::isLocalFilesystem("...gs://"));
    ASSERT_TRUE(ovms::FileSystem::isLocalFilesystem("/azfs://"));
    ASSERT_TRUE(ovms::FileSystem::isLocalFilesystem("o_O$az://"));
    ASSERT_TRUE(ovms::FileSystem::isLocalFilesystem("../"));
    ASSERT_TRUE(ovms::FileSystem::isLocalFilesystem("/localfilesystem"));
    ASSERT_TRUE(ovms::FileSystem::isLocalFilesystem("/long/local/filesystem"));
}

TEST(FileSystem, SetRootDirectoryPath) {
    std::string rootPath = "";
    std::string givenPath = "/givenpath";

    auto normalize_path = [](const std::string& path) -> std::string {
        return fs::weakly_canonical(fs::path(path)).string();
    };

    ovms::FileSystem::setRootDirectoryPath(rootPath, givenPath);
    ASSERT_EQ(normalize_path(rootPath), normalize_path("/"));

    givenPath = "/givenpath/longer";
    ovms::FileSystem::setRootDirectoryPath(rootPath, givenPath);
    ASSERT_EQ(normalize_path(rootPath), normalize_path("/givenpath/"));

    givenPath = "/givenpath/longer/somefile.txt";
    ovms::FileSystem::setRootDirectoryPath(rootPath, givenPath);
    ASSERT_EQ(normalize_path(rootPath), normalize_path("/givenpath/longer/"));

    givenPath = "givenpath";
    ovms::FileSystem::setRootDirectoryPath(rootPath, givenPath);
    std::string currentWorkingDir = std::filesystem::current_path();
    ASSERT_EQ(normalize_path(rootPath), normalize_path(ovms::FileSystem::joinPath({currentWorkingDir, ""})));

    givenPath = "/givenpath/";
    ovms::FileSystem::setRootDirectoryPath(rootPath, givenPath);
    ASSERT_EQ(normalize_path(rootPath), normalize_path(givenPath));

    givenPath = "1";
    ovms::FileSystem::setRootDirectoryPath(rootPath, givenPath);
    ASSERT_EQ(normalize_path(rootPath), normalize_path(ovms::FileSystem::joinPath({currentWorkingDir, ""})));

    givenPath = "";
    ovms::FileSystem::setRootDirectoryPath(rootPath, givenPath);
    ASSERT_EQ(normalize_path(rootPath), normalize_path(ovms::FileSystem::joinPath({currentWorkingDir, ""})));
}

TEST(FileSystem, SetPath) {
    std::string rootPath = "";
    std::string testPath = "";
    std::string givenPath = "";

    try {
        ovms::FileSystem::setPath(testPath, givenPath, rootPath);
    } catch (std::logic_error& e) {
    }

    rootPath = "/rootPath";
    testPath = "";
    givenPath = "";

    ovms::FileSystem::setPath(testPath, givenPath, rootPath);
    ASSERT_EQ(testPath, rootPath);

    testPath = "";
    givenPath = "/givenPath";

    ovms::FileSystem::setPath(testPath, givenPath, rootPath);
    ASSERT_EQ(testPath, "/givenPath");

    testPath = "";
    givenPath = "givenPath";

    ovms::FileSystem::setPath(testPath, givenPath, rootPath);
    ASSERT_EQ(testPath, "/rootPathgivenPath");

    testPath = "";
    givenPath = "long/givenPath";

    ovms::FileSystem::setPath(testPath, givenPath, rootPath);
    ASSERT_EQ(testPath, "/rootPathlong/givenPath");

    testPath = "";
    givenPath = "s3://long/givenPath";

    ovms::FileSystem::setPath(testPath, givenPath, rootPath);
    ASSERT_EQ(testPath, givenPath);
}

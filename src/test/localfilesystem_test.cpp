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
#ifdef _WIN32
#include <windows.h>
#include <iostream>
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../filesystem.hpp"
#include "../localfilesystem.hpp"

using namespace testing;
using ::testing::UnorderedElementsAre;

#ifdef __linux__
const std::filesystem::path TMP_PATH = "/tmp/structure/";
const std::string TMP_CONTENT = "filecontent123\n";
#elif _WIN32
const std::filesystem::path TMP_PATH = std::filesystem::temp_directory_path() / "structure";
const std::string TMP_CONTENT = "filecontent123\r\n";
#endif

const std::filesystem::path TMP_FILE = "file.txt";
const std::filesystem::path TMP_FILE1 = "file1.txt";
const std::filesystem::path TMP_DIR1 = "dir1";
const std::filesystem::path TMP_DIR2 = "dir2";
const std::filesystem::path TMP_DIR5345 = "dir5345";

static void createTmpFiles() {
    std::ofstream configFile((TMP_PATH / TMP_FILE1).string());
    configFile << TMP_CONTENT << std::endl;
    configFile.close();

    std::filesystem::create_directories(TMP_PATH / TMP_DIR1);
    std::filesystem::create_directories(TMP_PATH / TMP_DIR2);
}

TEST(LocalFileSystem, FileExists) {
    ovms::LocalFileSystem lfs;
    bool exists = false;
    createTmpFiles();

    auto status = lfs.fileExists((TMP_PATH / TMP_FILE).string(), &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, false);

    status = lfs.fileExists((TMP_PATH / TMP_DIR1).string(), &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, true);
}

TEST(LocalFileSystem, IsDirectory) {
    ovms::LocalFileSystem lfs;
    bool isDir = false;
    createTmpFiles();

    auto status = lfs.isDirectory((TMP_PATH / TMP_FILE).string(), &isDir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(isDir, false);

    status = lfs.isDirectory((TMP_PATH / TMP_DIR1).string(), &isDir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(isDir, true);

    status = lfs.isDirectory((TMP_PATH / TMP_DIR5345).string(), &isDir);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(isDir, false);
}

TEST(LocalFileSystem, GetDirectoryContents) {
    ovms::LocalFileSystem lfs;
    ovms::files_list_t files;
    createTmpFiles();

    auto status = lfs.getDirectoryContents((TMP_PATH / TMP_FILE).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryContents((TMP_PATH / TMP_FILE1).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryContents((TMP_PATH).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(files.size(), 3);
}

TEST(LocalFileSystem, GetDirectorySubdirs) {
    ovms::LocalFileSystem lfs;
    ovms::files_list_t files;
    createTmpFiles();

    auto status = lfs.getDirectorySubdirs((TMP_PATH / TMP_FILE).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectorySubdirs((TMP_PATH / TMP_FILE1).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectorySubdirs((TMP_PATH).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(files.size(), 2);
}

TEST(LocalFileSystem, GetDirectoryFiles) {
    ovms::LocalFileSystem lfs;
    ovms::files_list_t files;
    createTmpFiles();

    auto status = lfs.getDirectoryFiles((TMP_PATH / TMP_FILE).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryFiles((TMP_PATH / TMP_FILE1).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);

    status = lfs.getDirectoryFiles((TMP_PATH).string(), &files);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(files.size(), 1);
}

TEST(LocalFileSystem, DownloadFileFolder) {
    ovms::LocalFileSystem lfs;
    std::string location;
    const std::filesystem::path TMP_DOWNLOAD = "download";
    auto status = lfs.downloadFileFolder((TMP_PATH / TMP_DOWNLOAD).string(), location);
    EXPECT_EQ(status, ovms::StatusCode::OK);
}

TEST(LocalFileSystem, DestroyFileFolder) {
    ovms::LocalFileSystem lfs;
    bool exists = false;
    auto status = lfs.fileExists((TMP_PATH / TMP_DIR1).string(), &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, true);
    status = lfs.deleteFileFolder((TMP_PATH / TMP_DIR1).string());
    EXPECT_EQ(status, ovms::StatusCode::OK);
    status = lfs.fileExists((TMP_PATH / TMP_DIR1).string(), &exists);
    EXPECT_EQ(status, ovms::StatusCode::OK);
    EXPECT_EQ(exists, false);
    status = lfs.deleteFileFolder((TMP_PATH / TMP_DIR1).string());
    EXPECT_EQ(status, ovms::StatusCode::PATH_INVALID);
}

#ifdef _WIN32
bool CanAccessFolder(LPCTSTR folderName, DWORD genericAccessRights) {
    bool bRet = false;
    DWORD length = 0;
    if (!::GetFileSecurity(folderName, OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, NULL, NULL, &length) && ERROR_INSUFFICIENT_BUFFER == ::GetLastError()) {
        PSECURITY_DESCRIPTOR security = static_cast<PSECURITY_DESCRIPTOR>(::malloc(length));
        if (security && ::GetFileSecurity(folderName, OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, security, length, &length)) {
            HANDLE hToken = NULL;
            if (::OpenProcessToken(::GetCurrentProcess(), TOKEN_IMPERSONATE | TOKEN_QUERY | TOKEN_DUPLICATE | STANDARD_RIGHTS_READ, &hToken)) {
                HANDLE hImpersonatedToken = NULL;
                if (::DuplicateToken(hToken, SecurityImpersonation, &hImpersonatedToken)) {
                    GENERIC_MAPPING mapping = {0xFFFFFFFF};
                    PRIVILEGE_SET privileges = {0};
                    DWORD grantedAccess = 0, privilegesLength = sizeof(privileges);
                    BOOL result = FALSE;

                    mapping.GenericRead = FILE_GENERIC_READ;
                    mapping.GenericWrite = FILE_GENERIC_WRITE;
                    mapping.GenericExecute = FILE_GENERIC_EXECUTE;
                    mapping.GenericAll = FILE_ALL_ACCESS;

                    ::MapGenericMask(&genericAccessRights, &mapping);
                    if (::AccessCheck(security, hImpersonatedToken, genericAccessRights, &mapping, &privileges, &privilegesLength, &grantedAccess, &result)) {
                        bRet = (result == TRUE);
                    }
                    ::CloseHandle(hImpersonatedToken);
                }
                ::CloseHandle(hToken);
            }
            ::free(security);
        }
    }
    return bRet;
}
#endif

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
    EXPECT_TRUE((p & fs::perms::owner_read) != fs::perms::none);
#ifdef __linux__
    EXPECT_TRUE((p & fs::perms::group_read) == fs::perms::none);
    EXPECT_TRUE((p & fs::perms::others_read) == fs::perms::none);
#elif _WIN32
    LPCTSTR lpsz = TEXT(local_path.c_str());
    EXPECT_TRUE(CanAccessFolder(lpsz, GENERIC_WRITE));
    EXPECT_TRUE(CanAccessFolder(lpsz, GENERIC_READ));
    EXPECT_FALSE(CanAccessFolder(lpsz, GENERIC_EXECUTE));
#endif

    fs::remove(local_path);
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
        return std::filesystem::weakly_canonical(std::filesystem::path(path)).string();
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
    std::string currentWorkingDir = std::filesystem::current_path().string();
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

#ifdef __linux__
    rootPath = "/rootPath";
#elif _WIN32
    rootPath = "C:\\rootPath";
#endif
    testPath = "";
    givenPath = "";

    ovms::FileSystem::setPath(testPath, givenPath, rootPath);
    ASSERT_EQ(testPath, rootPath);

    testPath = "";
#ifdef __linux__
    givenPath = "/givenPath";
#elif _WIN32
    givenPath = "C:\\givenPath";
#endif

    ovms::FileSystem::setPath(testPath, givenPath, rootPath);
#ifdef __linux__
    ASSERT_EQ(testPath, "/givenPath");
#elif _WIN32
    ASSERT_EQ(testPath, "C:\\givenPath");
#endif

    testPath = "";
    givenPath = "givenPath";

    ovms::FileSystem::setPath(testPath, givenPath, rootPath);
#ifdef __linux__
    ASSERT_EQ(testPath, "/rootPathgivenPath");
#elif _WIN32
    ASSERT_EQ(testPath, "C:\\rootPathgivenPath");
#endif

    testPath = "";
    givenPath = "long/givenPath";

    ovms::FileSystem::setPath(testPath, givenPath, rootPath);

#ifdef __linux__
    ASSERT_EQ(testPath, "/rootPathlong/givenPath");
#elif _WIN32
    ASSERT_EQ(testPath, "C:\\rootPathlong/givenPath");
#endif

    testPath = "";
    givenPath = "s3://long/givenPath";

    ovms::FileSystem::setPath(testPath, givenPath, rootPath);
    ASSERT_EQ(testPath, givenPath);
}

//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "platform_utils.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <unordered_map>

// Static map workaround for char* pointers as paths
const std::string& getPathFromMap(std::string inputPath, std::string outputPath) {
    static std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);
    static std::unordered_map<std::string, std::string> inputMap = {};
    auto it = inputMap.find(inputPath);
    if (it != inputMap.end()) {
        // element exists
        return inputMap.at(inputPath);
    } else {
        // element does not exist
        inputMap.emplace(inputPath, outputPath);
        return inputMap.at(inputPath);
    }
}

// Function changes linux docker container path /ovms/src/test/dummy to windows workspace "C:\git\model_server\src\test\dummy"
// Depending on the ovms_test.exe location after build
const std::string& getGenericFullPathForSrcTest(const std::string& linuxPath, bool logChange) {
#ifdef __linux__
    return getPathFromMap(linuxPath, linuxPath);
#elif _WIN32
    // For ovms_test cwd = C:\git\model_server\bazel-out\x64_windows-opt\bin\src
    std::filesystem::path cwd = std::filesystem::current_path();
    std::size_t bazelOutIndex = cwd.string().find("bazel-out");

    // Example linuxPath "/ovms/src/test/dummy"
    std::size_t postOvmsIndex = linuxPath.find("/src/test");
    if (postOvmsIndex != std::string::npos) {
        // Setting winPath to "/src/test/dummy"
        std::string winPath = linuxPath.substr(postOvmsIndex);
        // Set basePath to "C:\git\model_server\"
        std::string basePath = bazelOutIndex != std::string::npos ? cwd.string().substr(0, bazelOutIndex) : cwd.string();
        // Combine "C:\git\model_server\" + "/src/test/dummy"
        std::string finalWinPath = basePath + winPath;
        // Change paths to linux separator for JSON parser compatybility in configs
        std::replace(finalWinPath.begin(), finalWinPath.end(), '\\', '/');

        if (logChange) {
            std::cout << "[WINDOWS DEBUG] Changed path: " << linuxPath << " to path: " << finalWinPath << " for Windows" << std::endl;
        }
        return getPathFromMap(linuxPath, finalWinPath);
    }
#endif
    return getPathFromMap(linuxPath, linuxPath);
}

// Function changes linux docker container path /ovms/bazel-out/src/lib_node_mock.so to windows workspace "C:\git\model_server\bazel-bin\src\lib_node_mock.so"
// Depending on the ovms_test.exe location after build
const std::string& getGenericFullPathForBazelOut(const std::string& linuxPath, bool logChange) {
#ifdef __linux__
    return getPathFromMap(linuxPath, linuxPath);
#elif _WIN32
    // For ovms_test cwd = C:\git\model_server\bazel-out\x64_windows-opt\bin\src
    std::filesystem::path cwd = std::filesystem::current_path();
    std::size_t bazelOutIndex = cwd.string().find("bazel-out");

    // Example linuxPath "/ovms/bazel-bin/src/lib_node_mock.so"
    std::size_t postOvmsIndex = linuxPath.find("/bazel-bin/src");
    if (postOvmsIndex != std::string::npos) {
        // Setting winPath to "/bazel-bin/src"
        std::string winPath = linuxPath.substr(postOvmsIndex);
        // Set basePath to "C:\git\model_server\"
        std::string basePath = bazelOutIndex != std::string::npos ? cwd.string().substr(0, bazelOutIndex) : cwd.string();
        // Combine "C:\git\model_server\" + "/bazel-bin/src"
        std::string finalWinPath = basePath + winPath;
        // Change paths to linux separator for JSON parser compatybility in configs
        std::replace(finalWinPath.begin(), finalWinPath.end(), '\\', '/');

        if (logChange) {
            std::cout << "[WINDOWS DEBUG] Changed path: " << linuxPath << " to path: " << finalWinPath << " for Windows" << std::endl;
        }
        return getPathFromMap(linuxPath, finalWinPath);
    }
#endif
    return getPathFromMap(linuxPath, linuxPath);
}

const std::string& getGenericFullPathForSrcTest(const char* linuxPath, bool logChange) {
    return getGenericFullPathForSrcTest(std::string(linuxPath, strlen(linuxPath)), logChange);
}

// Function changes docker linux paths starting with /tmp: "/tmp/dummy" to windows C:\git\model_server\tmp\dummy
const std::string& getGenericFullPathForTmp(const std::string& linuxPath, bool logChange) {
#ifdef __linux__
    return getPathFromMap(linuxPath, linuxPath);
#elif _WIN32
    // For ovms_test cwd = C:\git\model_server\bazel-out\x64_windows-opt\bin\src
    std::filesystem::path cwd = std::filesystem::current_path();
    size_t bazelOutIndex = cwd.string().find("bazel-out");

    // Example linuxPath "/tmp/dummy"
    const std::string tmpString = "/tmp";
    const size_t tmpStringSize = 4;

    size_t postTmpIndex = linuxPath.find(tmpString) + tmpStringSize;
    if (postTmpIndex != std::string::npos) {
        std::string winPath = linuxPath.substr(postTmpIndex);
        // Set basePath to "C:\git\model_server\"
        std::string basePath = bazelOutIndex != std::string::npos ? cwd.string().substr(0, bazelOutIndex) : cwd.string();
        // Combine "C:\git\model_server\" + "tmp" "\dummy"
        std::string finalWinPath = basePath + tmpString + winPath;
        // Change paths to linux separator for JSON parser compatybility in configs
        std::replace(finalWinPath.begin(), finalWinPath.end(), '\\', '/');

        if (logChange) {
            std::cout << "[WINDOWS DEBUG] Changed path: " << linuxPath << " to path: " << finalWinPath << " for Windows" << std::endl;
        }
        return getPathFromMap(linuxPath, finalWinPath);
    }
#endif
    return getPathFromMap(linuxPath, linuxPath);
}

const std::string& getGenericFullPathForTmp(const char* linuxPath, bool logChange) {
    return getGenericFullPathForTmp(std::string(linuxPath, strlen(linuxPath)), logChange);
}

#ifdef _WIN32
const std::string getWindowsRepoRootPath() {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::size_t bazelOutIndex = cwd.string().find("bazel-out");
    std::string rootPath = cwd.string().substr(0, bazelOutIndex);
    std::replace(rootPath.begin(), rootPath.end(), '\\', '/');
    return rootPath;
}
#endif
// Apply necessary changes so the graph config will comply with the platform
// that tests are run on
void adjustConfigForTargetPlatform(std::string& input) {
#ifdef _WIN32
    std::string repoTestPath = getWindowsRepoRootPath() + "/src/test";
    std::string searchString = "\"/ovms/src/test";
    std::string replaceString = "\"" + repoTestPath;
    size_t pos = 0;
    while ((pos = input.find(searchString, pos)) != std::string::npos) {
        input.replace(pos, searchString.length(), replaceString);
        pos += replaceString.length();
    }

    repoTestPath = getWindowsRepoRootPath() + "/tmp";
    searchString = "\"/tmp";
    replaceString = "\"" + repoTestPath;
    pos = 0;
    while ((pos = input.find(searchString, pos)) != std::string::npos) {
        input.replace(pos, searchString.length(), replaceString);
        pos += replaceString.length();
    }

    repoTestPath = getWindowsRepoRootPath() + "/bazel-bin/src";
    searchString = "\"/ovms/bazel-bin/src";
    replaceString = "\"" + repoTestPath;
    pos = 0;
    while ((pos = input.find(searchString, pos)) != std::string::npos) {
        input.replace(pos, searchString.length(), replaceString);
        pos += replaceString.length();
    }
#elif __linux__
    // No changes needed for linux now, but keeping it as a placeholder
#endif
}

// Apply necessary changes so the graph config will comply with the platform
// that tests are run on
const std::string& adjustConfigForTargetPlatformReturn(std::string& input) {
    adjustConfigForTargetPlatform(input);
    return input;
}

std::string adjustConfigForTargetPlatformCStr(const char* input) {
    std::string inputString(input);
    adjustConfigForTargetPlatform(inputString);
    return inputString;
}
std::string getOvmsTestExecutablePath() {
#ifdef __linux__
    return std::filesystem::canonical("/proc/self/exe").string();
#elif _WIN32
    char buffer[2000];
    GetModuleFileNameA(NULL, buffer, 2000);
    return std::filesystem::path(buffer).parent_path().string();
#endif
}

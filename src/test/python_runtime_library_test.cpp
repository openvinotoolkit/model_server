//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef __linux__
#include <dlfcn.h>
#endif

using testing::HasSubstr;

namespace {

#ifdef __linux__
using ValidatePythonEnvironmentFn = bool (*)(const char** errorMessage);

class ScopedSharedLibrary {
    void* handle;

public:
    explicit ScopedSharedLibrary(const std::filesystem::path& path) :
        handle(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)) {
    }

    ~ScopedSharedLibrary() {
        if (handle != nullptr) {
            dlclose(handle);
        }
    }

    void* get() const {
        return handle;
    }
};

class ScopedEnvironmentVariable {
    std::string name;
    bool hadValue;
    std::string previousValue;

public:
    ScopedEnvironmentVariable(const std::string& name, const std::string& value) :
        name(name),
        hadValue(false) {
        if (const char* currentValue = std::getenv(name.c_str()); currentValue != nullptr) {
            hadValue = true;
            previousValue = currentValue;
        }
        setenv(name.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvironmentVariable() {
        if (hadValue) {
            setenv(name.c_str(), previousValue.c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }
};

std::filesystem::path findLibrary(const std::string& libName) {
    std::vector<std::filesystem::path> searchPaths;

    if (const char* testSrcDir = std::getenv("TEST_SRCDIR"); testSrcDir != nullptr && testSrcDir[0] != '\0') {
        std::filesystem::path srcDir(testSrcDir);
        if (const char* workspace = std::getenv("TEST_WORKSPACE"); workspace != nullptr && workspace[0] != '\0') {
            searchPaths.emplace_back(srcDir / workspace / "src/python" / libName);
            searchPaths.emplace_back(srcDir / workspace / "bazel-bin" / "src/python" / libName);
        }
        searchPaths.emplace_back(srcDir / "_main" / "src/python" / libName);
        searchPaths.emplace_back(srcDir / "_main" / "bazel-bin" / "src/python" / libName);
        searchPaths.emplace_back(srcDir / "model_server" / "src/python" / libName);
        searchPaths.emplace_back(srcDir / "model_server" / "bazel-bin" / "src/python" / libName);
    }

    try {
        const auto testBinaryPath = std::filesystem::canonical("/proc/self/exe");
        const auto runfilesDir = testBinaryPath.string() + ".runfiles";
        searchPaths.emplace_back(std::filesystem::path(runfilesDir) / "src/python" / libName);
        searchPaths.emplace_back(std::filesystem::path(runfilesDir) / "_main" / "src/python" / libName);
        searchPaths.emplace_back(std::filesystem::path(runfilesDir) / "model_server" / "src/python" / libName);
    } catch (...) {
    }

    searchPaths.emplace_back(std::filesystem::path("bazel-bin/src/python") / libName);
    searchPaths.emplace_back(std::filesystem::path("src/python") / libName);
    searchPaths.emplace_back(libName);

    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

    return {};
}

std::filesystem::path findPyovmsBinding() {
    std::vector<std::filesystem::path> searchPaths;

    if (const char* testSrcDir = std::getenv("TEST_SRCDIR"); testSrcDir != nullptr && testSrcDir[0] != '\0') {
        std::filesystem::path srcDir(testSrcDir);
        if (const char* workspace = std::getenv("TEST_WORKSPACE"); workspace != nullptr && workspace[0] != '\0') {
            searchPaths.emplace_back(srcDir / workspace / "src/python/binding" / "pyovms.so");
            searchPaths.emplace_back(srcDir / workspace / "bazel-bin" / "src/python/binding" / "pyovms.so");
        }
        searchPaths.emplace_back(srcDir / "_main" / "src/python/binding" / "pyovms.so");
        searchPaths.emplace_back(srcDir / "_main" / "bazel-bin" / "src/python/binding" / "pyovms.so");
        searchPaths.emplace_back(srcDir / "model_server" / "src/python/binding" / "pyovms.so");
        searchPaths.emplace_back(srcDir / "model_server" / "bazel-bin" / "src/python/binding" / "pyovms.so");
    }

    try {
        const auto testBinaryPath = std::filesystem::canonical("/proc/self/exe");
        const auto runfilesDir = testBinaryPath.string() + ".runfiles";
        searchPaths.emplace_back(std::filesystem::path(runfilesDir) / "src/python/binding" / "pyovms.so");
        searchPaths.emplace_back(std::filesystem::path(runfilesDir) / "_main" / "src/python/binding" / "pyovms.so");
        searchPaths.emplace_back(std::filesystem::path(runfilesDir) / "model_server" / "src/python/binding" / "pyovms.so");
    } catch (...) {
    }

    searchPaths.emplace_back(std::filesystem::path("bazel-bin/src/python/binding/pyovms.so"));
    searchPaths.emplace_back(std::filesystem::path("src/python/binding/pyovms.so"));
    searchPaths.emplace_back("pyovms.so");

    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

    return {};
}
#endif

}  // namespace

TEST(PythonRuntimeLibrary, MissingLibraryPathFailsToLoad) {
#ifndef __linux__
    GTEST_SKIP() << "Linux-only libovmspython.so test";
#else
    const auto missingLibraryPath = std::filesystem::temp_directory_path() / "missing_libovmspython.so";
    ASSERT_FALSE(std::filesystem::exists(missingLibraryPath));

    ScopedSharedLibrary library(missingLibraryPath);

    EXPECT_EQ(library.get(), nullptr);
#endif
}

TEST(PythonRuntimeLibrary, ExistingLibraryExportsRequiredSymbols) {
#ifndef __linux__
    GTEST_SKIP() << "Linux-only libovmspython.so test";
#else
    const auto libraryPath = findLibrary("libovmspython.so");
    ASSERT_FALSE(libraryPath.empty()) << "Could not find libovmspython.so";

    ScopedSharedLibrary library(libraryPath);
    ASSERT_NE(library.get(), nullptr) << dlerror();

    EXPECT_NE(dlsym(library.get(), "OVMS_createPythonInterpreterModule"), nullptr);
    EXPECT_NE(dlsym(library.get(), "OVMS_validatePythonEnvironment"), nullptr);
#endif
}

TEST(PythonRuntimeLibrary, ValidationFailsWithoutBindingOnPythonPath) {
#ifndef __linux__
    GTEST_SKIP() << "Linux-only libovmspython.so test";
#else
    const auto libraryPath = findLibrary("libovmspython.so");
    ASSERT_FALSE(libraryPath.empty()) << "Could not find libovmspython.so";

    ScopedSharedLibrary library(libraryPath);
    ASSERT_NE(library.get(), nullptr) << dlerror();

    const auto emptyPythonPath = std::filesystem::temp_directory_path() / "ovms_empty_pythonpath";
    std::filesystem::create_directories(emptyPythonPath);
    ScopedEnvironmentVariable pythonPathEnv("PYTHONPATH", emptyPythonPath.string());

    auto validate = reinterpret_cast<ValidatePythonEnvironmentFn>(dlsym(library.get(), "OVMS_validatePythonEnvironment"));
    ASSERT_NE(validate, nullptr);

    const char* errorMessage = nullptr;
    EXPECT_FALSE(validate(&errorMessage));
    ASSERT_NE(errorMessage, nullptr);
    EXPECT_THAT(std::string(errorMessage), HasSubstr("pyovms"));
#endif
}

TEST(PythonRuntimeLibrary, ValidationSucceedsWithBindingOnPythonPath) {
#ifndef __linux__
    GTEST_SKIP() << "Linux-only libovmspython.so test";
#else
    const auto libraryPath = findLibrary("libovmspython.so");
    ASSERT_FALSE(libraryPath.empty()) << "Could not find libovmspython.so";
    const auto bindingPath = findPyovmsBinding();
    ASSERT_FALSE(bindingPath.empty()) << "Could not find pyovms.so";

    ScopedSharedLibrary library(libraryPath);
    ASSERT_NE(library.get(), nullptr) << dlerror();

    ScopedEnvironmentVariable pythonPathEnv("PYTHONPATH", bindingPath.parent_path().string());

    auto validate = reinterpret_cast<ValidatePythonEnvironmentFn>(dlsym(library.get(), "OVMS_validatePythonEnvironment"));
    ASSERT_NE(validate, nullptr);

    const char* errorMessage = nullptr;
    EXPECT_TRUE(validate(&errorMessage));
    EXPECT_EQ(errorMessage, nullptr);
#endif
}
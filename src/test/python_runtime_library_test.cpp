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
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef __linux__
#include <dlfcn.h>
#endif

using testing::HasSubstr;

namespace {

std::filesystem::path getExecutablePath() {
#ifdef __linux__
    return std::filesystem::canonical("/proc/self/exe");
#elif _WIN32
    return std::filesystem::current_path();
#endif
}

std::vector<std::filesystem::path> getRunfilesRoots() {
    std::vector<std::filesystem::path> roots;
    if (const char* testSrcDir = std::getenv("TEST_SRCDIR"); testSrcDir != nullptr && testSrcDir[0] != '\0') {
        roots.emplace_back(testSrcDir);
    }
    const auto executablePath = getExecutablePath();
    roots.emplace_back(executablePath.string() + ".runfiles");
    return roots;
}

std::vector<std::string> getWorkspacePrefixes() {
    std::vector<std::string> prefixes;
    if (const char* workspace = std::getenv("TEST_WORKSPACE"); workspace != nullptr && workspace[0] != '\0') {
        prefixes.emplace_back(workspace);
    }
    prefixes.emplace_back("_main");
    prefixes.emplace_back("model_server");
    return prefixes;
}

std::filesystem::path findRunfile(const std::filesystem::path& relativePath) {
    for (const auto& root : getRunfilesRoots()) {
        for (const auto& prefix : getWorkspacePrefixes()) {
            const auto candidate = root / prefix / relativePath;
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }
        const auto directCandidate = root / relativePath;
        if (std::filesystem::exists(directCandidate)) {
            return directCandidate;
        }
    }
    return {};
}

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
#ifdef __linux__
        setenv(name.c_str(), value.c_str(), 1);
#elif _WIN32
        _putenv_s(name.c_str(), value.c_str());
#endif
    }

    ~ScopedEnvironmentVariable() {
        if (hadValue) {
#ifdef __linux__
            setenv(name.c_str(), previousValue.c_str(), 1);
#elif _WIN32
            _putenv_s(name.c_str(), previousValue.c_str());
#endif
        } else {
#ifdef __linux__
            unsetenv(name.c_str());
#elif _WIN32
            _putenv_s(name.c_str(), "");
#endif
        }
    }
};

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

int runIsolatedOvmsTest(const std::string& gtestFilter, const std::string& pythonPath = "") {
    const auto executablePath = getExecutablePath();
    const std::string filterArgument = "--gtest_filter=" + gtestFilter;
    pid_t pid = fork();
    if (pid == 0) {
        setenv("OVMS_SKIP_GLOBAL_PYTHON_ENVIRONMENT", "1", 1);
        unsetenv("TEST_PREMATURE_EXIT_FILE");
        if (!pythonPath.empty()) {
            setenv("PYTHONPATH", pythonPath.c_str(), 1);
        }
        execl(executablePath.c_str(), executablePath.c_str(), filterArgument.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (!WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
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
    const auto libraryPath = findRunfile("src/python/libovmspython.so");
    ASSERT_FALSE(libraryPath.empty());

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
    const auto emptyPythonPath = std::filesystem::temp_directory_path() / "ovms_empty_pythonpath";
    std::filesystem::create_directories(emptyPythonPath);
    EXPECT_EQ(runIsolatedOvmsTest("PythonRuntimeLibraryIsolated.ValidationFailsWithoutBindingOnPythonPath", emptyPythonPath.string()), 0);
#endif
}

TEST(PythonRuntimeLibrary, ValidationSucceedsWithBindingOnPythonPath) {
#ifndef __linux__
    GTEST_SKIP() << "Linux-only libovmspython.so test";
#else
    const auto bindingPath = findRunfile("src/python/binding/pyovms.so");
    ASSERT_FALSE(bindingPath.empty());
    EXPECT_EQ(runIsolatedOvmsTest("PythonRuntimeLibraryIsolated.ValidationSucceedsWithBindingOnPythonPath", bindingPath.parent_path().string()), 0);
#endif
}

TEST(PythonRuntimeLibraryIsolated, ValidationFailsWithoutBindingOnPythonPath) {
#ifndef __linux__
    GTEST_SKIP() << "Linux-only libovmspython.so test";
#else
    const auto libraryPath = findRunfile("src/python/libovmspython.so");
    ASSERT_FALSE(libraryPath.empty());
    ASSERT_EQ(std::getenv("OVMS_SKIP_GLOBAL_PYTHON_ENVIRONMENT"), std::string("1"));

    ScopedSharedLibrary library(libraryPath);
    ASSERT_NE(library.get(), nullptr) << dlerror();

    auto validate = reinterpret_cast<ValidatePythonEnvironmentFn>(dlsym(library.get(), "OVMS_validatePythonEnvironment"));
    ASSERT_NE(validate, nullptr);

    const char* errorMessage = nullptr;
    EXPECT_FALSE(validate(&errorMessage));
    ASSERT_NE(errorMessage, nullptr);
    EXPECT_THAT(std::string(errorMessage), HasSubstr("pyovms"));
#endif
}

TEST(PythonRuntimeLibraryIsolated, ValidationSucceedsWithBindingOnPythonPath) {
#ifndef __linux__
    GTEST_SKIP() << "Linux-only libovmspython.so test";
#else
    const auto libraryPath = findRunfile("src/python/libovmspython.so");
    ASSERT_FALSE(libraryPath.empty());
    ASSERT_EQ(std::getenv("OVMS_SKIP_GLOBAL_PYTHON_ENVIRONMENT"), std::string("1"));

    ScopedSharedLibrary library(libraryPath);
    ASSERT_NE(library.get(), nullptr) << dlerror();

    auto validate = reinterpret_cast<ValidatePythonEnvironmentFn>(dlsym(library.get(), "OVMS_validatePythonEnvironment"));
    ASSERT_NE(validate, nullptr);

    const char* errorMessage = nullptr;
    EXPECT_TRUE(validate(&errorMessage));
    EXPECT_EQ(errorMessage, nullptr);
#endif
}
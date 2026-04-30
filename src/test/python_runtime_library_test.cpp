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

#ifdef _WIN32
#include <windows.h>
#endif

using testing::HasSubstr;

namespace {

using ValidatePythonEnvironmentFn = bool (*)(const char** errorMessage);

class ScopedSharedLibrary {
public:
#ifdef _WIN32
    using HandleType = HMODULE;
#else
    using HandleType = void*;
#endif

private:
    HandleType handle;

public:
    explicit ScopedSharedLibrary(const std::filesystem::path& path) :
        handle(
#ifdef _WIN32
            LoadLibraryA(path.string().c_str())
#else
            dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)
#endif
        ) {
    }

    ~ScopedSharedLibrary() {
        if (handle != nullptr) {
#ifdef _WIN32
            FreeLibrary(handle);
#else
            dlclose(handle);
#endif
        }
    }

    HandleType get() const {
        return handle;
    }
};

const char* getLibraryLoadError() {
#ifdef _WIN32
    return "LoadLibrary failed";
#else
    const char* error = dlerror();
    return error == nullptr ? "dlopen failed" : error;
#endif
}

void* findSymbol(ScopedSharedLibrary::HandleType handle, const char* symbolName) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(handle, symbolName));
#else
    return dlsym(handle, symbolName);
#endif
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
#ifdef _WIN32
        _putenv_s(name.c_str(), value.c_str());
#else
        setenv(name.c_str(), value.c_str(), 1);
#endif
    }

    ~ScopedEnvironmentVariable() {
        if (hadValue) {
#ifdef _WIN32
            _putenv_s(name.c_str(), previousValue.c_str());
#else
            setenv(name.c_str(), previousValue.c_str(), 1);
#endif
        } else {
#ifdef _WIN32
            _putenv_s(name.c_str(), "");
#else
            unsetenv(name.c_str());
#endif
        }
    }
};

std::string getRuntimeLibraryFilename() {
#ifdef _WIN32
    return "libovmspython.dll";
#else
    return "libovmspython.so";
#endif
}

std::string getBindingFilename() {
#ifdef _WIN32
    return "pyovms.pyd";
#else
    return "pyovms.so";
#endif
}

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
        const std::string bindingFilename = getBindingFilename();
        if (const char* workspace = std::getenv("TEST_WORKSPACE"); workspace != nullptr && workspace[0] != '\0') {
            searchPaths.emplace_back(srcDir / workspace / "src/python/binding" / bindingFilename);
            searchPaths.emplace_back(srcDir / workspace / "bazel-bin" / "src/python/binding" / bindingFilename);
        }
        searchPaths.emplace_back(srcDir / "_main" / "src/python/binding" / bindingFilename);
        searchPaths.emplace_back(srcDir / "_main" / "bazel-bin" / "src/python/binding" / bindingFilename);
        searchPaths.emplace_back(srcDir / "model_server" / "src/python/binding" / bindingFilename);
        searchPaths.emplace_back(srcDir / "model_server" / "bazel-bin" / "src/python/binding" / bindingFilename);
    }

    try {
        const std::string bindingFilename = getBindingFilename();
        const auto testBinaryPath = std::filesystem::canonical("/proc/self/exe");
        const auto runfilesDir = testBinaryPath.string() + ".runfiles";
        searchPaths.emplace_back(std::filesystem::path(runfilesDir) / "src/python/binding" / bindingFilename);
        searchPaths.emplace_back(std::filesystem::path(runfilesDir) / "_main" / "src/python/binding" / bindingFilename);
        searchPaths.emplace_back(std::filesystem::path(runfilesDir) / "model_server" / "src/python/binding" / bindingFilename);
    } catch (...) {
    }

    const std::string bindingFilename = getBindingFilename();
    searchPaths.emplace_back(std::filesystem::path("bazel-bin/src/python/binding") / bindingFilename);
    searchPaths.emplace_back(std::filesystem::path("src/python/binding") / bindingFilename);
    searchPaths.emplace_back(bindingFilename);

    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

    return {};
}

}  // namespace

TEST(PythonRuntimeLibrary, MissingLibraryPathFailsToLoad) {
    const auto missingLibraryPath = std::filesystem::temp_directory_path() / ("missing_" + getRuntimeLibraryFilename());
    ASSERT_FALSE(std::filesystem::exists(missingLibraryPath));

    ScopedSharedLibrary library(missingLibraryPath);

    EXPECT_EQ(library.get(), nullptr);
}

TEST(PythonRuntimeLibrary, ExistingLibraryExportsRequiredSymbols) {
    const auto runtimeLibraryFilename = getRuntimeLibraryFilename();
    const auto libraryPath = findLibrary(runtimeLibraryFilename);
    ASSERT_FALSE(libraryPath.empty()) << "Could not find " << runtimeLibraryFilename;

    ScopedSharedLibrary library(libraryPath);
    ASSERT_NE(library.get(), nullptr) << getLibraryLoadError();

    EXPECT_NE(findSymbol(library.get(), "OVMS_createPythonInterpreterModule"), nullptr);
    EXPECT_NE(findSymbol(library.get(), "OVMS_validatePythonEnvironment"), nullptr);
}

TEST(PythonRuntimeLibrary, ValidationFailsWithoutBindingOnPythonPath) {
    const auto runtimeLibraryFilename = getRuntimeLibraryFilename();
    const auto libraryPath = findLibrary(runtimeLibraryFilename);
    ASSERT_FALSE(libraryPath.empty()) << "Could not find " << runtimeLibraryFilename;

    ScopedSharedLibrary library(libraryPath);
    ASSERT_NE(library.get(), nullptr) << getLibraryLoadError();

    const auto emptyPythonPath = std::filesystem::temp_directory_path() / "ovms_empty_pythonpath";
    std::filesystem::create_directories(emptyPythonPath);
    ScopedEnvironmentVariable pythonPathEnv("PYTHONPATH", emptyPythonPath.string());

    auto validate = reinterpret_cast<ValidatePythonEnvironmentFn>(findSymbol(library.get(), "OVMS_validatePythonEnvironment"));
    ASSERT_NE(validate, nullptr);

    const char* errorMessage = nullptr;
    EXPECT_FALSE(validate(&errorMessage));
    ASSERT_NE(errorMessage, nullptr);
    EXPECT_THAT(std::string(errorMessage), HasSubstr("pyovms"));
}

TEST(PythonRuntimeLibrary, ValidationSucceedsWithBindingOnPythonPath) {
    const auto runtimeLibraryFilename = getRuntimeLibraryFilename();
    const auto libraryPath = findLibrary(runtimeLibraryFilename);
    ASSERT_FALSE(libraryPath.empty()) << "Could not find " << runtimeLibraryFilename;
    const auto bindingPath = findPyovmsBinding();
    ASSERT_FALSE(bindingPath.empty()) << "Could not find " << getBindingFilename();

    ScopedSharedLibrary library(libraryPath);
    ASSERT_NE(library.get(), nullptr) << getLibraryLoadError();

    ScopedEnvironmentVariable pythonPathEnv("PYTHONPATH", bindingPath.parent_path().string());

    auto validate = reinterpret_cast<ValidatePythonEnvironmentFn>(findSymbol(library.get(), "OVMS_validatePythonEnvironment"));
    ASSERT_NE(validate, nullptr);

    const char* errorMessage = nullptr;
    EXPECT_TRUE(validate(&errorMessage));
    EXPECT_EQ(errorMessage, nullptr);
}

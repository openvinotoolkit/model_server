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

#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "src/config.hpp"
#include "src/python/pythoninterpretermodule.hpp"
#include "src/status.hpp"

namespace ovms {
bool loadPythonCalculatorsPlugin() {
    return true;
}
}  // namespace ovms

namespace {

struct LifecycleTestConfig : ovms::Config {};

class ScopedPythonPath {
    bool hadPreviousValue = false;
    std::string previousValue;

public:
    bool addBindingRunfile() {
        std::vector<std::filesystem::path> bindingDirectories;

        if (const char* pythonPath = std::getenv("PYTHONPATH"); pythonPath != nullptr && pythonPath[0] != '\0') {
            std::string pathCopy(pythonPath);
            std::size_t start = 0;
            while (start <= pathCopy.size()) {
                const std::size_t sep = pathCopy.find_first_of(";:", start);
                const std::string entry = (sep == std::string::npos) ? pathCopy.substr(start) : pathCopy.substr(start, sep - start);
                if (!entry.empty()) {
                    bindingDirectories.emplace_back(entry);
                }
                if (sep == std::string::npos) {
                    break;
                }
                start = sep + 1;
            }
        }

        if (const char* testSrcDir = std::getenv("TEST_SRCDIR"); testSrcDir != nullptr && testSrcDir[0] != '\0') {
            const char* testWorkspace = std::getenv("TEST_WORKSPACE");
            std::filesystem::path runfilesRoot(testSrcDir);
            for (std::filesystem::path current = runfilesRoot; !current.empty(); current = current.parent_path()) {
                if (testWorkspace != nullptr && testWorkspace[0] != '\0') {
                    bindingDirectories.emplace_back(current / testWorkspace / "src/python/binding");
                }
                bindingDirectories.emplace_back(current / "bazel-bin" / "src/python/binding");
                bindingDirectories.emplace_back(current / "bazel-out" / "x64_windows-opt" / "bin" / "src/python/binding");
                bindingDirectories.emplace_back(current / "_main" / "src/python/binding");
                bindingDirectories.emplace_back(current / "model_server" / "src/python/binding");
                if (current == current.parent_path()) {
                    break;
                }
            }
        }

        const auto currentDir = std::filesystem::current_path();
        bindingDirectories.emplace_back(currentDir / "src/python/binding");
        bindingDirectories.emplace_back(currentDir / "bazel-bin/src/python/binding");
        bindingDirectories.emplace_back(currentDir / "bazel-out" / "x64_windows-opt" / "bin" / "src/python/binding");

#ifdef _WIN32
        const char* extension = "pyovms.pyd";
        const char separator = ';';
#else
        const char* extension = "pyovms.so";
        const char separator = ':';
#endif

        std::filesystem::path bindingDirectory;
        for (const auto& candidate : bindingDirectories) {
            if (std::filesystem::exists(candidate / extension)) {
                bindingDirectory = candidate;
                break;
            }
        }
        if (bindingDirectory.empty()) {
            return false;
        }

        const char* currentPythonPath = std::getenv("PYTHONPATH");
        if (currentPythonPath != nullptr) {
            hadPreviousValue = true;
            previousValue = currentPythonPath;
        }
        std::string pythonPath = bindingDirectory.string();
        if (currentPythonPath != nullptr && currentPythonPath[0] != '\0') {
            pythonPath += separator;
            pythonPath += currentPythonPath;
        }
#ifdef _WIN32
        _putenv_s("PYTHONPATH", pythonPath.c_str());
#else
        setenv("PYTHONPATH", pythonPath.c_str(), 1);
#endif
        return true;
    }

    ~ScopedPythonPath() {
#ifdef _WIN32
        _putenv_s("PYTHONPATH", hadPreviousValue ? previousValue.c_str() : "");
#else
        if (hadPreviousValue) {
            setenv("PYTHONPATH", previousValue.c_str(), 1);
        } else {
            unsetenv("PYTHONPATH");
        }
#endif
    }
};

}  // namespace

// Python state is process-wide: an externally initialized interpreter is borrowed, never initialized or finalized by this module.
TEST(PythonInterpreterModuleBorrowedInterpreter, DoesNotOwnOrFinalizeExternalInterpreter) {
    ASSERT_FALSE(Py_IsInitialized());
    ScopedPythonPath pythonPath;
    ASSERT_TRUE(pythonPath.addBindingRunfile()) << "Could not locate the pyovms runfile";

    py::initialize_interpreter();
    LifecycleTestConfig config;
    ovms::Status startStatus;
    bool moduleOwnsInterpreter = true;
    bool backendCreated = false;
    bool backendDestroyed = false;
    {
        // Release the initializer thread's GIL while the module runs on its lifecycle thread.
        py::gil_scoped_release releaseGIL;
        ovms::PythonInterpreterModule pythonModule;
        startStatus = pythonModule.start(config);
        moduleOwnsInterpreter = pythonModule.ownsPythonInterpreter();
        backendCreated = pythonModule.getPythonBackend() != nullptr;
        pythonModule.shutdown();
        backendDestroyed = pythonModule.getPythonBackend() == nullptr;
    }

    EXPECT_TRUE(Py_IsInitialized());
    EXPECT_EQ(startStatus.getCode(), ovms::StatusCode::OK);
    EXPECT_FALSE(moduleOwnsInterpreter);
    EXPECT_TRUE(backendCreated);
    EXPECT_TRUE(backendDestroyed);
    py::finalize_interpreter();
    EXPECT_FALSE(Py_IsInitialized());
}

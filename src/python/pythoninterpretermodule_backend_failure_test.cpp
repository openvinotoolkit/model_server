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

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

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
    std::filesystem::path directory;
    bool hadPreviousValue = false;
    std::string previousValue;

public:
    explicit ScopedPythonPath(std::filesystem::path directory) :
        directory(std::move(directory)) {
        if (const char* currentValue = std::getenv("PYTHONPATH"); currentValue != nullptr) {
            hadPreviousValue = true;
            previousValue = currentValue;
        }
#ifdef _WIN32
        _putenv_s("PYTHONPATH", this->directory.string().c_str());
#else
        setenv("PYTHONPATH", this->directory.string().c_str(), 1);
#endif
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

class ScopedTempDirectory {
    std::filesystem::path path;

public:
    ScopedTempDirectory() {
        const auto uniqueValue = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("ovms-python-backend-failure-" + std::to_string(uniqueValue));
        std::filesystem::create_directories(path);
    }

    ~ScopedTempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    const std::filesystem::path& get() const {
        return path;
    }
};

}  // namespace

// Keep this failure scenario in its own binary because interpreter reinitialization is unsupported within one process.
// Shadow pyovms so backend creation fails after Python initialization succeeds.
TEST(PythonInterpreterModuleStartupFailure, MissingBackendModuleReturnsSpecificStatusAndFinalizes) {
    ASSERT_FALSE(Py_IsInitialized());

    ScopedTempDirectory tempDirectory;
    {
        std::ofstream shadowModule(tempDirectory.get() / "pyovms.py");
        ASSERT_TRUE(shadowModule.is_open());
        shadowModule << "raise ImportError('forced PythonBackend initialization failure')\n";
    }
    ScopedPythonPath pythonPath(tempDirectory.get());

    LifecycleTestConfig config;
    ovms::PythonInterpreterModule pythonModule;
    const ovms::Status status = pythonModule.start(config);

    EXPECT_EQ(status.getCode(), ovms::StatusCode::PYTHON_BACKEND_CREATION_FAILED);
    EXPECT_EQ(pythonModule.getPythonBackend(), nullptr);
    EXPECT_TRUE(pythonModule.ownsPythonInterpreter());
    EXPECT_FALSE(Py_IsInitialized());
}

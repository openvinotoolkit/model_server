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
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "src/python/python_runtime_path.hpp"

namespace py = pybind11;

namespace {

class PythonRuntimePath : public ::testing::Test {
protected:
    py::scoped_interpreter interpreter;
    const std::filesystem::path packageRoot = std::filesystem::temp_directory_path() /
                                              ("ovms_python_path_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const std::filesystem::path executablePath = packageRoot / "bin/ovms";
    const std::filesystem::path pythonDirectory = packageRoot / "lib/python";

    void TearDown() override {
        std::filesystem::remove_all(packageRoot);
    }
};

TEST_F(PythonRuntimePath, MissingDirectoryPreservesSearchPaths) {
    const py::list paths = py::module_::import("sys").attr("path");
    const std::string before = py::repr(paths);

    ovms::addPackagedPythonPath(executablePath);

    EXPECT_EQ(std::string(py::repr(paths)), before);
}

TEST_F(PythonRuntimePath, AppendsAbsolutePathWithoutDuplicatesOrChangingPrecedence) {
    std::filesystem::create_directories(pythonDirectory);
    py::list paths = py::module_::import("sys").attr("path");
    const size_t originalSize = py::len(paths);
    const std::string before = py::repr(paths);

    ovms::addPackagedPythonPath(executablePath);
    ovms::addPackagedPythonPath(executablePath);

    ASSERT_EQ(py::len(paths), originalSize + 1);
    EXPECT_EQ(paths[originalSize].cast<std::string>(), pythonDirectory.string());
    paths.attr("pop")();
    EXPECT_EQ(std::string(py::repr(paths)), before);
}

TEST_F(PythonRuntimePath, ImportsModulesAfterInterpreterReinitialization) {
    std::filesystem::create_directories(pythonDirectory);
    {
        std::ofstream module(pythonDirectory / "ovms_packaged_path_probe.py");
        ASSERT_TRUE(module.is_open());
        module << "value = 42\n";
        module.close();
        ASSERT_FALSE(module.fail());
    }

    for (int iteration = 0; iteration < 2; ++iteration) {
        if (iteration != 0) {
            py::finalize_interpreter();
            py::initialize_interpreter();
        }
        ovms::addPackagedPythonPath(executablePath);
        EXPECT_EQ(py::module_::import("ovms_packaged_path_probe").attr("value").cast<int>(), 42);
    }
}

#ifdef __linux__
TEST_F(PythonRuntimePath, DefaultPathIsRelativeToResolvedExecutable) {
    const auto resolvedExecutable = std::filesystem::canonical("/proc/self/exe");
    const auto expectedDirectory = (resolvedExecutable.parent_path() / "../lib/python").lexically_normal();
    const py::list paths = py::module_::import("sys").attr("path");
    const size_t originalSize = py::len(paths);
    const bool alreadyPresent = paths.contains(py::str(expectedDirectory.string()));

    ovms::addPackagedPythonPath();

    if (std::filesystem::is_directory(expectedDirectory)) {
        EXPECT_TRUE(paths.contains(py::str(expectedDirectory.string())));
        EXPECT_EQ(py::len(paths), originalSize + (alreadyPresent ? 0 : 1));
    } else {
        EXPECT_EQ(py::len(paths), originalSize);
    }
}
#endif

}  // namespace

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

#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include "src/config.hpp"
#include "src/python/pythoninterpretermodule.hpp"
#include "src/status.hpp"
#include "src/test/test_with_temp_dir.hpp"
#include "src/utils/env_guard.hpp"

namespace ovms {
bool loadPythonCalculatorsPlugin() {
    return true;
}
}  // namespace ovms

namespace {

struct LifecycleTestConfig : ovms::Config {};

class PythonInterpreterModuleBackendFailureTest : public TestWithTempDir {};

}  // namespace

// Keep this failure scenario in its own binary because interpreter reinitialization is unsupported within one process.
// Shadow pyovms so backend creation fails after Python initialization succeeds.
TEST_F(PythonInterpreterModuleBackendFailureTest, MissingBackendModuleReturnsSpecificStatusAndFinalizes) {
    ASSERT_FALSE(Py_IsInitialized());

    {
        std::ofstream shadowModule(directoryPath + "/pyovms.py");
        ASSERT_TRUE(shadowModule.is_open());
        shadowModule << "raise ImportError('forced PythonBackend initialization failure')\n";
    }
    EnvGuard pythonPath;
    pythonPath.set("PYTHONPATH", directoryPath);

    LifecycleTestConfig config;
    ovms::PythonInterpreterModule pythonModule;
    const ovms::Status status = pythonModule.start(config);

    EXPECT_EQ(status.getCode(), ovms::StatusCode::PYTHON_BACKEND_CREATION_FAILED);
    EXPECT_EQ(pythonModule.getPythonBackend(), nullptr);
    EXPECT_TRUE(pythonModule.ownsPythonInterpreter());
    EXPECT_FALSE(Py_IsInitialized());
}

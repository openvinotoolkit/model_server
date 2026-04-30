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

#include <cstdlib>

#include "environment.hpp"
#include "gpuenvironment.hpp"
#include "gguf_environment.hpp"
#include "python_environment.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new Environment);
    ::testing::AddGlobalTestEnvironment(new GPUEnvironment);
    ::testing::AddGlobalTestEnvironment(new GGUFEnvironment);
    if (const char* skipPythonEnvironment = std::getenv("OVMS_SKIP_GLOBAL_PYTHON_ENVIRONMENT");
        skipPythonEnvironment == nullptr || std::string(skipPythonEnvironment) != "1") {
        ::testing::AddGlobalTestEnvironment(new PythonEnvironment);
    }
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    return RUN_ALL_TESTS();
}

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

#include <string>
#include <cstdlib>

namespace {

// GTest death tests run assertions in a dedicated child process launched with
// an internal argument. Detect that child to skip PythonEnvironment setup,
// which avoids teardown-time crashes in exit-based death-test flows.
bool isDeathTestSubprocess(int argc, char** argv) {
    const char* deathTestEnv = std::getenv("GTEST_INTERNAL_RUN_DEATH_TEST");
    if (deathTestEnv != nullptr && deathTestEnv[0] != '\0') {
        return true;
    }

    for (int i = 0; i < argc; ++i) {
        if (argv[i] == nullptr) {
            continue;
        }
        if (std::string(argv[i]).find("gtest_internal_run_death_test") != std::string::npos) {
            return true;
        }
    }
    return false;
}

}  // namespace

#include "environment.hpp"
#include "gpuenvironment.hpp"
#include "gguf_environment.hpp"
#include "python_environment.hpp"

int main(int argc, char** argv) {
    // Check before InitGoogleTest because it can consume/rewrite internal args.
    const bool deathTestSubprocess = isDeathTestSubprocess(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);

    // Keep death-test subprocesses minimal to avoid teardown-time side effects.
    ::testing::AddGlobalTestEnvironment(new Environment);
    if (!deathTestSubprocess) {
        ::testing::AddGlobalTestEnvironment(new GPUEnvironment);
        ::testing::AddGlobalTestEnvironment(new GGUFEnvironment);
        ::testing::AddGlobalTestEnvironment(new PythonEnvironment);
    }
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    return RUN_ALL_TESTS();
}

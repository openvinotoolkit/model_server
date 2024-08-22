//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include "gpuenvironment.hpp"

#include <string>

#include "../logging.hpp"

void GPUEnvironment::SetUp() {
    const char* runGpuTestsEnv = std::getenv("RUN_GPU_TESTS");
    if (runGpuTestsEnv) {
        std::string runGpuTestsEnvContent(runGpuTestsEnv);
        if (runGpuTestsEnvContent == "ON") {
            GPUEnvironment::gpuTestsEnabled = true;
            SPDLOG_INFO("RUN_GPU_TESTS was set to ON. Will run tests requiring GPU");
        } else {
            SPDLOG_ERROR("Tests on GPU will be skipped since RUN_GPU_TESTS env variable was not set to ON. It was set to: {}", runGpuTestsEnvContent);
        }
    } else {
        SPDLOG_INFO("Tests on GPU will be skipped since RUN_GPU_TESTS env variable was not set to ON. Remember to use bazel test parameter --test_env when triggering tests using bazel.");
    }
}

void GPUEnvironment::skipWithoutGPU() {
    if (!GPUEnvironment::gpuTestsEnabled) {
        GTEST_SKIP() << "Skipping GPU tests because those tests were not enabled. Check environment.hpp";
    }
}

bool GPUEnvironment::gpuTestsEnabled{false};

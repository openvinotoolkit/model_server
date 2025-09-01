//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "gguf_environment.hpp"

#include <string>

#include "../logging.hpp"

void GGUFEnvironment::SetUp() {
    const char* runGGUFTestsEnv = std::getenv("RUN_GGUF_TESTS");
    if (runGGUFTestsEnv) {
        std::string runGGUFTestsEnvContent(runGGUFTestsEnv);
        if (runGGUFTestsEnvContent == "1") {
            GGUFEnvironment::ggufTestsEnabled = true;
            SPDLOG_INFO("RUN_GGUF_TESTS was set to 1. Will run longer GGUF tests");
        } else {
            SPDLOG_ERROR("Long tests with GGUF pulling  will be skipped since RUN_GGUF_TESTS env variable was not set to 1. It was set to: {}", runGGUFTestsEnvContent);
        }
    } else {
        SPDLOG_INFO("Longer GGUF tests will be skipped since RUN_GGUF_TESTS env variable was not set to 1. Remember to use bazel test parameter --test_env when triggering tests using bazel.");
    }
}

bool GGUFEnvironment::shouldSkipWithoutGGUFEnabled() {
    return !GGUFEnvironment::ggufTestsEnabled;
}

void GGUFEnvironment::skipWithoutGPU() {
    if (shouldSkipWithoutGGUFEnabled()) {
        GTEST_SKIP() << "Skipping GGUF tests because those tests were not enabled. Check gguf_environment.[hc]pp";
    }
}

bool GGUFEnvironment::ggufTestsEnabled{false};

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
#include "environment.hpp"

#include <string>

#include "src/logging.hpp"

void Environment::SetUp() {
    ovms::configure_logger("TRACE", "");
    const char* runUnstableTestsEnv = std::getenv("RUN_UNSTABLE");
    if (runUnstableTestsEnv) {
        std::string runUnstableTestsEnvContent(runUnstableTestsEnv);
        if (runUnstableTestsEnvContent == "1") {
            Environment::runUnstableTests = true;
            SPDLOG_INFO("RUN_UNSTABLE was set to 1. Will run unstable tests");
        } else {
            SPDLOG_WARN("Unstable tests will be skipped since RUN_UNSTABLE env variable was not set to 1. It was set to: {}", runUnstableTestsEnvContent);
        }
    } else {
        SPDLOG_INFO("Unstable tests will be skipped since RUN_UNSTABLE env variable was not set to 1. Remember to use bazel test parameter --test_env when triggering tests using bazel.");
    }
}
bool Environment::shouldRunUnstableTests() {
    return Environment::runUnstableTests;
}

bool Environment::runUnstableTests = false;

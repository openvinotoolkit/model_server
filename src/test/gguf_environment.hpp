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
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#define SKIP_AND_EXIT_IF_NO_GGUF()                                                                                 \
    if (GGUFEnvironment::shouldSkipWithoutGGUFEnabled()) {                                                         \
        GTEST_SKIP() << "Skipping GGUF tests because those tests were not enabled. Check gguf_environment.[hc]pp"; \
        return;                                                                                                    \
    }

class GGUFEnvironment : public testing::Environment {
public:
    void SetUp() override;
    static bool ggufTestsEnabled;
    static bool shouldSkipWithoutGGUFEnabled();
    static void skipWithoutGPU();
};

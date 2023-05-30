//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <gtest/gtest.h>

#=//#include "../mediapipe_internal/mediapipedemo.hpp"
#include "mediapipe/framework/port/status.h"

TEST(Mediapipe, PureDummyGraph) {
    size_t requestCount = 3;
    size_t inputStreamDelayMs = 0;
    auto grphExecution = ExecuteDummy(requestCount, inputStreamDelayMs);
    CHECK(grphExecution.ok());
}

TEST(Mediapipe, PureAddGraph) {
    size_t requestCount = 3;
    size_t inputStreamDelayMs = 0;
    auto grphExecution = ExecuteAdd(requestCount, inputStreamDelayMs);
    CHECK(grphExecution.ok());
}

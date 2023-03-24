#pragma once
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
#include <string>

#include "absl/status/status.h"  // TODO include as system lib

extern const std::string DUMMY_MEDIAPIPE_GRAPH;
extern const std::string ADD_MEDIAPIPE_GRAPH;
extern const std::string DUMMY_MEDIAPIPE_GRAPH_ADAPT;
extern const std::string ADD_MEDIAPIPE_GRAPH_ADAPT;

absl::Status ExecuteDummy(size_t requestCount, size_t inputStreamDelayMs);
absl::Status ExecuteAdd(size_t requestCount, size_t inputStreamDelayMs);

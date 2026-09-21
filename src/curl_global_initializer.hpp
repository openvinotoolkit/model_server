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
#pragma once

#include "status.hpp"

namespace ovms {

// Lazily performs a single, process-wide curl_global_init() call and registers a
// matching curl_global_cleanup() via atexit. Safe to call from any thread and any
// number of times/places (module lifecycle, CLI paths, unit tests) - only the first
// caller actually initializes curl, and cleanup runs once, after all curl users in
// the process are done.
Status ensureCurlGlobalInit();

}  // namespace ovms

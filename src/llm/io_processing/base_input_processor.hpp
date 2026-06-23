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

#pragma warning(push)
#pragma warning(disable : 6001 4324 6385 6386)
#include "absl/status/status.h"
#pragma warning(pop)

#include "input_request.hpp"

namespace ovms {

// Abstract base for a single step in the input processing chain.
class BaseInputProcessor {
public:
    virtual ~BaseInputProcessor() = default;

    // Transform req in-place. A non-OK status aborts the chain.
    virtual absl::Status process(InputRequest& req) = 0;
};

}  // namespace ovms

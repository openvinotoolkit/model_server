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

#include <memory>
#include <vector>

#include "absl/status/status.h"

#include "base_input_processor.hpp"
#include "input_processing_config.hpp"
#include "input_processor_context.hpp"
#include "input_request.hpp"

namespace ovms {

// Orchestrates the input processing chain.
// The constructor selects concrete processors based on InputProcessorContext
// and the active InputPayload variant. The chain composition is an implementation detail.
class InputProcessor {
public:
    InputProcessor(InputProcessorContext& context,
        const InputRequest& req);

    // Execute the chain in order. Returns the first non-OK status encountered.
    absl::Status process(InputRequest& req);

private:
    std::vector<std::unique_ptr<BaseInputProcessor>> processors;
};

}  // namespace ovms

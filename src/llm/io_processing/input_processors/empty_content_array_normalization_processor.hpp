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

#include "../base_input_processor.hpp"

namespace ovms {

// Replaces empty content arrays ("content": []) in ChatHistory messages with null.
// Runs for all chat paths (LM and VLM) and must execute before ImageDecodingProcessor
// and TextContentNormalizationProcessor so downstream processors and chat templates
// see a null content instead of an empty array.
class EmptyContentArrayNormalizationProcessor : public BaseInputProcessor {
public:
    absl::Status process(InputRequest& req) override;
};

}  // namespace ovms

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

// Flattens text-only content arrays in ChatHistory messages to plain strings.
// Parts are joined with "\n" for backward compatibility with chat templates.
// Runs for both LM and VLM chat paths: arrays that contain images (or other
// non-text modalities) are left untouched for ImageDecodingProcessor.
// Must run before ChatTemplateProcessor.
class TextContentNormalizationProcessor : public BaseInputProcessor {
public:
    absl::Status process(InputRequest& req) override;
};

}  // namespace ovms

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

#include <optional>

namespace ovms {

struct AudioInputConfig {
    bool resamplingRequired = false;
    bool useAudioContentPart = false;
};

// Deployment-level configuration for InputProcessor, populated once at servable init.
struct InputProcessingConfig {
    // Enables image and video decoding.
    // TokenizationProcessor still runs (to populate inputIds for usage statistics
    // and max-length checks) but inputIds is not passed to the VLM pipeline.
    bool supportsVisualInputs = false;
    // An empty value means audio input is unsupported.
    std::optional<AudioInputConfig> audioInputConfig;
};

}  // namespace ovms

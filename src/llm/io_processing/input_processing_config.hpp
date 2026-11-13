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

#include <cstdint>
#include <optional>

namespace ovms {

struct AudioInputConfig {
    bool resamplingRequired = false;
    bool useAudioContentPart = false;

    static AudioInputConfig withResampling() {
        return AudioInputConfig{true, false};
    }

    static AudioInputConfig withResamplingAndAudioContentPart() {
        return AudioInputConfig{true, true};
    }
};

// Must match the video_max_frames default in llm_calculator.proto.
constexpr uint32_t DEFAULT_VIDEO_MAX_FRAMES = 32;

struct VisionInputConfig {
    std::optional<float> videoFps;
    uint32_t videoMaxFrames = DEFAULT_VIDEO_MAX_FRAMES;
};

// Deployment-level configuration for InputProcessor, populated once at servable init.
struct InputProcessingConfig {
    // An empty value means image and video input is unsupported.
    // When set, TokenizationProcessor still runs (to populate inputIds for usage statistics
    // and max-length checks) but inputIds is not passed to the VLM pipeline.
    std::optional<VisionInputConfig> visionInputConfig;
    // An empty value means audio input is unsupported.
    std::optional<AudioInputConfig> audioInputConfig;
};

}  // namespace ovms

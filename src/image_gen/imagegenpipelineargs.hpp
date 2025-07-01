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

#include <string>
#include <optional>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

namespace ovms {

using resolution_t = std::pair<int64_t, int64_t>;

struct StaticReshapeSettingsArgs {
    std::vector<resolution_t> resolution;
    std::optional<uint64_t> numImagesPerPrompt;
    std::optional<float> guidanceScale;

    StaticReshapeSettingsArgs(const std::vector<resolution_t>& resolutions, 
                              std::optional<uint64_t> numImages = std::nullopt,
                              std::optional<float> guidance = std::nullopt)
        : resolution(resolutions), numImagesPerPrompt(numImages), guidanceScale(guidance) {}
};

struct ImageGenPipelineArgs {
    std::string modelsPath;
    std::vector<std::string> device;
    ov::AnyMap pluginConfig;
    resolution_t maxResolution;
    std::optional<resolution_t> defaultResolution;
    std::optional<uint64_t> seed;
    uint64_t maxNumImagesPerPrompt;
    uint64_t defaultNumInferenceSteps;
    uint64_t maxNumInferenceSteps;

    std::optional<StaticReshapeSettingsArgs> staticReshapeSettings;
};
}  // namespace ovms

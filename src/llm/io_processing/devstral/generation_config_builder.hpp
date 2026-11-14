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
#include "../base_generation_config_builder.hpp"

namespace ovms {

/*
 * DevstralGenerationConfigBuilder extends BaseGenerationConfigBuilder to provide specific configuration for Devstral model.
 * It provides Devstral tool grammar construction for the shared structured-output setup.
 */
class DevstralGenerationConfigBuilder : public BaseGenerationConfigBuilder {
public:
    DevstralGenerationConfigBuilder() = delete;
    explicit DevstralGenerationConfigBuilder(const ov::genai::GenerationConfig& baseConfig, bool enableToolGuidedGeneration, DecodingMethod decodingMethod) :
        BaseGenerationConfigBuilder(baseConfig, enableToolGuidedGeneration, decodingMethod) {}

    bool supportsToolGuidance() const override { return true; }
    void applyToolGuidance(const OpenAIRequest& request) override;
};
}  // namespace ovms

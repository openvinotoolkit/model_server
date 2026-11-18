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

#include "../base_generation_config_builder.hpp"

namespace ovms {

/** Builds schema-constrained Qwen3-Coder XML tool calls. */
class Qwen3CoderGenerationConfigBuilder : public BaseGenerationConfigBuilder {
public:
    explicit Qwen3CoderGenerationConfigBuilder(const ov::genai::GenerationConfig& baseConfig, bool enableToolGuidedGeneration, DecodingMethod decodingMethod) :
        BaseGenerationConfigBuilder(baseConfig, enableToolGuidedGeneration, decodingMethod) {}

    bool supportsToolGuidance() const override { return true; }
    void applyToolGuidance(const OpenAIRequest& request, const ModelGenerationContext& generationContext) override;
};

}  // namespace ovms

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
 * Phi4GenerationConfigBuilder extends BaseGenerationConfigBuilder to provide specific configuration for Phi-4 model.
 * It overrides the parseConfigFromRequest method to set tool guided generation config.
 */
class Phi4GenerationConfigBuilder : public BaseGenerationConfigBuilder {
public:
    Phi4GenerationConfigBuilder() = delete;
    explicit Phi4GenerationConfigBuilder(ov::genai::GenerationConfig& baseConfig) :
        BaseGenerationConfigBuilder(baseConfig) {}

    void parseConfigFromRequest(const OpenAIChatCompletionsRequest& request) override;
};
}  // namespace ovms

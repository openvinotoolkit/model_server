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
 * Hermes3GenerationConfigBuilder extends BaseGenerationConfigBuilder to provide specific configuration for Hermes3 and Qwen3 models.
 * It overrides the parseConfigFromRequest method to set tool guided generation config.
 */
class Hermes3GenerationConfigBuilder : public BaseGenerationConfigBuilder {
public:
    Hermes3GenerationConfigBuilder() = delete;
    explicit Hermes3GenerationConfigBuilder(ov::genai::GenerationConfig& baseConfig) :
        BaseGenerationConfigBuilder(baseConfig) {}

    void parseConfigFromRequest(const OpenAIChatCompletionsRequest& request) override;
};
}  // namespace ovms

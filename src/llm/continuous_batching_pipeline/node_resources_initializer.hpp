//*****************************************************************************
// Copyright 2024 Intel Corporation
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
#include <string>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)
#include "../llmnoderesources_initializer.hpp"

namespace ovms {
class Status;
class ContinuousBatchingNodeResourcesInitializer: public LLMNodeResourcesInitializer {
    // Sets "text_processor" property in nodeResources - to be decided if it should be moved to the base class
    static void loadTextProcessor(std::shared_ptr<LLMNodeResources>& nodeResources, const std::string& chatTemplateDirectory);

    static ov::genai::SchedulerConfig prepareDraftPipelineSchedulerConfig(const mediapipe::LLMCalculatorOptions_PipelineConfig& draftModelConfig);
    static ov::genai::SchedulerConfig prepareDraftPipelineSchedulerConfigLegacy(const mediapipe::LLMCalculatorOptions& nodeOptions);
public:
    Status initialize(std::shared_ptr<LLMNodeResources>& nodeResources, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath) override;
    // Backward compatibility support. This method is required for as long as we want to support legacy LLMCalculatorOptions
    Status initializeLegacy(std::shared_ptr<LLMNodeResources>& nodeResources, const mediapipe::LLMCalculatorOptions& nodeOptions, std::string graphPath);
};
}  // namespace ovms

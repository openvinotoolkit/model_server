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
#include <memory>
#include <string>
#include <vector>

#include <openvino/genai/visual_language/pipeline.hpp>

#include "../../language_model/continuous_batching/servable.hpp"

namespace ovms {

/*
VisualLanguageModelServable extends ContinuousBatchingServable since in GenAI VLM is executed by CB engine, so many parts are common.
This servable also reuses CB servable initializer.
*/

using VisualLanguageModelServableProperties = ContinuousBatchingServableProperties;

struct VisualLanguageModelServableExecutionContext : public ContinuousBatchingServableExecutionContext {
};

class VisualLanguageModelServable : public ContinuousBatchingServable {
public:
    VisualLanguageModelServable() {
        properties = std::make_shared<VisualLanguageModelServableProperties>();
        properties->inputProcessorContext.config.isVLM = true;
    }

    // Overriding ContinuousBatchingServable method
    absl::Status addRequestToPipeline(std::shared_ptr<ContinuousBatchingServableExecutionContext>& executionContext) override;

    // Interface methods
    absl::Status loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const HttpPayload& payload) override;
    std::shared_ptr<GenAiServableExecutionContext> createExecutionContext() override;
    std::shared_ptr<GenAiServableProperties> getProperties() override;
};
}  // namespace ovms

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

#include <openvino/genai/continuous_batching_pipeline.hpp>

#include "../../servable.hpp"
#include "src/llm/llm_calculator.pb.h"

namespace ovms {

class LLMExecutorWrapper;

struct ContinuousBatchingServableExecutionContext : public GenAiServableExecutionContext {
    ov::genai::GenerationHandle generationHandle;
};

struct ContinuousBatchingServableProperties : public GenAiServableProperties {
    ov::genai::SchedulerConfig schedulerConfig;
    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> pipeline;
    std::shared_ptr<LLMExecutorWrapper> llmExecutorWrapper;
};

class ContinuousBatchingServable : public GenAiServable {
protected:
    std::shared_ptr<ContinuousBatchingServableProperties> properties;
    void notifyExecutorThread();

public:
    ContinuousBatchingServable() {
        properties = std::make_shared<ContinuousBatchingServableProperties>();
    }

    // addRequestToPipeline implementation can be specific for different servables with Continuous Batching engine
    // This method is used in scheduleExecution and MUST fill generationHandle in executionContext
    virtual absl::Status addRequestToPipeline(std::shared_ptr<ContinuousBatchingServableExecutionContext>& executionContext);

    // Interface methods
    std::shared_ptr<GenAiServableExecutionContext> createExecutionContext() override;
    std::shared_ptr<GenAiServableProperties> getProperties() override;
    absl::Status scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
};
}  // namespace ovms

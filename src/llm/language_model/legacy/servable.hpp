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

#include "openvino/genai/llm_pipeline.hpp"

#include "../../servable.hpp"
#include "legacy_executor.hpp"
#include "src/llm/llm_calculator.pb.h"

namespace ovms {

struct LegacyServableExecutionContext : public GenAiServableExecutionContext {
    ov::genai::EncodedResults results;
    std::promise<void> readySignal;
    std::future<void> finished = readySignal.get_future();
    std::mutex mutex;
    std::condition_variable executionInProgress;
    bool success = true;
};

struct LegacyServableProperties : public GenAiServableProperties {
    ov::genai::SchedulerConfig schedulerConfig;
    std::shared_ptr<ov::genai::LLMPipeline> pipeline;
    std::shared_ptr<LegacyExecutorWrapper> legacyExecutor;
    int64_t maxPromptLength = 1024;  // NPU property. 1024 is the default value in the plugin
};

class LegacyServable : public GenAiServable {
    std::shared_ptr<LegacyServableProperties> properties;

protected:
    void notifyExecutorThread();
    absl::Status validateInputComplianceWithProperties(const ov::Tensor& inputIds) const;

public:
    LegacyServable() {
        properties = std::make_shared<LegacyServableProperties>();
    }

    // Interface methods
    std::shared_ptr<GenAiServableExecutionContext> createExecutionContext() override;
    std::shared_ptr<GenAiServableProperties> getProperties() override;
    absl::Status parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status prepareInputs(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
};
}  // namespace ovms

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
#include <future>
#include <memory>
#include <string>
#include <vector>

#include <openvino/genai/visual_language/pipeline.hpp>

#include "../../servable.hpp"
#include "legacy_executor.hpp"
#include "src/llm/llm_calculator.pb.h"

namespace ovms {

struct VisualLanguageModelLegacyServableExecutionContext : public LegacyServableExecutionContextBase {
    ov::genai::VLMDecodedResults results;
    // readySignal, finished, success are inherited from LegacyServableExecutionContextBase
    // Workaround needed to pass generation config to the executor that requires it
    ov::genai::GenerationConfig baseGenerationConfig;

    // Disconnection handling
    std::atomic<bool> clientDisconnected{false};

    void signalDisconnection() {
        clientDisconnected = true;
        deltaChannel.signalComplete();
    }

    // Legacy generation path always runs with a single beam, so finish_reasons[0] is the result.
    ov::genai::GenerationFinishReason legacyFinishReason() const override {
        return results.finish_reasons.empty() ? ov::genai::GenerationFinishReason::STOP
                                              : results.finish_reasons[0];
    }
    void setLegacyUsage(OpenAIApiHandler& apiHandler) override {
        apiHandler.setPromptTokensUsage(results.perf_metrics.get_num_input_tokens());
        apiHandler.setCompletionTokensUsage(results.perf_metrics.get_num_generated_tokens());
    }
};

struct VisualLanguageModelLegacyServableProperties : public GenAiServableProperties {
    ov::genai::SchedulerConfig schedulerConfig;
    std::shared_ptr<ov::genai::VLMPipeline> pipeline;
    std::shared_ptr<VisualLanguageModelLegacyExecutorWrapper> legacyExecutor;
};

class VisualLanguageModelLegacyServable : public LegacyServableBase {
    std::shared_ptr<VisualLanguageModelLegacyServableProperties> properties;

protected:
    void notifyExecutorThread();

public:
    VisualLanguageModelLegacyServable() {
        properties = std::make_shared<VisualLanguageModelLegacyServableProperties>();
        properties->inputProcessorContext.config.isVLM = true;
    }

    // Interface methods
    absl::Status validateEndpoint(Endpoint endpoint) const override;
    std::shared_ptr<GenAiServableExecutionContext> createExecutionContext() override;
    std::shared_ptr<GenAiServableProperties> getProperties() override;
    absl::Status parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
};
}  // namespace ovms

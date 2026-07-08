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
#include <future>
#include <memory>
#include <string>
#include <vector>

#include <openvino/genai/omni/pipeline.hpp>
#include <openvino/genai/omni/decoded_results.hpp>

#include "../../servable.hpp"
#include "legacy_executor.hpp"
#include "src/llm/llm_calculator.pb.h"

namespace ovms {

struct OmniModelLegacyServableExecutionContext : public GenAiServableExecutionContext {
    ov::genai::OmniDecodedResults results;
    std::promise<void> readySignal;
    std::future<void> finished = readySignal.get_future();
    ov::genai::GenerationConfig baseGenerationConfig;
    bool success{true};
    std::string accumulatedUnaryText;

    std::atomic<bool> clientDisconnected{false};

    void signalDisconnection() {
        clientDisconnected = true;
        deltaChannel.signalComplete();
    }
};

struct OmniModelLegacyServableProperties : public GenAiServableProperties {
    std::shared_ptr<ov::genai::OmniPipeline> pipeline;
    std::shared_ptr<OmniModelLegacyExecutorWrapper> legacyExecutor;
};

class OmniModelLegacyServable : public GenAiServable {
    std::shared_ptr<OmniModelLegacyServableProperties> properties;

public:
    OmniModelLegacyServable() {
        properties = std::make_shared<OmniModelLegacyServableProperties>();
        properties->inputProcessorContext.config.isVLM = true;
    }

    absl::Status loadRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext, const HttpPayload& payload);
    std::shared_ptr<GenAiServableExecutionContext> createExecutionContext() override;
    std::shared_ptr<GenAiServableProperties> getProperties() override;
    absl::Status parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status scheduleExecution(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status readCompleteExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status prepareCompleteResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status readPartialExecutionResults(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
    absl::Status preparePartialResponse(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override;
};
}  // namespace ovms

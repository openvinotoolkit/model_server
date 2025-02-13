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

#include <openvino/genai/continuous_batching_pipeline.hpp>

#pragma warning(push)
#pragma warning(disable : 4005 4309 6001 6385 6386 6326 6011 4005)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

#include "../../logging.hpp"
#include "../../stringutils.hpp"
#include "src/llm/llm_calculator.pb.h"
#include "../llmnoderesources.hpp"

namespace ovms {

class LLMExecutorWrapper;

class Status;

#pragma GCC visibility push(hidden)


class ContinuousBatchingNodeResources : public LLMNodeResources {
private:
    void notifyExecutorThread();
public:
    // Sets "llm_execution_wrapper" property in nodeResources
    ovms::Status initialize() override;

    absl::Status createApiHandler(ov::AnyMap& executionContext) override;

    // Consider make apiHandler a member of the class
    absl::Status parseRequest(ov::AnyMap& executionContext) override;

    absl::Status preparePipelineInput(ov::AnyMap& executionContext) override;

    absl::Status schedulePipelineExecution(ov::AnyMap& executionContext) override;

    absl::Status readCompleteExecutionResults(ov::AnyMap& executionContext) override;

    absl::Status readPartialExecutionResults(ov::AnyMap& executionContext) override;
};
#pragma GCC visibility pop
using LLMNodeResourcesMap = std::unordered_map<std::string, std::shared_ptr<LLMNodeResources>>;

}  // namespace ovms

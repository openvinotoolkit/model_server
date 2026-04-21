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
#include "pipeline_factory.hpp"

#include <memory>
#include <string>

#include "src/logging.hpp"
#include "pipelinedefinition.hpp"

namespace ovms {
template <typename RequestType, typename ResponseType>
Status PipelineFactory::create(std::unique_ptr<Pipeline>& pipeline, const std::string& name, const RequestType* request, ResponseType* response, ModelInstanceProvider& provider) const {
    std::shared_lock lock(definitionsMtx);
    auto it = definitions.find(name);
    if (it == definitions.end()) {
        SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Pipeline with requested name: {} does not exist", name);
        return StatusCode::PIPELINE_DEFINITION_NAME_MISSING;
    }
    auto& definition = *it->second;
    lock.unlock();
    return definition.create(pipeline, request, response, provider);
}
}  // namespace ovms

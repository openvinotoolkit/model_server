//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#include "nodeinputhandler.hpp"

#include "logging.hpp"

namespace ovms {

NodeInputHandler::NodeInputHandler(uint32_t inputsMissingCount) :
    remainingDependencies(inputsMissingCount) {
}

void NodeInputHandler::setInput(const std::string& inputName, InferenceEngine::Blob::Ptr& ptr, session_id_t shardId) {
    if (inputBlobs.find(inputName) != inputBlobs.end()) {
        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Tried to set the same input: {} twice for the NodeInputHandler.", inputName);
        throw std::logic_error("Tried to set the same input twice for the NodeInputHandler");
    }
    inputBlobs.emplace(inputName, ptr);
}

void NodeInputHandler::clearInputs() {
    inputBlobs.clear();
}

bool NodeInputHandler::isReady() {
    return remainingDependencies == 0;
}

Status NodeInputHandler::notifyFinishedDependency() {
    SPDLOG_LOGGER_DEBUG(dag_executor_logger, "Remaining dependencies count for input handler decreased from: {} to: {}", remainingDependencies, remainingDependencies - 1);
    --remainingDependencies;
    return StatusCode::OK;
}
}  // namespace ovms

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

#include "global_sequences_viewer.hpp"

#include <limits>
#include <memory>
#include <utility>

#include "logging.hpp"
//#include "model.hpp"
#include "modelversion.hpp"
#include "sequence_manager.hpp"
//#include "statefulmodelinstance.hpp"
#include "status.hpp"

namespace ovms {

static const std::string separator = "_";

Status GlobalSequencesViewer::registerForCleanup(std::string modelName, model_version_t modelVersion, std::shared_ptr<SequenceManager> sequenceManager) {
    std::string registration_id = modelName + separator + std::to_string(modelVersion);
    std::unique_lock<std::mutex> viewerLock(viewerMutex);
    if (registeredSequenceManagers.find(registration_id) != registeredSequenceManagers.end()) {
        SPDLOG_LOGGER_ERROR(sequence_manager_logger, "Model: {}, version: {}, cannot register model instance in sequence cleaner. Already registered.", modelName, modelVersion);
        return StatusCode::INTERNAL_ERROR;
    } else {
        registeredSequenceManagers.emplace(registration_id, sequenceManager);
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model: {}, version: {}, has been successfully registered in sequence cleaner", modelName, modelVersion);
    }

    return StatusCode::OK;
}

Status GlobalSequencesViewer::unregisterFromCleanup(std::string modelName, model_version_t modelVersion) {
    std::string registration_id = modelName + separator + std::to_string(modelVersion);
    std::unique_lock<std::mutex> viewerLock(viewerMutex);
    auto it = registeredSequenceManagers.find(registration_id);
    if (it != registeredSequenceManagers.end()) {
        registeredSequenceManagers.erase(it);
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model: {}, version: {}, has been successfully unregistered from sequence cleaner", modelName, modelVersion);
    } else {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model: {}, version: {}, cannot unregister model instance from sequence cleaner. It has not been registered.", modelName, modelVersion);
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

Status GlobalSequencesViewer::removeIdleSequences() {
    std::unique_lock<std::mutex> viewerLock(viewerMutex);
    for (auto it = registeredSequenceManagers.begin(); it != registeredSequenceManagers.end();) {
        auto sequenceManager = it->second;
        auto status = sequenceManager->removeIdleSequences();
        it++;
        if (status.getCode() != ovms::StatusCode::OK)
            return status;
    }

    return ovms::StatusCode::OK;
}
}  // namespace ovms

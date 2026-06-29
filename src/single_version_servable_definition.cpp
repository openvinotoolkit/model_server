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
#include "single_version_servable_definition.hpp"

#include <chrono>
#include <mutex>

#include <spdlog/spdlog.h>

#include "dags/pipelinedefinitionstatus.hpp"
#include "servable_definition_unload_guard.hpp"

namespace ovms {

Status SingleVersionServableDefinition::waitForLoaded(std::unique_ptr<ServableDefinitionUnloadGuard>& guard, const uint32_t waitForLoadedTimeoutMicroseconds) {
    guard = std::make_unique<ServableDefinitionUnloadGuard>(*this);

    const uint32_t waitLoadedTimestepMicroseconds = 1000;
    const uint32_t waitCheckpoints = waitForLoadedTimeoutMicroseconds / waitLoadedTimestepMicroseconds;
    uint32_t waitCheckpointsCounter = waitCheckpoints;
    std::mutex cvMtx;
    std::unique_lock<std::mutex> cvLock(cvMtx);
    while (waitCheckpointsCounter-- != 0) {
        if (getStatus().isAvailable()) {
            SPDLOG_DEBUG("Successfully waited for definition: {}", getName());
            return StatusCode::OK;
        }
        guard.reset();
        if (!getStatus().canEndLoaded()) {
            if (getStatus().getStateCode() != PipelineDefinitionStateCode::RETIRED) {
                SPDLOG_DEBUG("Waiting for definition: {} ended due to timeout.", getName());
                return notLoadedYetCode();
            } else {
                SPDLOG_DEBUG("Waiting for definition: {} ended since it failed to load.", getName());
                return notLoadedAnymoreCode();
            }
        }
        SPDLOG_DEBUG("Waiting for available state for definition: {}, with timestep: {}us timeout: {}us check count: {}",
            getName(), waitLoadedTimestepMicroseconds, waitForLoadedTimeoutMicroseconds, waitCheckpointsCounter);
        loadedNotify.wait_for(cvLock,
            std::chrono::microseconds(waitLoadedTimestepMicroseconds),
            [this]() {
                return this->getStatus().isAvailable() ||
                       !this->getStatus().canEndLoaded();
            });
        guard = std::make_unique<ServableDefinitionUnloadGuard>(*this);
    }
    if (!getStatus().isAvailable()) {
        if (getStatus().getStateCode() != PipelineDefinitionStateCode::RETIRED) {
            SPDLOG_DEBUG("Waiting for definition: {} ended due to timeout.", getName());
            return notLoadedYetCode();
        } else {
            SPDLOG_DEBUG("Waiting for definition: {} ended since it failed to load.", getName());
            return notLoadedAnymoreCode();
        }
    }
    SPDLOG_DEBUG("Successfully waited for definition: {}", getName());
    return StatusCode::OK;
}

}  // namespace ovms

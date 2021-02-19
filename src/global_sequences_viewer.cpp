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
#include "model.hpp"
#include "modelversion.hpp"
#include "statefulmodelinstance.hpp"
#include "status.hpp"

namespace ovms {

static bool sequenceWatcherStarted = false;
static std::string separator = "_";

ovms::Status GlobalSequencesViewer::addVersions(std::shared_ptr<ovms::Model>& model,
    std::shared_ptr<model_versions_t> versionsToAdd,
    std::shared_ptr<model_versions_t> versionsFailed) {
    for (const auto version : *versionsToAdd) {
        if (std::count(versionsFailed->begin(), versionsFailed->end(), version))
            continue;
        auto modelInstance = model->getModelInstanceByVersion(version);
        if (!modelVersion) {
            Status status = StatusCode::UNKNOWN_ERROR;
            SPDLOG_ERROR("Error occurred while getting model instance for model: {}; version: {}; error: {}",
                model->getName(),
                version,
                status.string());
            result = status;
            continue;
        }
        auto stetefulModelInstance = std::static_pointer_cast<StatefulModelInstance>(modelInstance);
        std::string managerId = model->getName() + separator + std::to_string(version);

        auto status = registerManager(managerId, stetefulModelInstance->getSequenceManager().get());
        if (status.getCode() != ovms::StatusCode::OK)
            return status;
    }

    updateThreadInterval();
    return ovms::StatusCode::OK;
}

ovms::Status GlobalSequencesViewer::retireVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<model_versions_t> versionsToRetire) {
    for (const auto version : *versionsToRetire) {
        std::string managerId = model->getName() + separator + std::to_string(version);
        auto status = unregisterManager(managerId);
        if (status.getCode() != ovms::StatusCode::OK)
            return status;
    }

    updateThreadInterval();
    return ovms::StatusCode::OK;
}

ovms::Status GlobalSequencesViewer::reloadVersions(std::shared_ptr<ovms::Model>& model,
    std::shared_ptr<model_versions_t> versionsToReload,
    std::shared_ptr<model_versions_t> versionsFailed) {
    for (const auto version : *versionsToReload) {
        if (std::count(versionsFailed->begin(), versionsFailed->end(), version))
            continue;
        std::string managerId = model->getName() + separator + std::to_string(version);
        auto status = unregisterManager(managerId);
        if (status.getCode() != ovms::StatusCode::OK)
            return status;

        auto modelInstance = model->getModelInstanceByVersion(version);
        if (!modelVersion) {
            Status status = StatusCode::UNKNOWN_ERROR;
            SPDLOG_ERROR("Error occurred while getting model instance for model: {}; version: {}; error: {}",
                model->getName(),
                version,
                status.string());
            result = status;
            continue;
        }
        auto stetefulModelInstance = std::static_pointer_cast<StatefulModelInstance>(modelInstance);

        status = registerManager(managerId, stetefulModelInstance->getSequenceManager().get());
        if (status.getCode() != ovms::StatusCode::OK)
            return status;
    }
    updateThreadInterval();
    return ovms::StatusCode::OK;
}

void GlobalSequencesViewer::updateThreadInterval() {
    uint32_t lowestHalfTimeoutInterval = std::numeric_limits<uint32_t>::max();
    for (auto const&[key, val] : registeredSequenceManagers) {
        auto sequenceManager = val;
        uint32_t newInterval = sequenceManager->getTimeout() / 2;
        if (newInterval > 0 && newInterval < lowestHalfTimeoutInterval) {
            lowestHalfTimeoutInterval = newInterval;
        }
        else if (sequenceManager->getTimeout() == 1){
            lowestHalfTimeoutInterval = 1;
        }
    }

    sequenceWatcherIntervalSec = lowestHalfTimeoutInterval;
}

ovms::Status GlobalSequencesViewer::registerManager(std::string managerId, SequenceManager* sequenceManager) {
    std::unique_lock<std::mutex> viewerLock(mutex);
    if (registeredSequenceManagers.count(managerId)) {
        SPDLOG_LOGGER_ERROR(sequence_manager_logger, "Sequence manager {} already exists", managerId);
        return StatusCode::INTERNAL_ERROR;
    } else {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Sequence manager {} registered in the sequence timeout watcher", managerId);
        registeredSequenceManagers.emplace(managerId, sequenceManager);
    }

    return StatusCode::OK;
}

ovms::Status GlobalSequencesViewer::unregisterManager(std::string managerId) {
    std::unique_lock<std::mutex> viewerLock(mutex);
    if (registeredSequenceManagers.count(managerId)) {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Sequence manager {} unregistred from sequence timeout watcher", managerId);
        registeredSequenceManagers.erase(managerId);
    } else {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Sequence manager {} does not exists", managerId);
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

ovms::Status GlobalSequencesViewer::removeTimedOutSequences() {
    std::unique_lock<std::mutex> viewerLock(mutex);
    for (auto it = registeredSequenceManagers.begin(); it != registeredSequenceManagers.end();) {
        auto sequenceManager = it->second;
        auto status = sequenceManager->removeTimedOutSequences();
        it++;
        if (status.getCode() != ovms::StatusCode::OK)
            return status;
    }

    return ovms::StatusCode::OK;
}

void GlobalSequencesViewer::sequenceWatcher(std::future<void> exit) {
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Started sequence timeout watcher thread");

    while (exit.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
        std::this_thread::sleep_for(std::chrono::seconds(sequenceWatcherIntervalSec));
        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Sequence watcher thread check cycle begin");

        removeTimedOutSequences();

        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Sequence watcher thread check cycle end");
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Exited sequence timeout watcher thread");
}

void GlobalSequencesViewer::join() {
    if (sequenceWatcherStarted) {
        exit.set_value();
        if (sequenceMonitor.joinable()) {
            sequenceMonitor.join();
            sequenceWatcherStarted = false;
        }
    }
}

void GlobalSequencesViewer::startWatcher() {
    if ((!sequenceWatcherStarted) && (sequenceWatcherIntervalSec > 0)) {
        std::future<void> exitSignal = exit.get_future();
        std::thread t(std::thread(&GlobalSequencesViewer::sequenceWatcher, this, std::move(exitSignal)));
        sequenceWatcherStarted = true;
        sequenceMonitor = std::move(t);
    }
}

}  // namespace ovms

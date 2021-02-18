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

#pragma once

#include "global_sequences_viewer.hpp"

#include "model.hpp"
#include "modelversion.hpp"
#include "stategulmodelinstance.hpp"
#include "status.hpp"

namespace ovms {

    static bool sequenceWatcherStarted = false;

    std::mutex& GlobalSequencesViewer::getMutex() {
        return mutex;
    }

    Status GlobalSequencesViewer::addVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<model_versions_t> versionsToAdd) {
        for (const auto version : *versionsToStart) {
            auto modelInstance = model->getModelInstanceByVersion(version);
            auto stetefulModelInstance = std::static_pointer_cast<StatefulModelInstance>(modelInstance);
            std::string managerId = model->getName() + std::to_string(version);
            register(managerId, statefulModelInstance.getSequenceManager());
        }

        updateThreadInterval();
    }

    Status GlobalSequencesViewer::retireVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<model_versions_t> versionsToRetire) {
        for (const auto version : *versionsToStart) {
            std::string managerId = model->getName() + std::to_string(version);
            unregister(managerId);
        }

        updateThreadInterval();
    }

    Status GlobalSequencesViewer::reloadVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<model_versions_t> versionsToReload) {
        // TODO: Do we need to implement this ?
        updateThreadInterval();
    }

    status GlobalSequencesViewer::updateThreadInterval() {
        uint32_t lowestHalfTimeoutInterval = DEFAULT_SEQUENCE_TIMEOUT_SECONDS/2;
        for (auto const&[key, val] : registeredSequenceManagers) {
            auto sequenceManager = val;
            uint32_t newInterval = sequenceManager.getTimeout() / 2;
            if (newInterval < lowestHalfTimeoutInterval)
                lowestHalfTimeoutInterval = newInterval;
        }

        sequenceWatcherIntervalSec = lowestHalfTimeoutInterval;
    }

    Status GlobalSequencesViewer::register(std::string managerId, std::shared_ptr<SequenceManager> sequenceManager) {
        std::unique_lock<std::mutex> viewerLock(mutex);
        if (registeredSequenceManagers.count(managerId)) {
            SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Sequence manager {} already exists", managerId);
            return StatusCode::SEQUENCE_ALREADY_EXISTS;
        }
        else {
            registeredSequenceManagers.emplace(managerId, sequenceManager);
        }

        return StatusCode::OK;
    }

    Status GlobalSequencesViewer::unregister(std::string managerId) {
        std::unique_lock<std::mutex> viewerLock(mutex);
        if (registeredSequenceManagers.count(managerId)) {
            registeredSequenceManagers.erase(managerId);
        }
        else {
            SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Sequence manager {} already exists", managerId);
            return StatusCode::SEQUENCE_MISSING;
        }
        return StatusCode::OK;
    }

    Status GlobalSequencesViewer::removeTimedOutSequences()
    {
        std::unique_lock<std::mutex> viewerLock(mutex);
        for (auto it = registeredSequenceManagers.begin(); it != registeredSequenceManagers.end();) {
            SequenceManager& sequenceManager = it->second;
            seqenceManager.removeTimeOutedSequences();
        }
    }

    void GlobalSequencesViewer::sequenceWatcher(std::future<void> exit) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Started sequence timeout watcher thread");

        while (exit.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
            std::this_thread::sleep_for(std::chrono::seconds(sequenceWatcherIntervalSec));
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Sequence watcher thread check cycle begin");

            removeTimedOutSequences();

            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Sequence watcher thread check cycle end");
        }
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Exited sequence timeout watcher thread");
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
            std::thread t(std::thread(&ModelManager::sequenceWatcher, this, std::move(exitSignal)));
            sequenceWatcherStarted = true;
            sequenceMonitor = std::move(t);
        }
    }

}  // namespace ovms

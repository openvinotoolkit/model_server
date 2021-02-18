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

#include <mutex>
#include <string>
#include <unordered_map>

#include "modelconfig.hpp"
#include "modelversion.hpp"
#include "sequence_manager.hpp"
#include "status.hpp"

namespace ovms {

class GlobalSequencesViewer {
    private:
        std::mutex mutex;
        std::map<std::string, std::shared_ptr<SequenceManager>> registeredSequenceManagers;

        /**
         * @brief sequence Watcher thread for monitor changes in config
         */
        void sequenceWatcher(std::future<void> exit);

        /**
         * @brief A thread object used for monitoring sequence timeouts
         */
        std::thread sequenceMonitor;

        /**
         * @brief An exit signal to notify watcher thread to exit
         */
        std::promise<void> exit;

        /**
         * Time interval between each sequence timeout check
         */
        uint sequenceWatcherIntervalSec = DEFAULT_SEQUENCE_TIMEOUT_SECONDS / 2;

        std::map<std::string, std::shared_ptr<SequenceManager>> getSequenceManagers;

        Status register(std::string managerId, std::shared_ptr<SequenceManager> sequenceManager);

        Status unregister(std::string managerId);

        Status RemoveTimedOutSequences();

    public:
        GlobalSequencesViewer() = default;

        std::mutex& getMutex();

        Status addVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<model_versions_t> versionsToAdd);

        Status retireVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<model_versions_t> versionsToRetire);

        Status reloadVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<model_versions_t> versionsToReload);

        /**
         *  @brief Gets the sequence watcher interval timestep in seconds
         */
        uint getSequenceWatcherIntervalSec() {
            return sequenceWatcherIntervalSec;
        }

        /**
         * @brief Gracefully finish the thread
         */
        void join();

    };
}  // namespace ovms

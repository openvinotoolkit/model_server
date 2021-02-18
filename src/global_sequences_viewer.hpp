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

#include "model.hpp"
#include "modelconfig.hpp"
#include "modelversion.hpp"
#include "sequence_manager.hpp"
#include "status.hpp"

namespace ovms {

class GlobalSequencesViewer {
private:
    std::mutex mutex;
    std::map<std::string, SequenceManager*> registeredSequenceManagers;

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
    uint sequenceWatcherIntervalSec = 1;

    ovms::Status registerManager(std::string managerId, SequenceManager* sequenceManager);

    ovms::Status unregisterManager(std::string managerId);

    ovms::Status removeTimedOutSequences();

    void updateThreadInterval();

public:
    GlobalSequencesViewer() = default;

    void startWatcher();

    std::mutex& getMutex();

    ovms::Status addVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<model_versions_t> versionsToAdd);

    ovms::Status retireVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<model_versions_t> versionsToRetire);

    ovms::Status reloadVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<model_versions_t> versionsToReload);

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

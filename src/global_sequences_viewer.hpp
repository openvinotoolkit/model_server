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

#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "sequence_manager.hpp"
#include "status.hpp"

namespace ovms {
const uint32_t DEFAULT_SEQUENCE_CLEANER_INTERVAL = 5;  // in minutes
class GlobalSequencesViewer {
private:
    // used to block parallel access to registered sequence managers map
    std::mutex viewerMutex;

    // used to send exit signal to sequence cleaner thread and force termination
    std::promise<void> exitTrigger;

    std::map<std::string, std::shared_ptr<SequenceManager>> registeredSequenceManagers;

    void sequenceCleanerRoutine(uint32_t sequenceCleanerIntervalMinutes, std::future<void> exitSignal);

    std::thread sequenceCleanerThread;

protected:
    Status removeIdleSequences();

public:
    void startCleanerThread(uint32_t sequenceCleanerIntervalMinutes = DEFAULT_SEQUENCE_CLEANER_INTERVAL);

    // Gracefully finish sequence cleaner thread
    void join();

    Status registerForCleanup(std::string modelName, model_version_t modelVersion, std::shared_ptr<SequenceManager> sequenceManager);

    Status unregisterFromCleanup(std::string modelName, model_version_t modelVersion);
};
}  // namespace ovms

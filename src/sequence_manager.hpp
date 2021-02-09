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
#include <memory>
#include <mutex>
#include <unordered_map>

#include "sequence.hpp"
#include "sequence_processing_spec.hpp"
#include "status.hpp"

namespace ovms {

const uint32_t NO_CONTROL_INPUT = 0;
const uint32_t SEQUENCE_START = 1;
const uint32_t SEQUENCE_END = 2;

class SequenceManager {
private:
    uint32_t timeout;
    uint32_t maxSequenceNumber;
    std::mutex mutex;
    /**
     * Time interval between each sequences timeout check
     */
    uint sequenceWatcherIntervalSec;

protected:
    std::unordered_map<uint64_t, Sequence> sequences;

    Status hasSequence(const uint64_t sequenceId);

    Status createSequence(const uint64_t sequenceId);

    Status terminateSequence(const uint64_t sequenceId);

public:
    SequenceManager() = default;
    SequenceManager(uint32_t timeout, uint32_t maxSequenceNumber) :
        timeout(timeout),
        maxSequenceNumber(maxSequenceNumber),
        sequenceWatcherIntervalSec(timeout / 2) {
        startWatcher();
    }

    ~SequenceManager();

    uint64_t getSequencesCount() {
        return sequences.size();
    }

    const uint32_t getTimeout() const;

    void setTimeout(uint32_t timeout);

    const uint32_t getMaxSequenceNumber() const;

    void setMaxSequenceNumber(uint32_t maxSequenceNumber);

    std::mutex& getMutex();

    bool sequenceExists(const uint64_t sequenceId) const;

    Sequence& getSequence(const uint64_t sequenceId);

    Status removeSequence(const uint64_t sequenceId);

    Status removeTimedOutSequences();

    Status processRequestedSpec(SequenceProcessingSpec& sequenceProcessingSpec);

    /**
     * @brief Starts sequence monitoring as new thread
     *
     */
    void startWatcher();

    /**
     * @brief Gracefully finish the monitoring thread
     */
    void join();

    /**
     * @brief A thread object used for monitoring changes in sequences
     */
    std::thread monitor;

    /**
     * @brief An exit signal to notify watcher thread to exit
     */
    std::promise<void> exit;

    /**
    * @brief Watcher thread for monitor changes in config
    */
    void watcher(std::future<void> exit);
};
}  // namespace ovms

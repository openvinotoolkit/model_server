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

#include "sequence_manager.hpp"

#include <utility>

#include "logging.hpp"

namespace ovms {

const uint32_t SequenceManager::getTimeout() const {
    return timeout;
}

void SequenceManager::setTimeout(uint32_t timeout) {
    this->timeout = timeout;
}

const uint32_t SequenceManager::getMaxSequenceNumber() const {
    return maxSequenceNumber;
}

void SequenceManager::setMaxSequenceNumber(uint32_t maxSequenceNumber) {
    this->maxSequenceNumber = maxSequenceNumber;
}

std::mutex& SequenceManager::getMutex() {
    return mutex;
}

bool SequenceManager::sequenceExists(const uint64_t sequenceId) const {
    return sequences.count(sequenceId);
}

Status SequenceManager::removeTimedOutSequences() {
    std::unique_lock<std::mutex> sequenceManagerLock(mutex);
    for (auto it = sequences.cbegin(); it != sequences.cend();) {
        Sequence& sequence = getSequence(it->second.getId());
        if (sequence.isTimedOut()) {
            SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} Removing timeouted sequence - Id: {}", modelName, sequence.getId());
            it = sequences.erase(it);
        } else {
            ++it;
        }
    }
    return StatusCode::OK;
}

Status SequenceManager::checkForTimedOutSequences() {
    std::unique_lock<std::mutex> sequenceManagerLock(mutex);
    std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
    for (auto it = sequences.cbegin(); it != sequences.cend();) {
        Sequence& sequence = getSequence(it->second.getId());
        std::unique_lock<std::mutex> sequenceLock(sequence.getMutex());
        if (!sequence.isTerminated()) {
            auto timeDiff = currentTime - sequence.getLastActivityTime();
            if (std::chrono::duration_cast<std::chrono::seconds>(timeDiff).count() > timeout) {
                SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} sequence set for timeout - Id: {}", modelName, sequence.getId());
                sequence.setTimedOut();
                sequenceLock.unlock();
            }
        }
        ++it;
    }
    return StatusCode::OK;
}

Status SequenceManager::hasSequence(const uint64_t sequenceId) {
    if (!sequenceExists(sequenceId))
        return StatusCode::SEQUENCE_MISSING;

    if (getSequence(sequenceId).isTerminated())
        return StatusCode::SEQUENCE_TERMINATED;

    if (getSequence(sequenceId).isTimedOut())
        return StatusCode::SEQUENCE_TIMEDOUT;

    return StatusCode::OK;
}

Status SequenceManager::createSequence(const uint64_t sequenceId) {
    /* TODO: Generate unique ID if not provided by the client
    if (sequenceId == 0) {
    } 
    */
    if (sequenceExists(sequenceId)) {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} Sequence with provided ID already exists", modelName);
        return StatusCode::SEQUENCE_ALREADY_EXISTS;
    } else {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} Adding new sequence with ID: {}", modelName, sequenceId);
        sequences.emplace(sequenceId, sequenceId);
    }
    return StatusCode::OK;
}

Status SequenceManager::terminateSequence(const uint64_t sequenceId) {
    auto status = hasSequence(sequenceId);
    if (!status.ok())
        return status;

    getSequence(sequenceId).setTerminated();
    return StatusCode::OK;
}

Sequence& SequenceManager::getSequence(const uint64_t sequenceId) {
    return sequences.at(sequenceId);
}

Status SequenceManager::removeSequence(const uint64_t sequenceId) {
    if (sequences.count(sequenceId)) {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} Removing sequence with ID: {}", modelName, sequenceId);
        sequences.erase(sequenceId);
    } else {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} Sequence with provided ID does not exists", modelName);
        return StatusCode::SEQUENCE_MISSING;
    }
    return StatusCode::OK;
}

Status SequenceManager::processRequestedSpec(SequenceProcessingSpec& sequenceProcessingSpec) {
    const uint32_t sequenceControlInput = sequenceProcessingSpec.getSequenceControlInput();
    const uint64_t sequenceId = sequenceProcessingSpec.getSequenceId();
    Status status;

    if (sequenceControlInput == SEQUENCE_START) {
        status = createSequence(sequenceId);
    } else if (sequenceControlInput == NO_CONTROL_INPUT) {
        status = hasSequence(sequenceId);
    } else {  // sequenceControlInput == SEQUENCE_END
        status = terminateSequence(sequenceId);
    }
    return status;
}

void SequenceManager::startWatcher() {
    if ((!sequenceWatcherStarted) && (sequenceWatcherIntervalSec > 0)) {
        std::future<void> exitSignal = exit.get_future();
        std::thread t(std::thread(&SequenceManager::watcher, this, std::move(exitSignal)));
        sequenceWatcherStarted = true;
        monitor = std::move(t);
    }
}

void SequenceManager::join() {
    if (sequenceWatcherStarted) {
        exit.set_value();
        if (monitor.joinable()) {
            monitor.join();
            sequenceWatcherStarted = false;
        }
    }
}

void SequenceManager::watcher(std::future<void> exit) {
    SPDLOG_LOGGER_INFO(sequence_manager_logger, "Model {} Started sequence watcher thread with interval {} seconds", modelName, sequenceWatcherIntervalSec);

    while (exit.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
        std::this_thread::sleep_for(std::chrono::seconds(sequenceWatcherIntervalSec));
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} Sequence watcher thread check cycle begin", modelName);

        checkForTimedOutSequences();
        removeTimedOutSequences();

        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} Sequence watcher thread check cycle end", modelName);
    }
    SPDLOG_LOGGER_INFO(sequence_manager_logger, "Model {} Exited sequence watcher thread", modelName);
}

SequenceManager::~SequenceManager() {
    join();
}

}  // namespace ovms

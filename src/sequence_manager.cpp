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
#include "sequence_processing_spec.hpp"
#include "status.hpp"

namespace ovms {

uint64_t SequenceManager::getUniqueSequenceId() {
    SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "No sequence id has been provided on SEQUENCE_START. Seeking unique sequence id...");
    bool uniqueIdFound = false;
    while (!uniqueIdFound) {
        if (sequenceExists(this->sequenceIdCounter) || this->sequenceIdCounter == 0)
            this->sequenceIdCounter++;
        else
            uniqueIdFound = true;
    }
    SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Found unique sequence id: {}", this->sequenceIdCounter);
    return this->sequenceIdCounter;
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
    return sequences.find(sequenceId) != sequences.end();
}

Status SequenceManager::removeIdleSequences() {
    std::unique_lock<std::mutex> sequenceManagerLock(mutex);
    for (auto it = sequences.begin(); it != sequences.end();) {
        Sequence& sequence = it->second;
        // Non blocking try to get mutex
        std::unique_lock<std::mutex> sequenceLock(sequence.getMutex(), std::try_to_lock);
        if (!sequence.isTerminated() && sequenceLock.owns_lock()) {
            sequenceLock.unlock();
            // We hold sequence manager lock before lock and after unlock so no other thread even attempts accessing that sequence at that moment
            if (sequence.isIdle()) {
                SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "[Idle sequence cleanup] Removing sequence with id: {} on model {}, version: {}", sequence.getId(), modelName, modelVersion);
                it = sequences.erase(it);
                continue;
            } else {
                sequence.setIdle();
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
        return StatusCode::SEQUENCE_MISSING;

    return StatusCode::OK;
}

Status SequenceManager::createSequence(SequenceProcessingSpec& sequenceProcessingSpec) {
    if (sequences.size() >= this->maxSequenceNumber) {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} version {} Max sequence number has been reached. Could not create new sequence.", modelName, modelVersion);
        return StatusCode::MAX_SEQUENCE_NUMBER_REACHED;
    }

    uint64_t sequenceId = sequenceProcessingSpec.getSequenceId();

    if (sequenceId == 0) {
        uint64_t uniqueSequenceId = getUniqueSequenceId();
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} version {} Adding new sequence with ID: {}", modelName, modelVersion, uniqueSequenceId);
        sequences.emplace(uniqueSequenceId, uniqueSequenceId);
        sequenceProcessingSpec.setSequenceId(uniqueSequenceId);
        return StatusCode::OK;
    }

    if (sequenceExists(sequenceId)) {
        if (getSequence(sequenceId).isTerminated()) {
            SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} version {} Sequence with provided ID is currently being removed", modelName, modelVersion);
            return StatusCode::SEQUENCE_TERMINATED;
        }
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} version {} Sequence with provided ID already exists", modelName, modelVersion);
        return StatusCode::SEQUENCE_ALREADY_EXISTS;
    } else {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} version {} Adding new sequence with ID: {}", modelName, modelVersion, sequenceId);
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
    auto it = sequences.find(sequenceId);
    if (it != sequences.end()) {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} versions {} Removing sequence with ID: {}", modelName, modelVersion, sequenceId);
        sequences.erase(it);
    } else {
        SPDLOG_LOGGER_DEBUG(sequence_manager_logger, "Model {} version {} Sequence with provided ID does not exists", modelName, modelVersion);
        return StatusCode::SEQUENCE_MISSING;
    }
    return StatusCode::OK;
}

Status SequenceManager::processRequestedSpec(SequenceProcessingSpec& sequenceProcessingSpec) {
    const uint32_t sequenceControlInput = sequenceProcessingSpec.getSequenceControlInput();
    const uint64_t sequenceId = sequenceProcessingSpec.getSequenceId();
    Status status;

    if (sequenceControlInput == SEQUENCE_START) {
        status = createSequence(sequenceProcessingSpec);
    } else if (sequenceControlInput == NO_CONTROL_INPUT) {
        status = hasSequence(sequenceId);
    } else {  // sequenceControlInput == SEQUENCE_END
        status = terminateSequence(sequenceId);
    }
    return status;
}

}  // namespace ovms

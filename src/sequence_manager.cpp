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

#include <spdlog/spdlog.h>

namespace ovms {
bool SequenceManager::hasSequence(uint64_t sequenceId) const {
    return sequences.count(sequenceId);
}

Status SequenceManager::addSequence(uint64_t sequenceId) {
    if (sequences.count(sequenceId)) {
        spdlog::debug("Sequence with provided ID already exists");
        return StatusCode::SEQUENCE_ALREADY_EXISTS;
    } else {
        spdlog::debug("Adding new sequence with ID: {}", sequenceId);
        sequences[sequenceId] = Sequence();
    }
    return StatusCode::OK;
}

Status SequenceManager::removeSequence(uint64_t sequenceId) {
    if (sequences.count(sequenceId)) {
        // TO DO: care for thread safety
        spdlog::debug("Removing sequence with ID: {}", sequenceId);
        sequences.erase(sequenceId);
    } else {
        spdlog::debug("Sequence with provided ID does not exists");
        return StatusCode::SEQUENCE_MISSING;
    }
    return StatusCode::OK;
}

Status SequenceManager::removeTimedOutSequences(std::chrono::steady_clock::time_point currentTime) {
    return StatusCode::OK;
}

const std::mutex& SequenceManager::getMutexRef() const {
    return mutex;
}

const sequence_memory_state_t& SequenceManager::getSequenceMemoryState(uint64_t sequenceId) const {
    return sequences.at(sequenceId).getMemoryState();
}

Status SequenceManager::updateSequenceMemoryState(uint64_t sequenceId, model_memory_state_t& newState) {
    return sequences.at(sequenceId).updateMemoryState(newState);
}
}  // namespace ovms

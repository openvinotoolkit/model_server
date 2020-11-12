//*****************************************************************************
// Copyright 2020 Intel Corporation
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


using namespace InferenceEngine;

namespace ovms {

// SEQUENCE METHODS IMPLEMENTATION
const sequence_memory_state_t& Sequence::getLastMemoryState() {
    return lastMemoryState;
}

Status Sequence::updateLastMemoryState(model_memory_state_t& newState) {
    for (auto && state : newState) {
        auto stateName = state.GetName();
        Blob::CPtr originalBlobPtr = state.GetLastState();
        Blob::Ptr copyBlobPtr = constBlobClone(originalBlobPtr);
        lastMemoryState[stateName] = copyBlobPtr;
    }
    return StatusCode::OK;
}

// SEQUENCE MANAGER METHODS IMPLEMENTATION

Status SequenceManager::addSequence(uint64_t sequenceId) {
    if (sequences.count(sequenceId)) {
        spdlog::info("Sequence with provided ID already exists");
        // return StatusCode::SEQUENCE_ALREADY_EXISTS
    } else {
        spdlog::info("Adding new sequence with ID: {}", sequenceId);
        sequences[sequenceId] = Sequence();
    }
    return StatusCode::OK;
}

Status SequenceManager::removeSequence(uint64_t sequenceId) {
    if (sequences.count(sequenceId)) {
        // TO DO: care for thread safety
        spdlog::info("Removing sequence with ID: {}", sequenceId);
        sequences.erase(sequenceId);
    } else {
        spdlog::info("Sequence with provided ID does not exists");
        // return StatusCode::SEQUENCE_NOT_FOUND
    }
    return StatusCode::OK;
}

const sequence_memory_state_t& SequenceManager::getSequenceMemoryState(uint64_t sequenceId) {
    spdlog::info("Obtaining sequence state for the sequence with ID: {}", sequenceId);
    return sequences[sequenceId].getLastMemoryState();
}

Status SequenceManager::updateSequenceMemoryState(uint64_t sequenceId, model_memory_state_t& newState) {
    spdlog::info("Updating sequence state for the sequence with ID: {}", sequenceId);
    return sequences[sequenceId].updateLastMemoryState(newState);
}

} //namespace ovms
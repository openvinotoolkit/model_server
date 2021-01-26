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

protected:
    std::unordered_map<uint64_t, Sequence> sequences;

public:
    SequenceManager() = default;
    SequenceManager(uint32_t timeout, uint32_t maxSequenceNumber) :
        timeout(timeout),
        maxSequenceNumber(maxSequenceNumber) {}

    uint64_t getSequencesCount() {
        return sequences.size();
    }

    const uint32_t getTimeout() const;

    void setTimeout(uint32_t timeout);

    const uint32_t getMaxSequenceNumber() const;

    void setMaxSequenceNumber(uint32_t maxSequenceNumber);

    std::mutex& getMutex();

    bool sequenceExists(const uint64_t& sequenceId) const;

    Status removeSequence(const uint64_t& sequenceId);

    Status removeTimedOutSequences(std::chrono::steady_clock::time_point currentTime);

    Status addSequence(const uint64_t& sequenceId);

    Status hasSequence(const uint64_t& sequenceId, MutexPtr& sequenceMutexPtr);

    Status createSequence(const uint64_t& sequenceId, MutexPtr& sequenceMutexPtr);

    Status terminateSequence(const uint64_t& sequenceId, MutexPtr& sequenceMutexPtr);

    Status getSequenceMutexPtr(SequenceProcessingSpec& sequenceProcessingSpec, MutexPtr& sequenceMutexPtr);

    const sequence_memory_state_t& getSequenceMemoryState(uint64_t sequenceId) const;

    Status updateSequenceMemoryState(uint64_t sequenceId, model_memory_state_t& newState);
};
}  // namespace ovms

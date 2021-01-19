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
#include <unordered_map>

#include "sequence.hpp"
#include "status.hpp"

namespace ovms {
class SequenceManager {
private:
    std::unordered_map<uint64_t, Sequence> sequences;
    uint32_t timeout;
    uint32_t maxSequenceNumber;
    std::mutex mutex;

public:
    SequenceManager() = default;
    SequenceManager(uint32_t timeout, uint32_t maxSequenceNumber) :
        timeout(timeout),
        maxSequenceNumber(maxSequenceNumber) {}

    const uint32_t getTimeout() const;
    void setTimeout(uint32_t timeout);
    const uint32_t getMaxSequenceNumber() const;
    void setMaxSequenceNumber(uint32_t maxSequenceNumber);
    const std::mutex& getMutexRef() const;

    bool hasSequence(uint64_t sequenceId) const;
    Status addSequence(uint64_t sequenceId);
    Status removeSequence(uint64_t sequenceId);
    Status removeTimedOutSequences(std::chrono::steady_clock::time_point currentTime);
    const sequence_memory_state_t& getSequenceMemoryState(uint64_t sequenceId) const;
    Status updateSequenceMemoryState(uint64_t sequenceId, model_memory_state_t& newState);
};
}  // namespace ovms

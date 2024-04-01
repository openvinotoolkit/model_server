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
#include <string>
#include <unordered_map>
#include <utility>

#include "modelversion.hpp"
#include "sequence.hpp"

namespace ovms {

const uint32_t NO_CONTROL_INPUT = 0;
const uint32_t SEQUENCE_START = 1;
const uint32_t SEQUENCE_END = 2;

class SequenceProcessingSpec;
class Status;

class SequenceManager {
private:
    uint32_t maxSequenceNumber;
    std::string modelName;
    model_version_t modelVersion;
    std::mutex mutex;

protected:
    std::unordered_map<uint64_t, Sequence> sequences;

    uint64_t sequenceIdCounter;

    uint64_t getUniqueSequenceId();

    Status hasSequence(const uint64_t sequenceId);

    Status createSequence(SequenceProcessingSpec& sequenceProcessingSpec);

    Status terminateSequence(const uint64_t sequenceId);

public:
    SequenceManager() = default;
    SequenceManager(uint32_t maxSequenceNumber, std::string modelName, model_version_t modelVersion) :
        maxSequenceNumber(maxSequenceNumber),
        modelName(std::move(modelName)),
        modelVersion(modelVersion),
        sequenceIdCounter(1) {}

    uint64_t getSequencesCount() {
        return sequences.size();
    }

    const uint32_t getMaxSequenceNumber() const;

    void setMaxSequenceNumber(uint32_t maxSequenceNumber);

    std::mutex& getMutex();

    bool sequenceExists(const uint64_t sequenceId) const;

    Sequence& getSequence(const uint64_t sequenceId);

    Status removeSequence(const uint64_t sequenceId);

    Status removeIdleSequences();

    Status processRequestedSpec(SequenceProcessingSpec& sequenceProcessingSpec);
};
}  // namespace ovms

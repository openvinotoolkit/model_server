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
#pragma once

#include <spdlog/spdlog.h>

#include "ov_utils.hpp"
#include "status.hpp"

namespace ovms {

using sequence_memory_state_t = std::unordered_map<std::string, InferenceEngine::Blob::Ptr>;
using model_memory_state_t = std::vector<InferenceEngine::MemoryState>;

class Sequence {
private:
	sequence_memory_state_t lastMemoryState;
	// lastActivityTime
public:
	const sequence_memory_state_t& getLastMemoryState();
	Status updateLastMemoryState(model_memory_state_t& newState);
};


class SequenceManager {
private:
	std::unordered_map<uint64_t, Sequence> sequences;
	// timeout
	// maxSequenceNumber
public:
    // Sequence& getSequence(uint64_t sequenceId);
	Status addSequence(uint64_t sequenceId);
	Status removeSequence(uint64_t sequenceId);
	// removeTimedOutSequences(time_t currentTime);
	const sequence_memory_state_t& getSequenceMemoryState(uint64_t sequenceId);
	Status updateSequenceMemoryState(uint64_t sequenceId, model_memory_state_t& newState);
};

}  // namespace ovms

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
#include "sequence.hpp"

#include <utility>

#include "ov_utils.hpp"
#include "status.hpp"

namespace ovms {

const uint64_t Sequence::getId() const {
    return sequenceId;
}

const sequence_memory_state_t& Sequence::getMemoryState() const {
    return memoryState;
}

const bool Sequence::isIdle() const {
    return idle;
}

void Sequence::setIdle(bool idle) {
    this->idle = idle;
}

Status Sequence::updateMemoryState(model_memory_state_t& newState) {
    for (auto&& state : newState) {
        auto stateName = state.get_name();
        ov::Tensor tensor = state.get_state();
        ov::Tensor copyTensor;
        auto status = tensorClone(copyTensor, tensor);
        if (!status.ok()) {
            return status;
        }
        memoryState[stateName] = copyTensor;
    }
    setIdle(false);
    return StatusCode::OK;
}

std::mutex& Sequence::getMutex() {
    return mutex;
}

bool Sequence::isTerminated() const {
    return terminated;
}

void Sequence::setTerminated() {
    this->terminated = true;
}
}  // namespace ovms

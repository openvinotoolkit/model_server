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

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#include "ov_utils.hpp"
#include "status.hpp"

namespace ovms {

using sequence_memory_state_t = std::unordered_map<std::string, InferenceEngine::Blob::Ptr>;
using sequence_memory_state_t_2 = std::unordered_map<std::string, std::shared_ptr<ov::runtime::Tensor>>;
using model_memory_state_t = std::vector<InferenceEngine::VariableState>;
using model_memory_state_t_2 = std::vector<ov::runtime::VariableState>;

class Sequence {
private:
    uint64_t sequenceId;
    sequence_memory_state_t memoryState;
    sequence_memory_state_t_2 memoryState_2;
    std::mutex mutex;
    bool terminated;
    bool idle;

public:
    Sequence(uint64_t sequenceId) :
        sequenceId(sequenceId),
        terminated(false),
        idle(false) {}
    const sequence_memory_state_t& getMemoryState() const;
    const sequence_memory_state_t_2& getMemoryState_2() const;
    const uint64_t getId() const;
    const bool isIdle() const;
    void setIdle(bool idle = true);
    // In case updateMemoryState returns non-OK status code the sequence should be dropped
    Status updateMemoryState(model_memory_state_t& newState);
    Status updateMemoryState_2(model_memory_state_t_2& newState);
    std::mutex& getMutex();
    bool isTerminated() const;
    void setTerminated();
};

}  // namespace ovms

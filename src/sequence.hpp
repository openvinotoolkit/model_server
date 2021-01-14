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

#include <spdlog/spdlog.h>

#include "ov_utils.hpp"
#include "status.hpp"

namespace ovms {

using sequence_memory_state_t = std::unordered_map<std::string, InferenceEngine::Blob::Ptr>;
using model_memory_state_t = std::vector<InferenceEngine::VariableState>;

class Sequence {
private:
    sequence_memory_state_t memoryState;
    std::chrono::steady_clock::time_point lastActivityTime;
    std::unique_ptr<std::mutex> mutex;

    void updateLastActivityTime();

public:
    Sequence() :
        mutex(new std::mutex()) {}
    const sequence_memory_state_t& getMemoryState() const;
    // In case updateMemoryState returns non-OK status code the sequence should be dropped
    Status updateMemoryState(model_memory_state_t& newState);
    std::chrono::steady_clock::time_point getLastActivityTime() const;
    const std::unique_ptr<std::mutex>& getMutexRef() const;
    std::unique_ptr<std::mutex>&& moveMutex();
};

}  // namespace ovms

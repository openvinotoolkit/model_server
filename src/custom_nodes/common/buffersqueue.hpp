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

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "../../queue.hpp"

namespace ovms {
namespace custom_nodes_common {

class BuffersQueue : protected Queue<char*> {
    size_t singleBufferSize;
    size_t size;
    std::unique_ptr<char[]> memoryPool;

public:
    BuffersQueue(size_t singleBufferSize, int streamsLength) :
        Queue(streamsLength),
        singleBufferSize(singleBufferSize),
        size(singleBufferSize * streamsLength),
        memoryPool(std::make_unique<char[]>(size)) {
        for (int i = 0; i < streamsLength; ++i) {
            inferRequests.push_back(memoryPool.get() + i * singleBufferSize);
        }
    }

    void* getBuffer() {
        // can be easily switched to async version if need arise
        auto idleId = getIdleStream();
        if (idleId.wait_for(std::chrono::nanoseconds(0)) == std::future_status::timeout) {
            return nullptr;
        }
        return getInferRequest(idleId.get());
    }
    
    bool returnBuffer(void* buffer) {
        if ((static_cast<char*>(buffer) < memoryPool.get()) ||
            ((memoryPool.get() + size - 1) < buffer) ||
            ((static_cast<char*>(buffer) - memoryPool.get()) % singleBufferSize != 0)) {
            return false;
        }
        returnStream(getBufferId(buffer));
        return true;
    }

    size_t getSize() {
        return this->size;
    }

    size_t getSingleBufferSize() {
        return this->singleBufferSize;
    }

private:
    int getBufferId(void* buffer) {
        return (static_cast<char*>(buffer) - memoryPool.get()) / singleBufferSize;
    }
};
}  // namespace custom_nodes_common
}  // namespace ovms

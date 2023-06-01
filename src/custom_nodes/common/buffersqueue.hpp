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
    BuffersQueue(size_t singleBufferSize, int streamsLength);
    ~BuffersQueue();
    void* getBuffer();
    bool returnBuffer(void* buffer);
    const size_t getSize();
    const size_t getSingleBufferSize();

private:
    int getBufferId(void* buffer);
};
}  // namespace custom_nodes_common
}  // namespace ovms

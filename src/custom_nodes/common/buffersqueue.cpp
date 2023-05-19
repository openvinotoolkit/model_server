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

#include "buffersqueue.hpp"

#include <iostream>

namespace ovms {
namespace custom_nodes_common {
BuffersQueue::BuffersQueue(size_t singleBufferSize, int streamsLength) :
    Queue(streamsLength),
    singleBufferSize(singleBufferSize),
    size(singleBufferSize * streamsLength),
    memoryPool(std::make_unique<char[]>(size)) {
    for (int i = 0; i < streamsLength; ++i) {
        inferRequests.push_back(memoryPool.get() + i * singleBufferSize);
    }
}

BuffersQueue::~BuffersQueue() {}

void* BuffersQueue::getBuffer() {
    // can be easily switched to async version if need arise
    auto idleId = tryToGetIdleStream();
    if (idleId.has_value()) {
        return getInferRequest(idleId.value());
    }
    return nullptr;
}

bool BuffersQueue::returnBuffer(void* buffer) {
    if ((static_cast<char*>(buffer) < memoryPool.get()) ||
        ((memoryPool.get() + size - 1) < buffer) ||
        ((static_cast<char*>(buffer) - memoryPool.get()) % singleBufferSize != 0)) {
        return false;
    }
    returnStream(getBufferId(buffer));
    return true;
}

int BuffersQueue::getBufferId(void* buffer) {
    return (static_cast<char*>(buffer) - memoryPool.get()) / singleBufferSize;
}

const size_t BuffersQueue::getSize() {
    return this->size;
}

const size_t BuffersQueue::getSingleBufferSize() {
    return this->singleBufferSize;
}
}  // namespace custom_nodes_common
}  // namespace ovms

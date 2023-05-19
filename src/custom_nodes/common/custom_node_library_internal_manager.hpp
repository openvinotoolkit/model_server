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
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "../../custom_node_interface.h"
#include "../common/buffersqueue.hpp"

namespace ovms {
namespace custom_nodes_common {

class CustomNodeLibraryInternalManager {
    std::unordered_map<std::string, std::unique_ptr<BuffersQueue>> outputBuffers;
    std::shared_timed_mutex internalManagerLock;

public:
    CustomNodeLibraryInternalManager();
    ~CustomNodeLibraryInternalManager();
    bool createBuffersQueue(const std::string& name, size_t singleBufferSize, int streamsLength);
    bool recreateBuffersQueue(const std::string& name, size_t singleBufferSize, int streamsLength);
    BuffersQueue* getBuffersQueue(const std::string& name);
    bool releaseBuffer(void* ptr);
    std::shared_timed_mutex& getInternalManagerLock();
};
}  // namespace custom_nodes_common
}  // namespace ovms

template <typename T>
bool get_buffer(ovms::custom_nodes_common::CustomNodeLibraryInternalManager* internalManager, T** buffer, const char* buffersQueueName, uint64_t byte_size) {
    auto buffersQueue = internalManager->getBuffersQueue(buffersQueueName);
    if (!(buffersQueue == nullptr)) {
        *buffer = static_cast<T*>(buffersQueue->getBuffer());
    }
    if (*buffer == nullptr || buffersQueue == nullptr) {
        *buffer = (T*)malloc(byte_size);
        if (*buffer == nullptr) {
            return false;
        }
    }
    return true;
}

void cleanup(CustomNodeTensor& tensor, ovms::custom_nodes_common::CustomNodeLibraryInternalManager* internalManager);

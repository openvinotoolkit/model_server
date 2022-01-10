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

#include "custom_node_library_internal_manager.hpp"

#include <iostream>
#include <memory>
#include <shared_mutex>
#include <string>

namespace ovms {
namespace custom_nodes_common {
bool CustomNodeLibraryInternalManager::createBuffersQueue(const std::string& name, size_t singleBufferSize, int streamsLength) {
    auto it = outputBuffers.find(name);
    if (it != outputBuffers.end()) {
        return false;
    }
    outputBuffers.emplace(name, std::make_unique<BuffersQueue>(singleBufferSize, streamsLength));
    return true;
}

bool CustomNodeLibraryInternalManager::recreateBuffersQueue(const std::string& name, size_t singleBufferSize, int streamsLength) {
    auto it = outputBuffers.find(name);
    if (it != outputBuffers.end()) {
        if (!(it->second->getSize() == singleBufferSize &&
                it->second->getSingleBufferSize() == streamsLength * singleBufferSize)) {
            it->second.reset(new BuffersQueue(singleBufferSize, streamsLength));
        }
        return true;
    }
    return false;
}

BuffersQueue* CustomNodeLibraryInternalManager::getBuffersQueue(const std::string& name) {
    auto it = outputBuffers.find(name);
    if (it == outputBuffers.end())
        return nullptr;
    return it->second.get();
}

bool CustomNodeLibraryInternalManager::releaseBuffer(void* ptr) {
    for (auto it = outputBuffers.begin(); it != outputBuffers.end(); ++it) {
        if (it->second->returnBuffer(ptr)) {
            return true;
        }
    }
    return false;
}

std::shared_timed_mutex& CustomNodeLibraryInternalManager::getInternalManagerLock() {
    return this->internalManagerLock;
}

template <typename T>
bool get_buffer(CustomNodeLibraryInternalManager* internalManager, T** buffer, const char* buffersQueueName, uint64_t byte_size) {
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

void cleanup(CustomNodeTensor& tensor, CustomNodeLibraryInternalManager* internalManager) {
    release(tensor.data, internalManager);
    release(tensor.dims, internalManager);
}
}  // namespace custom_nodes_common
}  // namespace ovms

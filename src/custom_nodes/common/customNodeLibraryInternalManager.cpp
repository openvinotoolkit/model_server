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

#include "customNodeLibraryInternalManager.hpp"

#include <iostream>

namespace ovms {
namespace custom_nodes_common {
void CustomNodeLibraryInternalManager::createBuffersQueue(std::string name, size_t singleBufferSize, int streamsLength) {
    auto it = outputBuffers.find(name);
    if (it != outputBuffers.end()) {
        delete it->second;
        outputBuffers.erase(it);
    }
    outputBuffers.insert({name, new BuffersQueue(singleBufferSize, streamsLength)});
}

BuffersQueue* CustomNodeLibraryInternalManager::getBuffersQueue(std::string name) {
    return outputBuffers.find(name)->second;
}

int CustomNodeLibraryInternalManager::releaseBuffer(void* ptr) {
    for (auto it = outputBuffers.begin(); it != outputBuffers.end();) {
        if (it->second->returnBuffer(ptr)) {
            return 0;
        }
        ++it;
    }
    return 1;
}

void CustomNodeLibraryInternalManager::clear() {
    for (auto it = outputBuffers.begin(); it != outputBuffers.end();) {
        delete it->second;
        outputBuffers.erase(it);
        ++it;
    }
}
}  // namespace custom_nodes_common
}  // namespace ovms

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

#include <string>
#include <unordered_map>

#include "../common/buffersqueue.hpp"

namespace ovms {
namespace custom_nodes_common {

class CustomNodeLibraryInternalManager {
    std::unordered_map<std::string, std::unique_ptr<BuffersQueue>> outputBuffers;

public:
    CustomNodeLibraryInternalManager() = default;
    bool createBuffersQueue(const std::string& name, size_t singleBufferSize, int streamsLength);
    bool recreateBuffersQueue(const std::string& name, size_t singleBufferSize, int streamsLength);
    BuffersQueue* getBuffersQueue(const std::string& name);
    bool releaseBuffer(void* ptr);
};
}  // namespace custom_nodes_common
}  // namespace ovms

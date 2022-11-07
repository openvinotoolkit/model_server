//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include "buffer.hpp"

#include <cstring>

#include "logging.hpp"

namespace ovms {
Buffer::Buffer(const void* pptr, size_t byteSize, BufferType bufferType, std::optional<uint32_t> bufferDeviceId, bool createCopy) :
    ptr(createCopy ? nullptr : pptr),
    byteSize(byteSize),
    bufferType(bufferType),
    bufferDeviceId(bufferDeviceId) {
    // TODO in later stages
    // it can be advantageous to have custom made release fn especially with buffers passed by external users
    if (!createCopy)
        return;
    ownedCopy = std::make_unique<char[]>(byteSize);
    std::memcpy(ownedCopy.get(), pptr, byteSize);
}
const void* Buffer::data() const {
    return (ptr != nullptr) ? ptr : ownedCopy.get();
}
Buffer::~Buffer() = default;
}  // namespace ovms

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

namespace ovms {
Buffer::Buffer(const void* pptr, size_t byteSize, OVMS_BufferType bufferType, std::optional<uint32_t> bufferDeviceId, bool createCopy) :
    ptr(createCopy ? nullptr : pptr),
    byteSize(byteSize),
    bufferType(bufferType),
    bufferDeviceId(bufferDeviceId) {
    if (!createCopy)
        return;
    ownedCopy = std::make_unique<char[]>(byteSize);
    std::memcpy(ownedCopy.get(), pptr, byteSize);
}
Buffer::Buffer(size_t byteSize, OVMS_BufferType bufferType, std::optional<uint32_t> bufferDeviceId) :
    ptr(nullptr),
    byteSize(byteSize),
    bufferType(bufferType),
    bufferDeviceId(bufferDeviceId) {
    ownedCopy = std::make_unique<char[]>(byteSize);
}

const void* Buffer::data() const {
    return (ptr != nullptr) ? ptr : ownedCopy.get();
}

void* Buffer::data() {
    return (ptr != nullptr) ? nullptr : ownedCopy.get();
}

size_t Buffer::getByteSize() const {
    return byteSize;
}

OVMS_BufferType Buffer::getBufferType() const {
    return this->bufferType;
}

const std::optional<uint32_t>& Buffer::getDeviceId() const {
    return bufferDeviceId;
}

Buffer::~Buffer() = default;
}  // namespace ovms

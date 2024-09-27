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
#include <utility>

#include "../logging.hpp"

namespace ovms {
Buffer::Buffer(std::unique_ptr<std::vector<std::string>>&& values) :
    byteSize(values->size() * sizeof(std::string)),
    bufferType(OVMS_BUFFERTYPE_CPU),
    stringVec(std::move(values)),
    ptr(stringVec->data()) {
}
Buffer::Buffer(const void* pptr, size_t byteSize, OVMS_BufferType bufferType, std::optional<uint32_t> bufferDeviceId, bool createCopy) :
    byteSize(byteSize),
    bufferType(bufferType),
    bufferDeviceId(bufferDeviceId),
    ownedCopy(createCopy ? std::make_unique<char[]>(byteSize) : nullptr),
    ptr(createCopy ? ownedCopy.get() : const_cast<void*>(pptr)) {
    if (!createCopy)
        return;
    std::memcpy(ownedCopy.get(), pptr, byteSize);
}
Buffer::Buffer(size_t byteSize, OVMS_BufferType bufferType, std::optional<uint32_t> bufferDeviceId) :
    byteSize(byteSize),
    bufferType(bufferType),
    bufferDeviceId(bufferDeviceId),
    ownedCopy(std::make_unique<char[]>(byteSize)),
    ptr(ownedCopy.get()) {
}
const void* Buffer::data() const {
    return ptr;
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

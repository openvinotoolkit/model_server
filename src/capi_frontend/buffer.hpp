#pragma once
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
#include <memory>
#include <optional>

#include "../ovms.h"  // NOLINT
namespace ovms {

class Buffer {
    const void* ptr;
    size_t byteSize;
    OVMS_BufferType bufferType;
    std::optional<uint32_t> bufferDeviceId;
    std::unique_ptr<char[]> ownedCopy = nullptr;

public:
    Buffer(const void* ptr, size_t byteSize, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU, std::optional<uint32_t> bufferDeviceId = std::nullopt, bool createCopy = false);
    Buffer(size_t byteSize, OVMS_BufferType bufferType = OVMS_BUFFERTYPE_CPU, std::optional<uint32_t> bufferDeviceId = std::nullopt);
    ~Buffer();
    const void* data() const;
    void* data();
    OVMS_BufferType getBufferType() const;
    const std::optional<uint32_t>& getDeviceId() const;
    size_t getByteSize() const;
};

}  // namespace ovms

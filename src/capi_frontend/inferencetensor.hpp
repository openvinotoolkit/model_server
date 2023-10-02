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
#include <string>
#include <vector>

#include "../ovms.h"  // NOLINT
#include "../shape.hpp"

namespace ovms {
class Buffer;
class Status;

class InferenceTensor {
    const OVMS_DataType datatype;
    signed_shape_t shape;
    std::unique_ptr<Buffer> buffer;

public:
    InferenceTensor(OVMS_DataType datatype, const int64_t* shape, size_t dimCount);
    ~InferenceTensor();
    InferenceTensor(InferenceTensor&&);
    InferenceTensor(const InferenceTensor&) = delete;
    InferenceTensor& operator=(const InferenceTensor&) = delete;
    InferenceTensor& operator=(const InferenceTensor&&);
    Status setBuffer(const void* addr, size_t byteSize, OVMS_BufferType bufferType, std::optional<uint32_t> deviceId, bool createCopy = false);
    Status setBuffer(std::unique_ptr<Buffer>&& buffer);
    Status removeBuffer();
    OVMS_DataType getDataType() const;
    const signed_shape_t& getShape() const;
    const Buffer* const getBuffer() const;
};
}  // namespace ovms

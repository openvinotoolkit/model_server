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
#include "inferencetensor.hpp"

#include <utility>

#include "../ovms.h"  // NOLINT
#include "../status.hpp"
#include "buffer.hpp"

namespace ovms {
InferenceTensor::~InferenceTensor() = default;
InferenceTensor::InferenceTensor(InferenceTensor&& rhs) :
    datatype(std::move(rhs.datatype)),
    shape(std::move(rhs.shape)),
    buffer(std::move(rhs.buffer)) {}
InferenceTensor::InferenceTensor(OVMS_DataType datatype, const int64_t* shape, size_t dimCount) :
    datatype(datatype),
    shape(shape, shape + dimCount) {}

Status InferenceTensor::setBuffer(const void* addr, size_t byteSize, OVMS_BufferType bufferType, std::optional<uint32_t> deviceId, bool createCopy) {
    if (nullptr != this->buffer) {
        return StatusCode::DOUBLE_BUFFER_SET;
    }
    this->buffer = std::make_unique<Buffer>(addr, byteSize, bufferType, deviceId, createCopy);
    return StatusCode::OK;
}

Status InferenceTensor::setBuffer(std::unique_ptr<Buffer>&& buffer) {
    if (nullptr != this->buffer) {
        return StatusCode::DOUBLE_BUFFER_SET;
    }
    this->buffer = std::move(buffer);
    return StatusCode::OK;
}

OVMS_DataType InferenceTensor::getDataType() const {
    return this->datatype;
}
const signed_shape_t& InferenceTensor::getShape() const {
    return this->shape;
}
const Buffer* const InferenceTensor::getBuffer() const {
    return this->buffer.get();
}
Status InferenceTensor::removeBuffer() {
    if (nullptr != this->buffer) {
        this->buffer.reset();
        return StatusCode::OK;
    }
    return StatusCode::NONEXISTENT_BUFFER_FOR_REMOVAL;
}
}  // namespace ovms

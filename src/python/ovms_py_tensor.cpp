//*****************************************************************************
// Copyright 2023 Intel Corporation
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

#include "ovms_py_tensor.hpp"

#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ovms;

OvmsPyTensor::OvmsPyTensor(const std::string& name, const py::buffer& buffer, const std::optional<std::vector<py::ssize_t>>& shape, const std::optional<std::string>& datatype) :
    name(name),
    refObj(buffer) {
    py::buffer_info bufferInfo = buffer.request();
    ptr = bufferInfo.ptr;
    bufferShape = bufferInfo.shape;
    ndim = bufferInfo.ndim;
    format = bufferInfo.format;
    itemsize = bufferInfo.itemsize;
    strides = bufferInfo.strides;

    size = std::accumulate(std::begin(bufferShape), std::end(bufferShape), 1, std::multiplies<py::ssize_t>()) * itemsize;

    userShape = shape.value_or(bufferShape);

    if (datatype.has_value()) {
        this->datatype = datatype.value();
    } else {
        auto it = bufferFormatToDatatype.find(format);
        this->datatype = it != bufferFormatToDatatype.end() ? it->second : format;
    }
}

OvmsPyTensor::OvmsPyTensor(const std::string& name, const std::vector<py::ssize_t>& shape, const std::string& datatype, py::ssize_t size, bool allocate) :
    name(name),
    datatype(datatype),
    userShape(shape),
    size(size) {
    // Map datatype to struct syntax format if it's known. Otherwise assume raw binary (UINT8 type)
    auto it = datatypeToBufferFormat.find(datatype);
    if (it != datatypeToBufferFormat.end()) {
        format = it->second;
        bufferShape = userShape;
    } else {
        format = RAW_BINARY_FORMAT;
        bufferShape = std::vector<py::ssize_t>{size};
    }

    ndim = bufferShape.size();
    itemsize = bufferFormatToItemsize.at(format);
    if (ndim > 0) {
        strides.insert(strides.begin(), itemsize);
        for (int i = 1; i < ndim; i++) {
            py::ssize_t stride = bufferShape[ndim - i] * strides[0];
            strides.insert(strides.begin(), stride);
        }
    }
    if (allocate) {
        ownedDataPtr = std::make_unique<char[]>(size);
        ptr = this->ownedDataPtr.get();
    } else {
        ptr = nullptr;
    }
}

OvmsPyTensor::OvmsPyTensor(const std::string& name, void* data, const std::vector<py::ssize_t>& shape, const std::string& datatype, py::ssize_t size, bool copy) :
    OvmsPyTensor(name, shape, datatype, size, /*allocate=*/copy) {
    if (copy) {
        memcpy(this->ownedDataPtr.get(), data, size);
    } else {
        ptr = data;
    }
}

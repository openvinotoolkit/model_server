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
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace ovms;

OvmsPyTensor::OvmsPyTensor(const std::string& name, void* data, const std::vector<py::ssize_t>& shape, const std::string& datatype, py::ssize_t size, bool copy) :
    name(name),
    datatype(datatype),
    userShape(shape),
    size(size) {
    // Map datatype to struct syntax format if it's known. Otherwise assume raw binary (UINT8 type)
    std::cout << "Calling OvmsPyTensor constructor from data" << std::endl;
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
    if (copy) {
        ownedDataPtr = std::make_unique<char[]>(size);
        memcpy(this->ownedDataPtr.get(), data, size);
        ptr = this->ownedDataPtr.get();
    } else {
        ptr = data;
    }
}

OvmsPyTensor::OvmsPyTensor(const std::string& name, const py::buffer& buffer) :
    name(name),
    refObj(buffer) {
    std::cout << "Calling OvmsPyTensor constructor from buffer" << std::endl;
    py::buffer_info bufferInfo = buffer.request();
    ptr = bufferInfo.ptr;
    bufferShape = bufferInfo.shape;
    ndim = bufferInfo.ndim;
    format = bufferInfo.format;
    itemsize = bufferInfo.itemsize;
    strides = bufferInfo.strides;

    size = std::accumulate(std::begin(bufferShape), std::end(bufferShape), 1, std::multiplies<py::ssize_t>()) * itemsize;
    userShape = bufferShape;
    datatype = format;
    auto it = bufferFormatToDatatype.find(format);
    datatype = it != datatypeToBufferFormat.end() ? it->second : format;
}

OvmsPyTensor::~OvmsPyTensor() {
    std::cout << "Calling OvmsPyTensor destructor" << std::endl;
}

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

OvmsPyTensor::OvmsPyTensor(std::string name, void* ptr, std::vector<py::ssize_t> shape, std::string datatype, size_t size) :
    name(name),
    ptr(ptr),
    shape(shape),
    ndim(shape.size()),
    format(),
    itemsize(),
    datatype(datatype),
    size(size) {
    // Map datatype to struct syntax format if it's known. Otherwise assume raw binary (UINT8 type)
    auto it = datatypeToBufferFormat.find(datatype);
    format = it != datatypeToBufferFormat.end() ? it->second : RAW_BINARY_FORMAT;

    itemsize = bufferFormatToItemsize.at(format);
    strides.insert(strides.begin(), itemsize);
    for (int i = 1; i < ndim; i++) {
        py::ssize_t stride = shape[ndim - i] * strides[0];
        strides.insert(strides.begin(), stride);
    }
}

OvmsPyTensor::OvmsPyTensor(std::string name, py::buffer_info bufferInfo) :
    name(name),
    ptr(bufferInfo.ptr),
    shape(bufferInfo.shape),
    ndim(bufferInfo.ndim),
    format(bufferInfo.format),
    itemsize(bufferInfo.itemsize),
    strides(bufferInfo.strides) {
    size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<py::ssize_t>()) * itemsize;
    datatype = format;
    auto it = bufferFormatToDatatype.find(format);
    datatype = it != datatypeToBufferFormat.end() ? it->second : format;
}

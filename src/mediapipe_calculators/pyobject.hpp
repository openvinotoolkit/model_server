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

// Declaration common for C++ and Python
// C++ definition part in separate location
// pybind11 binding part also in separate location
#ifndef SRC_MEDIAPIPE_CALCULATORS_PYOBJECT_HPP_
#define SRC_MEDIAPIPE_CALCULATORS_PYOBJECT_HPP_
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <pybind11/embed.h>  // everything needed for embedding

namespace py = pybind11;
class PYOBJECT {
    // --- These we get from the request

    void* data;
    std::vector<py::ssize_t> shape;
    // Can be one of predefined types (like int8, float32 etc.) or totally custom like numpy (for example "<U83")
    std::string datatype;

    // ---

    // These we compute if possible
    py::ssize_t itemsize;
    py::ssize_t ndim;
    std::vector<py::ssize_t> strides;

    // ---

    // User interface from Python code
public:
    // Try to create using other Python object that exposes buffer protocol (like numpy array, pytorch tensor etc.)
    PYOBJECT(py::object object);

    // True if object implements protocol buffer. Example when it can be false:
    // Irregular (non-padded) string or other binary data - impossible to determine itemsize and compute strides.
    // Will return True if datatype is predefined type and False for custom type
    bool implements_buffer();

    // Developer interface
private:
    // itemsize deduced from datatype if predefined
    // ndim deduced from shape
    // strides computed from shape and itemsize
    // size might be useful for custom data types handling
    PYOBJECT(void* data, size_t size, std::vector<py::ssize_t> shape, std::string datatype);
    // TODO: fix this -> PYOBJECT(&ov::Tensor);
    ov::Tensor converToOVTensor();
    // TODO: fix this -> bool packToResponse(T& response);
};

#endif // SRC_MEDIAPIPE_CALCULATORS_PYOBJECT_HPP_

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

#include "src/python/ovms_py_tensor.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ovms;

PYBIND11_MODULE(pyovms, m) {
    py::class_<OvmsPyTensor>(m, "Tensor", py::buffer_protocol())
        .def_buffer([](OvmsPyTensor& m) -> py::buffer_info {
            return py::buffer_info(
                m.ptr,
                m.itemsize,
                m.format,
                m.ndim,
                m.bufferShape,
                m.strides,
                true);  // Underlying buffer is readonly
        })
        .def(py::init([](std::string& name, const py::buffer& buffer, const std::optional<std::vector<py::ssize_t>>& shape, const std::optional<std::string>& datatype) {
            return std::make_unique<OvmsPyTensor>(name, buffer, shape, datatype);
        }),
            py::arg("name"),
            py::arg("buffer"),
            py::arg("shape") = std::nullopt,
            py::arg("datatype") = std::nullopt)
        .def_static("_create_from_data", [](const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, const std::string& datatype, py::ssize_t size, bool copy) {
            return std::make_unique<OvmsPyTensor>(name, ptr, shape, datatype, size, copy);
        })
        .def_static("_create_without_data", [](const std::string& name, const std::vector<py::ssize_t>& shape, const std::string& datatype, py::ssize_t size) {
            return std::make_unique<OvmsPyTensor>(name, shape, datatype, size);
        })
        .def_readonly("name", &OvmsPyTensor::name)
        .def_readonly("ptr", &OvmsPyTensor::ptr)
        .def_readonly("size", &OvmsPyTensor::size)
        .def_property_readonly("data", [](const py::object& m) -> py::memoryview { return py::memoryview(m); })
        .def_property_readonly("shape", [](const OvmsPyTensor& m) -> py::tuple { return py::tuple(py::cast(m.userShape)); })
        .def_readonly("datatype", &OvmsPyTensor::datatype)
        .def_readonly("ref_obj", &OvmsPyTensor::refObj);
}

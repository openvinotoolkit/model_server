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

#include "python_backend.hpp"

#include <pybind11/stl.h>

#include "../logging.hpp"

namespace py = pybind11;
using namespace py::literals;
using namespace ovms;

bool PythonBackend::createPythonBackend(PythonBackend** pythonBackend) {
    py::gil_scoped_acquire acquire;
    try {
        *pythonBackend = new PythonBackend();
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_DEBUG("PythonBackend initialization failed: {}", e.what());
        return false;
    } catch (std::exception& e) {
        SPDLOG_DEBUG("PythonBackend initialization failed: {}", e.what());
        return false;
    }
    return true;
}

PythonBackend::PythonBackend() {
    py::gil_scoped_acquire acquire;
    SPDLOG_DEBUG("Creating python backend");
    pyovmsModule = std::make_unique<py::module_>(py::module_::import("pyovms"));
    tensorClass = std::make_unique<py::object>(pyovmsModule->attr("Tensor"));
}

PythonBackend::~PythonBackend() {
    py::gil_scoped_acquire acquire;
    tensorClass.reset();
    pyovmsModule.reset();
}

bool PythonBackend::createOvmsPyTensor(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape,
    const std::string& datatype, py::ssize_t size, std::unique_ptr<PyObjectWrapper<py::object>>& outTensor) {
    py::gil_scoped_acquire acquire;
    try {
        py::object ovmsPyTensor = tensorClass->attr("create_from_data")(name, ptr, shape, datatype, size);
        outTensor = std::make_unique<PyObjectWrapper<py::object>>(ovmsPyTensor);
        return true;
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_DEBUG("PythonBackend::createOvmsPyTensor - Py Error: {}", e.what());
        return false;
    } catch (std::exception& e) {
        SPDLOG_DEBUG("PythonBackend::createOvmsPyTensor - Error: {}", e.what());
        return false;
    }
}

void PythonBackend::validateOvmsPyTensor(const py::object& object) const {
    py::gil_scoped_acquire acquire;
    if (!py::isinstance(object, *tensorClass)) {
        throw UnexpectedPythonObjectError(object, tensorClass->attr("__name__").cast<std::string>());
    }
}

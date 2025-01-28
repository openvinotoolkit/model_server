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

#pragma once
#include <memory>
#include <string>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace ovms {

class PythonBackend {
    std::unique_ptr<py::module_> pyovmsModule;
    std::unique_ptr<py::object> tensorClass;

public:
    PythonBackend();
    ~PythonBackend();
    static bool createPythonBackend(std::unique_ptr<PythonBackend>& pythonBackend);

    bool createOvmsPyTensor(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape, const std::string& datatype,
        py::ssize_t size, std::unique_ptr<PyObjectWrapper<py::object>>& outTensor, bool copy = false);

    bool createEmptyOvmsPyTensor(const std::string& name, const std::vector<py::ssize_t>& shape, const std::string& datatype,
        py::ssize_t size, std::unique_ptr<PyObjectWrapper<py::object>>& outTensor);

    // Checks if object is tensorClass instance. Throws UnexpectedPythonObjectError if it's not.
    void validateOvmsPyTensor(const py::object& object) const;

    bool getOvmsPyTensorData(std::unique_ptr<PyObjectWrapper<py::object>>& outTensor, void** data);
};
}  // namespace ovms

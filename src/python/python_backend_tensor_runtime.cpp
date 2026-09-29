//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include "../logging.hpp"

namespace ovms {

bool PythonBackend::createOvmsPyTensor(const std::string& name, void* ptr, const std::vector<py::ssize_t>& shape,
    const std::string& datatype, py::ssize_t size, std::unique_ptr<PyObjectWrapper<py::object>>& outTensor, bool copy) {
    (void)name;
    (void)ptr;
    (void)shape;
    (void)datatype;
    (void)size;
    (void)outTensor;
    (void)copy;
    SPDLOG_DEBUG("PythonBackend::createOvmsPyTensor runtime bridge is unavailable without embedded Python runtime");
    return false;
}

bool PythonBackend::createEmptyOvmsPyTensor(const std::string& name, const std::vector<py::ssize_t>& shape, const std::string& datatype,
    py::ssize_t size, std::unique_ptr<PyObjectWrapper<py::object>>& outTensor) {
    (void)name;
    (void)shape;
    (void)datatype;
    (void)size;
    (void)outTensor;
    SPDLOG_DEBUG("PythonBackend::createEmptyOvmsPyTensor runtime bridge is unavailable without embedded Python runtime");
    return false;
}

bool PythonBackend::getOvmsPyTensorData(std::unique_ptr<PyObjectWrapper<py::object>>& outTensor, void** data) {
    (void)outTensor;
    (void)data;
    SPDLOG_DEBUG("PythonBackend::getOvmsPyTensorData runtime bridge is unavailable without embedded Python runtime");
    return false;
}

}  // namespace ovms

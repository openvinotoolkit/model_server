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

#if (PYTHON_DISABLE == 0)
#include <pybind11/embed.h>  // everything needed for embedding

#include "src/mediapipe_calculators/python_executor_calculator_options.pb.h"

namespace py = pybind11;
#endif

namespace ovms {
class Status;
class PythonBackend;

class PythonNodeResource {
public:
    PythonNodeResource(const PythonNodeResource&) = delete;
    PythonNodeResource& operator=(PythonNodeResource&) = delete;
#if (PYTHON_DISABLE == 0)
    std::unique_ptr<py::object> nodeResourceObject;
    PythonBackend * pythonBackend;

    PythonNodeResource(PythonBackend* pythonBackend);
    ~PythonNodeResource();
    static Status createPythonNodeResource(std::shared_ptr<PythonNodeResource>& nodeResource, const google::protobuf::Any& nodeOptions,  PythonBackend* pythonBackend);
#endif
};

}  // namespace ovms

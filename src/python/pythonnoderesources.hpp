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
#include <unordered_map>

#include <pybind11/embed.h>  // everything needed for embedding

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop

namespace py = pybind11;

namespace ovms {
class Status;
class PythonBackend;

class PythonNodeResources {
public:
    PythonNodeResources(const PythonNodeResources&) = delete;
    PythonNodeResources& operator=(PythonNodeResources&) = delete;

    std::unique_ptr<py::object> ovmsPythonModel;
    PythonBackend* pythonBackend;
    std::string handlerPath;
    std::unordered_map<std::string, std::string> outputsNameTagMapping;

    PythonNodeResources(PythonBackend* pythonBackend);
    ~PythonNodeResources();

    static Status createPythonNodeResources(std::shared_ptr<PythonNodeResources>& nodeResources, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, PythonBackend* pythonBackend, std::string graphPath);

    void finalize();

private:
    static py::dict preparePythonNodeInitializeArguments(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, const std::string& basePath);
};
using PythonNodeResourcesMap = std::unordered_map<std::string, std::shared_ptr<PythonNodeResources>>;
}  // namespace ovms

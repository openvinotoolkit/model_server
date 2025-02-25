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
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>  // everything needed for embedding
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable : 4309 4005 6001 6011 6326 6385 6246 6386 6326 6011 4005 4456)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#pragma warning(pop)

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

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
#include "pythonnoderesources.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "../logging.hpp"
#include "../status.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#include <pybind11/embed.h>  // everything needed for embedding

#include "../mediapipe_internal/mediapipe_utils.hpp"
#include "src/python/python_executor_calculator.pb.h"

namespace ovms {

PythonNodeResources::PythonNodeResources(PythonBackend* pythonBackend) {
    this->ovmsPythonModel = nullptr;
    this->pythonBackend = pythonBackend;
    this->handlerPath = "";
}

void PythonNodeResources::finalize() {
    if (this->ovmsPythonModel) {
        py::gil_scoped_acquire acquire;
        try {
            if (!py::hasattr(*ovmsPythonModel.get(), "finalize")) {
                SPDLOG_DEBUG("Python node resource does not have a finalize method. Python node handler_path: {} ", this->handlerPath);
                return;
            }

            ovmsPythonModel.get()->attr("finalize")();
        } catch (const pybind11::error_already_set& e) {
            SPDLOG_ERROR("Failed to process python node finalize method. {}  Python node handler_path: {} ", e.what(), this->handlerPath);
            return;
        } catch (...) {
            SPDLOG_ERROR("Failed to process python node finalize method. Python node handler_path: {} ", this->handlerPath);
            return;
        }
    }
}

// IMPORTANT: This is an internal method meant to be run in a specific context.
// It assumes GIL is being held by the thread and doesn't handle potential errors.
// It MUST be called in the scope of py::gil_scoped_acquire and within the try - catch block
py::dict PythonNodeResources::preparePythonNodeInitializeArguments(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, const std::string& basePath) {
    py::dict kwargsParam = py::dict();
    std::string nodeName = graphNodeConfig.name();
    py::list inputStreams = py::list();
    py::list outputStreams = py::list();
    for (auto& name : graphNodeConfig.input_stream()) {
        inputStreams.append(getStreamName(name));
    }

    for (auto& name : graphNodeConfig.output_stream()) {
        outputStreams.append(getStreamName(name));
    }

    kwargsParam["input_names"] = inputStreams;
    kwargsParam["output_names"] = outputStreams;
    kwargsParam["node_name"] = nodeName;
    kwargsParam["base_path"] = py::str(basePath);

    return kwargsParam;
}

PythonNodeResources::~PythonNodeResources() {
    SPDLOG_DEBUG("Calling Python node resource destructor");
    this->finalize();
    py::gil_scoped_acquire acquire;
    this->ovmsPythonModel.reset();
}

}  // namespace ovms

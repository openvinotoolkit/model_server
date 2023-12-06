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
#include "pythonnoderesource.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "../logging.hpp"
#include "../status.hpp"

#if (PYTHON_DISABLE == 0)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_graph.h"
#pragma GCC diagnostic pop
#include <pybind11/embed.h>  // everything needed for embedding

#include "src/mediapipe_calculators/python_executor_calculator_options.pb.h"
#endif

namespace ovms {

#if (PYTHON_DISABLE == 0)
PythonNodeResource::PythonNodeResource(PythonBackend* pythonBackend) {
    this->nodeResourceObject = nullptr;
    this->pythonBackend = pythonBackend;
    this->pythonNodeFilePath = "";
}

void PythonNodeResource::finalize() {
    if (this->nodeResourceObject) {
        py::gil_scoped_acquire acquire;
        try {
            if (!py::hasattr(*nodeResourceObject.get(), "finalize")) {
                SPDLOG_DEBUG("Python node resource does not have a finalize method. Python node path {} ", this->pythonNodeFilePath);
                return;
            }

            nodeResourceObject.get()->attr("finalize")();
        } catch (const pybind11::error_already_set& e) {
            SPDLOG_ERROR("Failed to process python node finalize method. {}  Python node path {} ", e.what(), this->pythonNodeFilePath);
            return;
        } catch (...) {
            SPDLOG_ERROR("Failed to process python node finalize method. Python node path {} ", this->pythonNodeFilePath);
            return;
        }
    } else {
        SPDLOG_ERROR("nodeResourceObject is not initialized. Python node path {} ", this->pythonNodeFilePath);
        throw std::exception();
    }
}

// IMPORTANT: This is an internal method meant to be run in a specific context.
// It assumes GIL is being held by the thread and doesn't handle potential errors.
// It MUST be called in the scope of py::gil_scoped_acquire and within the try - catch block
py::dict PythonNodeResource::preparePythonNodeInitializeArguments(const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig) {
    py::dict kwargsParam = py::dict();
    std::string nodeName = graphNodeConfig.name();
    py::list inputStreams = py::list();
    py::list outputStreams = py::list();
    for (auto& name : graphNodeConfig.input_stream()) {
        inputStreams.append(name);
    }

    for (auto& name : graphNodeConfig.output_stream()) {
        outputStreams.append(name);
    }

    kwargsParam["input_streams"] = inputStreams;
    kwargsParam["output_streams"] = outputStreams;
    kwargsParam["node_name"] = nodeName;

    return kwargsParam;
}

Status PythonNodeResource::createPythonNodeResource(std::shared_ptr<PythonNodeResource>& nodeResource, const ::mediapipe::CalculatorGraphConfig::Node& graphNodeConfig, PythonBackend* pythonBackend) {
    mediapipe::PythonExecutorCalculatorOptions nodeOptions;
    graphNodeConfig.node_options(0).UnpackTo(&nodeOptions);
    if (!std::filesystem::exists(nodeOptions.handler_path())) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Python node file: {} does not exist. ", nodeOptions.handler_path());
        return StatusCode::PYTHON_NODE_FILE_DOES_NOT_EXIST;
    }
    auto fsHandlerPath = std::filesystem::path(nodeOptions.handler_path());
    fsHandlerPath.replace_extension();

    std::string parentPath = fsHandlerPath.parent_path();
    std::string filename = fsHandlerPath.filename();

    py::gil_scoped_acquire acquire;
    try {
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")(parentPath.c_str());
        py::module_ script = py::module_::import(filename.c_str());
        py::object OvmsPythonModel = script.attr("OvmsPythonModel");
        py::object pythonModel = OvmsPythonModel();
        py::dict kwargsParam = preparePythonNodeInitializeArguments(graphNodeConfig);
        pythonModel.attr("initialize")(kwargsParam);

        nodeResource = std::make_shared<PythonNodeResource>(pythonBackend);
        nodeResource->nodeResourceObject = std::make_unique<py::object>(pythonModel);
        nodeResource->pythonNodeFilePath = nodeOptions.handler_path();
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_ERROR("Failed to process python node file {} : {}", nodeOptions.handler_path(), e.what());
        return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Failed to process python node file {}", nodeOptions.handler_path());
        return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
    }
    return StatusCode::OK;
}

PythonNodeResource::~PythonNodeResource() {
    SPDLOG_DEBUG("Calling Python node resource destructor");
    this->finalize();
    py::gil_scoped_acquire acquire;
    this->nodeResourceObject.reset();
}
#endif

}  // namespace ovms

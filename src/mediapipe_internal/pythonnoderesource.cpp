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

#include <spdlog/spdlog.h>

#include "../logging.hpp"
#include "../status.hpp"

#if (PYTHON_DISABLE == 0)
#include <pybind11/embed.h>  // everything needed for embedding

#include "src/mediapipe_calculators/python_backend_calculator.pb.h"
#endif

namespace ovms {

#if (PYTHON_DISABLE == 0)
PythonNodeResource::PythonNodeResource() {
    this->nodeResourceObject = nullptr;
}

Status PythonNodeResource::createPythonNodeResource(std::shared_ptr<PythonNodeResource>& nodeResource, const google::protobuf::Any& nodeOptions) {
    mediapipe::PythonBackendCalculatorOptions options;
    nodeOptions.UnpackTo(&options);
    if (!std::filesystem::exists(options.handler_path())) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Python node file: {} does not exist. ", options.handler_path());
        return StatusCode::PYTHON_NODE_FILE_DOES_NOT_EXIST;
    }
    auto fsHandlerPath = std::filesystem::path(options.handler_path());
    fsHandlerPath.replace_extension();

    std::string parentPath = fsHandlerPath.parent_path();
    std::string filename = fsHandlerPath.filename();

    try {
        py::gil_scoped_acquire acquire;
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")(parentPath.c_str());
        py::module_ script = py::module_::import(filename.c_str());
        py::object OvmsPythonModel = script.attr("OvmsPythonModel");
        py::object pythonModel = OvmsPythonModel();
        py::object kwargsParam = pybind11::dict();
        // TODO: check bool if true
        py::bool_ success = pythonModel.attr("initialize")(kwargsParam);

        if (!success) {
            SPDLOG_ERROR("Python node initialize script call returned false for: {}", options.handler_path());
            return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
        }

        nodeResource = std::make_shared<PythonNodeResource>();
        nodeResource->nodeResourceObject = std::make_unique<py::object>(pythonModel);
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_ERROR("Failed to process python node file {} : {}", options.handler_path(), e.what());
        return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Failed to process python node file {}", options.handler_path());
        return StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
    }

    return StatusCode::OK;
}

PythonNodeResource::~PythonNodeResource() {
    py::gil_scoped_acquire acquire;
    this->nodeResourceObject.get()->dec_ref();
}
#endif

}  // namespace ovms

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
#include "nodestate.hpp"

#include <filesystem>
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
NodeState::NodeState(const google::protobuf::Any& node_options, Status& status) {
    mediapipe::PythonBackendCalculatorOptions options;
    node_options.UnpackTo(&options);
    if (!std::filesystem::exists(options.handler_path())) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Python node file: {} does not exist. ", options.handler_path());
        status = StatusCode::PYTHON_NODE_FILE_DOES_NOT_EXIST;
        return;
    }
    auto fs_handler_path = std::filesystem::path(options.handler_path());
    fs_handler_path.replace_extension();

    std::string parent_path = fs_handler_path.parent_path();
    std::string filename = fs_handler_path.filename();

    try {
        py::gil_scoped_acquire acquire;
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")(parent_path.c_str());
        py::module_ script = py::module_::import(filename.c_str());
        py::object OvmsPythonModel = script.attr("OvmsPythonModel");
        py::object pythonModel = OvmsPythonModel();
        py::object kwargsParam = pybind11::dict();
        // TODO: check bool if true
        pythonModel.attr("initialize")(kwargsParam);
        status = StatusCode::OK;
        this->pythonNodeState = pythonModel;
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to process python node file {} : {}", options.handler_path(), e.what());
        status = StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
    } catch (...) {
        SPDLOG_ERROR("Failed to process python node file {}", options.handler_path());
        status = StatusCode::PYTHON_NODE_FILE_STATE_INITIALIZATION_FAILED;
    }
}
NodeState::~NodeState() {
    py::gil_scoped_acquire acquire;
    ~pythonNodeState;
}
#endif

}  // namespace ovms

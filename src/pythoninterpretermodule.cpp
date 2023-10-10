//***************************************************************************
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
#include "pythoninterpretermodule.hpp"

#include <string>
#include <utility>

#include <pybind11/embed.h>  // everything needed for embedding

#include "config.hpp"
#include "logging.hpp"
#include "server.hpp"
#include "status.hpp"

namespace py = pybind11;

namespace ovms {

PythonInterpreterModule::PythonInterpreterModule() {}

Status PythonInterpreterModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", PYTHON_INTERPRETER_MODULE);
    // Initialize Python interpreter
    py::initialize_interpreter();
    py::exec(R"(
        import sys
        print("Python version")
        print (sys.version)
    )");
    py::gil_scoped_release release;  // GIL only needed in Python custom node
    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", PYTHON_INTERPRETER_MODULE);
    return StatusCode::OK;
}

void PythonInterpreterModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", PYTHON_INTERPRETER_MODULE);
    py::finalize_interpreter();
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", PYTHON_INTERPRETER_MODULE);
}

PythonInterpreterModule::~PythonInterpreterModule() {
    this->shutdown();
}

}  // namespace ovms

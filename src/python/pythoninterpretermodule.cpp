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

#include "../config.hpp"
#include "../logging.hpp"
#include "../module.hpp"
#include "../module_names.hpp"
#include "../status.hpp"
#include "python_backend.hpp"

namespace py = pybind11;

namespace ovms {
Status PythonInterpreterModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", PYTHON_INTERPRETER_MODULE_NAME);
    this->threadId = std::this_thread::get_id();
    py::initialize_interpreter();
    py::exec(R"(
        import sys
        print("Python version")
        print(sys.version)
        print(sys.executable)
    )");
    if (!PythonBackend::createPythonBackend(&pythonBackend))
        return StatusCode::INTERNAL_ERROR;
    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", PYTHON_INTERPRETER_MODULE_NAME);
    return StatusCode::OK;
}

void PythonInterpreterModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    else if (state == ModuleState::NOT_INITIALIZED)
        throw std::runtime_error("PythonInterpreterModule has not been initialized. Could not shut down.");

    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", PYTHON_INTERPRETER_MODULE_NAME);
    reacquireGILForThisThread();
    if (pythonBackend != nullptr)
        delete pythonBackend;
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", PYTHON_INTERPRETER_MODULE_NAME);
    py::finalize_interpreter();
}

void PythonInterpreterModule::releaseGILFromThisThread() const {
    if (std::this_thread::get_id() != this->threadId) {
        SPDLOG_ERROR("Cannot use {} from different thread than the one starting module", __FUNCTION__);
        throw std::logic_error("Cannot use method from different thread than the one starting python module");
    }
    this->GILScopedRelease = std::make_unique<py::gil_scoped_release>();
}

void PythonInterpreterModule::reacquireGILForThisThread() const {
    if (std::this_thread::get_id() != this->threadId) {
        SPDLOG_ERROR("Cannot use {} from different thread than the one starting module", __FUNCTION__);
        throw std::logic_error("Cannot use method from different thread than the one starting python module");
    }
    this->GILScopedRelease.reset();
}

PythonBackend* PythonInterpreterModule::getPythonBackend() const {
    return pythonBackend;
}

PythonInterpreterModule::PythonInterpreterModule() = default;

PythonInterpreterModule::~PythonInterpreterModule() {
    this->shutdown();
}

}  // namespace ovms

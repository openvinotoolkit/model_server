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
#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>  // everything needed for embedding
#pragma warning(pop)

#include "../config.hpp"
#include "../logging.hpp"
#include "../module.hpp"
#include "../module_names.hpp"
#include "../status.hpp"
#include "python_backend.hpp"

namespace py = pybind11;

namespace ovms {

Status PythonInterpreterModule::start(const ovms::Config&) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", PYTHON_INTERPRETER_MODULE_NAME);
    this->threadId = std::this_thread::get_id();

    if (!Py_IsInitialized()) {
        SPDLOG_INFO("Initializing python interpreter", PYTHON_INTERPRETER_MODULE_NAME);
        try {
            py::initialize_interpreter();
            ownsInterpreter = true;
        } catch (const pybind11::error_already_set& e) {
            SPDLOG_ERROR("Failed to initialize Python interpreter: {}. "
                         "Ensure libpython.so and dependencies are installed on the system. "
                         "For Ubuntu: apt install libpython3.12-dev; For RHEL: yum install python312-devel. "
                         "Check that PYTHONPATH is set correctly if using custom Python installation.",
                e.what());
            return StatusCode::PYTHON_INTERPRETER_INITIALIZATION_FAILED;
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Failed to initialize Python interpreter (unexpected error): {}. "
                         "The Python runtime may not be properly configured. "
                         "Verify libpython.so is in library search path: export LD_LIBRARY_PATH=/path/to/python/lib:$LD_LIBRARY_PATH",
                e.what());
            return StatusCode::PYTHON_INTERPRETER_INITIALIZATION_FAILED;
        }
    } else {
        SPDLOG_INFO("Python interpreter already initialized", PYTHON_INTERPRETER_MODULE_NAME);
        ownsInterpreter = false;
    }

    try {
        py::gil_scoped_acquire acquire;
        py::exec(R"(
            import sys
            print("Python version:")
            print(sys.version)
            print("Python sys.path output:")
            print(sys.path)
        )");
    } catch (const pybind11::error_already_set& e) {
        SPDLOG_ERROR("Python initialization check failed: {}. "
                     "Python interpreter may be corrupted or missing required modules.",
            e.what());
        return StatusCode::PYTHON_INTERPRETER_INITIALIZATION_FAILED;
    }

    if (!PythonBackend::createPythonBackend(pythonBackend)) {
        SPDLOG_ERROR("Failed to create Python backend. "
                     "Ensure pyovms module is available in PYTHONPATH and libovmspython.so is installed. "
                     "Set PYTHONPATH=/opt/intel/openvino/python:/ovms/lib/python or appropriate paths.");
        return StatusCode::PYTHON_BACKEND_CREATION_FAILED;
    }
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
    pythonBackend.reset();
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", PYTHON_INTERPRETER_MODULE_NAME);
    if (ownsInterpreter)
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

bool PythonInterpreterModule::ownsPythonInterpreter() const {
    return ownsInterpreter;
}

PythonBackend* PythonInterpreterModule::getPythonBackend() const {
    return pythonBackend.get();
}

PythonInterpreterModule::PythonInterpreterModule() {
    ownsInterpreter = false;
}

PythonInterpreterModule::~PythonInterpreterModule() {
    this->shutdown();
}

}  // namespace ovms

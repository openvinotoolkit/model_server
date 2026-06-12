//*****************************************************************************
// Copyright 2026 Intel Corporation
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

#include "../module.hpp"
#include "pythoninterpretermodule.hpp"

#include <string>

#pragma warning(push)
#pragma warning(disable : 6326 28182 6011 28020)
#include <pybind11/embed.h>
#pragma warning(pop)

namespace py = pybind11;

#if defined(_WIN32)
#define PYTHON_RUNTIME_EXPORT __declspec(dllexport)
#else
#define PYTHON_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif

extern "C" PYTHON_RUNTIME_EXPORT ovms::Module* OVMS_createPythonInterpreterModule() {
    return new ovms::PythonInterpreterModule();
}

extern "C" PYTHON_RUNTIME_EXPORT bool OVMS_validatePythonEnvironment(const char** errorMessage) {
    static thread_local std::string lastError;
    if (errorMessage != nullptr) {
        *errorMessage = nullptr;
    }

    bool ownsInterpreter = false;
    try {
        if (!Py_IsInitialized()) {
            py::initialize_interpreter();
            ownsInterpreter = true;
        }
        {
            py::gil_scoped_acquire acquire;
            // Validate that OVMS Python bindings are importable and executable.
            py::module_::import("pyovms");
        }
        if (ownsInterpreter) {
            py::finalize_interpreter();
        }
        return true;
    } catch (const py::error_already_set& e) {
        lastError = e.what();
    } catch (const std::exception& e) {
        lastError = e.what();
    } catch (...) {
        lastError = "Unknown python runtime validation error";
    }

    if (ownsInterpreter && Py_IsInitialized()) {
        py::finalize_interpreter();
    }
    if (errorMessage != nullptr) {
        *errorMessage = lastError.c_str();
    }
    return false;
}

#undef PYTHON_RUNTIME_EXPORT

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
#include "python_environment.hpp"

#include <memory>
#include <stdexcept>

#include "../config.hpp"
#include "../status.hpp"

namespace {
PythonEnvironment* g_pythonEnvironment = nullptr;
}

void PythonEnvironment::SetUp() {
#if (PYTHON_DISABLE == 0)
    pythonModule = std::make_unique<ovms::PythonInterpreterModule>();
    auto status = pythonModule->start(ovms::Config::instance());
    if (!status.ok()) {
        throw std::runtime_error("Global python interpreter module failed to start");
    }
    if (pythonModule->ownsPythonInterpreter()) {
        pythonModule->releaseGILFromThisThread();
    }
    g_pythonEnvironment = this;
#endif
}

void PythonEnvironment::TearDown() {
#if (PYTHON_DISABLE == 0)
    g_pythonEnvironment = nullptr;
    if (pythonModule != nullptr) {
        if (pythonModule->ownsPythonInterpreter()) {
            pythonModule->reacquireGILForThisThread();
        }
        pythonModule->shutdown();
        pythonModule.reset();
    }
#endif
}

ovms::PythonBackend* PythonEnvironment::getPythonBackend() const {
#if (PYTHON_DISABLE == 0)
    if (pythonModule == nullptr) {
        return nullptr;
    }
    return pythonModule->getPythonBackend();
#else
    return nullptr;
#endif
}

ovms::PythonInterpreterModule* PythonEnvironment::getPythonInterpreterModule() const {
#if (PYTHON_DISABLE == 0)
    return pythonModule.get();
#else
    return nullptr;
#endif
}

ovms::PythonBackend* getGlobalPythonBackend() {
    auto* pythonInterpreterModule = getGlobalPythonInterpreterModule();
    if (pythonInterpreterModule == nullptr) {
        return nullptr;
    }
    return pythonInterpreterModule->getPythonBackend();
}

ovms::PythonInterpreterModule* getGlobalPythonInterpreterModule() {
#if (PYTHON_DISABLE == 0)
    if (g_pythonEnvironment == nullptr) {
        return nullptr;
    }
    return g_pythonEnvironment->getPythonInterpreterModule();
#else
    return nullptr;
#endif
}

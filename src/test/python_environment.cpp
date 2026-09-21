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

#include "src/config.hpp"
#include "src/status.hpp"

namespace {
PythonEnvironment* pythonEnvironment = nullptr;
}  // namespace

void PythonEnvironment::SetUp() {
    pythonModule = std::make_unique<ovms::PythonInterpreterModule>();
    auto status = pythonModule->start(ovms::Config::instance());
    if (!status.ok()) {
        throw std::runtime_error("Global python interpreter module failed to start");
    }
    pythonEnvironment = this;
}

void PythonEnvironment::TearDown() {
    pythonEnvironment = nullptr;
    if (pythonModule != nullptr) {
        if (pythonModule->ownsPythonInterpreter()) {
            pythonModule->reacquireGILForThisThread();
        }
        pythonModule->shutdown();
        pythonModule.reset();
    }
}

ovms::PythonBackend* PythonEnvironment::getPythonBackend() const {
    if (pythonModule == nullptr) {
        return nullptr;
    }
    return pythonModule->getPythonBackend();
}

ovms::PythonInterpreterModule* PythonEnvironment::getPythonInterpreterModule() const {
    return pythonModule.get();
}

ovms::PythonBackend* getGlobalPythonBackend() {
    auto* pythonInterpreterModule = getGlobalPythonInterpreterModule();
    if (pythonInterpreterModule == nullptr) {
        return nullptr;
    }
    return pythonInterpreterModule->getPythonBackend();
}

ovms::PythonInterpreterModule* getGlobalPythonInterpreterModule() {
    if (pythonEnvironment == nullptr) {
        return nullptr;
    }
    return pythonEnvironment->getPythonInterpreterModule();
}

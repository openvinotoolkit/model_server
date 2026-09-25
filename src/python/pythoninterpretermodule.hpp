//****************************************************************************
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
#pragma once
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "../module.hpp"
#include "python_runtime_module_api.hpp"

namespace pybind11 {
class gil_scoped_release;
}
namespace py = pybind11;

namespace ovms {
class Config;
class PythonBackend;

class PythonInterpreterModule : public Module, public PythonRuntimeModuleApi {
#if defined(__cpp_lib_jthread) && __cpp_lib_jthread >= 201911L
    using LifecycleThread = std::jthread;
#else
    using LifecycleThread = std::thread;
#endif

    std::unique_ptr<PythonBackend> pythonBackend;
    mutable std::unique_ptr<py::gil_scoped_release> GILScopedRelease;
#if defined(__cpp_lib_jthread) && __cpp_lib_jthread >= 201911L
    std::condition_variable_any shutdownCondition;
#else
    std::condition_variable shutdownCondition;
#endif
    std::mutex lifecycleMtx;
    std::mutex lifecycleControlMtx;
    LifecycleThread lifecycleThread;
    std::thread::id threadId;
#if !defined(__cpp_lib_jthread) || __cpp_lib_jthread < 201911L
    bool shutdownRequested = false;
#endif
    bool startCalled = false;
    bool ownsInterpreter;

public:
    PythonInterpreterModule();
    ~PythonInterpreterModule();
    Status start(const ovms::Config& config) override;
    void shutdown() override;
    PythonBackend* getPythonBackend() const override;
    void releaseGILFromThisThread() const override;
    void reacquireGILForThisThread() const;
    bool ownsPythonInterpreter() const override;

private:
    Status initialize();
    void shutdownOnLifecycleThread() noexcept;
    // Load MediaPipe Python calculators plugin after interpreter is operational
    void loadPythonCalculatorsPlugin();
};
}  // namespace ovms

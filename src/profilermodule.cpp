//****************************************************************************
// Copyright 2022 Intel Corporation
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
#include "profilermodule.hpp"

#include <memory>
#include <utility>

#include "config.hpp"
#include "logging.hpp"
#include "module.hpp"
#include "profiler.hpp"
#include "server.hpp"
#include "status.hpp"

namespace ovms {
#ifdef MTR_ENABLED
ProfilerModule::ProfilerModule() = default;
Status ProfilerModule::start(const Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", PROFILER_MODULE_NAME);
    this->profiler = std::make_unique<Profiler>(config.tracePath());
    if (!this->profiler) {
        return StatusCode::INTERNAL_ERROR;
    }
    if (!this->profiler->isInitialized()) {
        SPDLOG_ERROR("Cannot open file for profiler, --trace_path: {}", config.tracePath());
        return StatusCode::INTERNAL_ERROR;
    }
    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", PROFILER_MODULE_NAME);
    return StatusCode::OK;
}

void ProfilerModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", PROFILER_MODULE_NAME);
    profiler.reset();
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", PROFILER_MODULE_NAME);
}

ProfilerModule::~ProfilerModule() {
    this->shutdown();
}
#endif
}  // namespace ovms

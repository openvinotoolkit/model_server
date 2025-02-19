//****************************************************************************
// Copyright 2025 Intel Corporation
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
#include "capimodule.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../config.hpp"
#include "../logging.hpp"
#include "../server.hpp"
#include "../status.hpp"

namespace ovms {
CAPIModule::~CAPIModule() {
    this->shutdown();
}

CAPIModule::CAPIModule(Server& server) :
    server(server) {}
Status CAPIModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", CAPI_MODULE_NAME);
    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", CAPI_MODULE_NAME);
    return StatusCode::OK;
}

void CAPIModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", CAPI_MODULE_NAME);
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", CAPI_MODULE_NAME);
}
}  // namespace ovms

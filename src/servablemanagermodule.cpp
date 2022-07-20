//***************************************************************************
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
#include "servablemanagermodule.hpp"

#include <string>
#include <utility>

#include "config.hpp"
#include "logging.hpp"
#include "modelmanager.hpp"

namespace ovms {

ServableManagerModule::ServableManagerModule() :
    servableManager(std::make_unique<ModelManager>()) {}

int ServableManagerModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", SERVABLE_MANAGER_MODULE_NAME);
    auto status = servableManager->start(config);
    if (status.ok()) {
        state = ModuleState::INITIALIZED;
        SPDLOG_INFO("{} started", SERVABLE_MANAGER_MODULE_NAME);
        // FIXME this should be reenabled in grpcserver module when functional tests are switched to wait
        // for servablemanager module start log
        // #KFS_CLEANUP
        SPDLOG_INFO("Server started on port {}", config.port());
        return EXIT_SUCCESS;
    }
    SPDLOG_ERROR("ovms::ModelManager::Start() Error: {}", status.string());
    return EXIT_FAILURE;
}
void ServableManagerModule::shutdown() {
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", SERVABLE_MANAGER_MODULE_NAME);
    servableManager->join();
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", SERVABLE_MANAGER_MODULE_NAME);
}
ModelManager& ServableManagerModule::getServableManager() const {
    return *servableManager;
}
}  // namespace ovms

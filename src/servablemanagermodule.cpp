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
#include "metric_module.hpp"
#include "modelmanager.hpp"
#include "server.hpp"

namespace ovms {

ServableManagerModule::ServableManagerModule(ovms::Server& ovmsServer) {
    this->servableManager = std::make_unique<ModelManager>("", &dynamic_cast<const MetricModule*>(ovmsServer.getModule(METRICS_MODULE_NAME))->getRegistry());
    if (nullptr == ovmsServer.getModule(METRICS_MODULE_NAME)) {
        const char* message = "Tried to create servable manager module without metrics module";
        SPDLOG_ERROR(message);
        throw std::logic_error(message);
    }
}

Status ServableManagerModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", SERVABLE_MANAGER_MODULE_NAME);
    auto status = getServableManager().start(config);
    if (status.ok()) {
        state = ModuleState::INITIALIZED;
        SPDLOG_INFO("{} started", SERVABLE_MANAGER_MODULE_NAME);
        return status;
    }
    SPDLOG_ERROR("ovms::ModelManager::Start() Error: {}", status.string());
    return status;
}
void ServableManagerModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", SERVABLE_MANAGER_MODULE_NAME);
    getServableManager().join();
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", SERVABLE_MANAGER_MODULE_NAME);
}

ServableManagerModule::~ServableManagerModule() {
    this->shutdown();
}

ModelManager& ServableManagerModule::getServableManager() const {
    return *servableManager;
}
}  // namespace ovms

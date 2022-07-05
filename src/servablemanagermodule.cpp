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
#include "servablemanagermodule.hpp"

#include <string>
#include <utility>

#include "logging.hpp"
#include "modelmanager.hpp"

namespace ovms {

ServableManagerModule::ServableManagerModule() :
    servableManager(std::make_unique<ModelManager>()) {}

int ServableManagerModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    auto status = servableManager->start(config);
    if (status.ok()) {
        state = ModuleState::INITIALIZED;
        return EXIT_SUCCESS;
    }
    SPDLOG_ERROR("ovms::ModelManager::Start() Error: {}", status.string());
    return EXIT_FAILURE;
}
void ServableManagerModule::shutdown() {
    state = ModuleState::STARTED_SHUTDOWN;
    servableManager->join();
    state = ModuleState::SHUTDOWN;
}
const ModelManager& ServableManagerModule::getServableManager() const {
    return *servableManager;
}
}  // namespace ovms

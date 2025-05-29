//***************************************************************************
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
#include "servablesconfigmanagermodule.hpp"

#include <string>
#include <sstream>

#include "../config_export_module/config_export.hpp"
#include "../config.hpp"
#include "../logging.hpp"
#include "../module_names.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "listmodels.hpp"

namespace ovms {
ServablesConfigManagerModule::ServablesConfigManagerModule() {}

Status ServablesConfigManagerModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_TRACE("{} starting", SERVABLES_CONFIG_MANAGER_MODULE_NAME);
    state = ModuleState::INITIALIZED;
    SPDLOG_TRACE("{} started", SERVABLES_CONFIG_MANAGER_MODULE_NAME);
    if (config.getServerSettings().serverMode == LIST_MODELS_MODE) {
        const auto& repositoryPath = config.getServerSettings().hfSettings.downloadPath;
        auto map = listServables(repositoryPath);
        std::stringstream ss;
        for (const auto& [k, v] : map) {
            ss << k << std::endl;
        }
        std::cout << "Available servables to serve from path: " << repositoryPath << " are: " << std::endl << ss.str();
    } else {
        auto status = updateConfig(config.getModelSettings(), config.getServerSettings().exportConfigType);
        if (status.ok()) {
            std::cout << "Config updated: " << config.getModelSettings().configPath << std::endl;
        } else {
            std::cout << "Error on config update : " << status.string() << std::endl;
        }
        return status;
    }
    return StatusCode::OK;
}

void ServablesConfigManagerModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_TRACE("{} shutting down", SERVABLES_CONFIG_MANAGER_MODULE_NAME);
    state = ModuleState::SHUTDOWN;
    SPDLOG_TRACE("{} shutdown", SERVABLES_CONFIG_MANAGER_MODULE_NAME);
}

ServablesConfigManagerModule::~ServablesConfigManagerModule() {
    this->shutdown();
}

}  // namespace ovms

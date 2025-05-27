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

#include "../config.hpp"
#include "../logging.hpp"
#include "../module_names.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "listmodels.hpp"

namespace ovms {
ServablesConfigManagerModule::ServablesConfigManagerModule() {}

#define RETURN_IF_ERROR(StatusOr)                       \
    do {                                                \
        if (std::holds_alternative<Status>(StatusOr)) { \
            return std::get<Status>(StatusOr);          \
        }                                               \
    } while (0)

Status ServablesConfigManagerModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", SERVABLES_CONFIG_MANAGER_MODULE_NAME);
    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", SERVABLES_CONFIG_MANAGER_MODULE_NAME);
    const auto& repositoryPath = config.getServerSettings().hfSettings.downloadPath;
    auto map = listServables(repositoryPath);
    std::stringstream ss;
    for (const auto& [k, v] : map) {
        ss << k << std::endl;
    }
    SPDLOG_INFO("Available servables to serve from path: {} are:\n{}", repositoryPath, ss.str());
    return StatusCode::OK;
}

void ServablesConfigManagerModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", SERVABLES_CONFIG_MANAGER_MODULE_NAME);
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", SERVABLES_CONFIG_MANAGER_MODULE_NAME);
}

ServablesConfigManagerModule::~ServablesConfigManagerModule() {
    this->shutdown();
}

}  // namespace ovms

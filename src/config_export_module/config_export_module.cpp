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
#include "config_export_module.hpp"

#include <string>

#include "../capi_frontend/server_settings.hpp"
#include "../config.hpp"
#include "../config_export_module/config_export.hpp"
#include "../logging.hpp"
#include "../module_names.hpp"
#include "../status.hpp"

namespace ovms {
ConfigExportModule::ConfigExportModule() {}

Status ConfigExportModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", CONFIG_EXPORT_MODULE_NAME);
    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", CONFIG_EXPORT_MODULE_NAME);
    return updateConfig(config.getModelSettings(), config.getServerSettings().exportConfigType);
}

void ConfigExportModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", CONFIG_EXPORT_MODULE_NAME);
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", CONFIG_EXPORT_MODULE_NAME);
}

ConfigExportModule::~ConfigExportModule() {
    this->shutdown();
}

}  // namespace ovms

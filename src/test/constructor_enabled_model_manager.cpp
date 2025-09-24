//*****************************************************************************
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
#include "constructor_enabled_model_manager.hpp"

#include <spdlog/spdlog.h>

#include "../status.hpp"

ConstructorEnabledModelManager::ConstructorEnabledModelManager(const std::string& modelCacheDirectory, ovms::PythonBackend* pythonBackend) :
    ovms::ModelManager(modelCacheDirectory, &registry, pythonBackend) {}
ConstructorEnabledModelManager::~ConstructorEnabledModelManager() {
    join();
    spdlog::info("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
    models.clear();
    spdlog::info("Destructor of modelmanager(Enabled one). Models #:{}", models.size());
}
ovms::Status ConstructorEnabledModelManager::loadConfig(const std::string& jsonFilename) {
    this->configFilename = jsonFilename;
    return ModelManager::loadConfig();
}
void ConstructorEnabledModelManager::updateConfigurationWithoutConfigFile() {
    ModelManager::updateConfigurationWithoutConfigFile();
}
void ConstructorEnabledModelManager::setWaitForModelLoadedTimeoutMs(int value) {
    this->waitForModelLoadedTimeoutMs = value;
}
int ResourcesAccessModelManager::getResourcesSize() {
    return resources.size();
}
void ResourcesAccessModelManager::setResourcesCleanupIntervalMillisec(uint32_t value) {
    this->resourcesCleanupIntervalMillisec = value;
}

#pragma once
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

#include <string>

#include "src/metrics/metric_registry.hpp"
#include "src/model_management/modelmanager.hpp"
#include "src/model_management/servable_loading_queue.hpp"

class ConstructorEnabledModelManager : public ovms::ModelManager {
    ovms::MetricRegistry registry;

public:
    ConstructorEnabledModelManager(const std::string& modelCacheDirectory = "", ovms::PythonBackend* pythonBackend = nullptr);
    ConstructorEnabledModelManager(uint64_t idleTimeoutMicroseconds);
    ~ConstructorEnabledModelManager();

    ovms::Status loadConfig(const std::string& jsonFilename);
    void updateConfigurationWithoutConfigFile();
    void setWaitForModelLoadedTimeoutMs(int value);
    ovms::ServableLoadingQueue& getLoadingQueue() { return *loadingQueue; }
};
class ResourcesAccessModelManager : public ConstructorEnabledModelManager {
public:
    int getResourcesSize();
    void setResourcesCleanupIntervalMillisec(uint32_t value);
};

//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "test_model_manager_utils.hpp"

#include <chrono>
#include <thread>

void waitForOVMSConfigReload(ovms::ModelManager& manager) {
    const float WAIT_MULTIPLIER_FACTOR = 5;
    const uint32_t waitTime = WAIT_MULTIPLIER_FACTOR * manager.getWatcherIntervalMillisec() * 1000;
    bool reloadIsNeeded = true;
    int timestepMs = 10;

    auto start = std::chrono::high_resolution_clock::now();
    while (reloadIsNeeded &&
           (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() < waitTime)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(timestepMs));
        manager.configFileReloadNeeded(reloadIsNeeded);
    }
}

//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "test_utils.hpp"
void waitForOVMSConfigReload(ovms::ModelManager& manager) {
    // This is effectively multiplying by 1.2 to have 1 config reload in between
    // two test steps
    const float WAIT_MULTIPLIER_FACTOR = 1.2;
    const uint waitTime = WAIT_MULTIPLIER_FACTOR * manager.getWatcherIntervalSec() * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
}

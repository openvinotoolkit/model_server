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
#include "hf_pull_model_module.hpp"

#include <string>
#include <utility>

#include "config.hpp"
#include "logging.hpp"
#include "server.hpp"
#include "status.hpp"
#include "libgt2/libgt2.hpp"

namespace ovms {

HfPullModelModule::HfPullModelModule() {}

Status HfPullModelModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", HF_MODEL_PULL_MODULE_NAME);

    this->hfDownloader = std::make_unique<HfDownloader>(config.getHfSettings().sourceModel, config.getHfSettings().repoPath, config.getHfSettings().pullHfModelMode);

    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", HF_MODEL_PULL_MODULE_NAME);

    return StatusCode::OK;
}
void HfPullModelModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_INFO("{} shutting down", HF_MODEL_PULL_MODULE_NAME);
    state = ModuleState::SHUTDOWN;
    SPDLOG_INFO("{} shutdown", HF_MODEL_PULL_MODULE_NAME);
}

HfPullModelModule::~HfPullModelModule() {
    this->shutdown();
}

HfDownloader& HfPullModelModule::getHfDownloader() const {
    return *hfDownloader;
}

}  // namespace ovms

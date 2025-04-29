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
#include "hf_pull_model_module.hpp"

#include <string>
#include <utility>

#include "../config.hpp"
#include "libgt2.hpp"
#include "../graph_export/graph_export.hpp"
#include "../logging.hpp"
#include "../module_names.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"

namespace ovms {

HfPullModelModule::HfPullModelModule() {}

Status HfPullModelModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_INFO("{} starting", HF_MODEL_PULL_MODULE_NAME);

    std::unique_ptr<Libgt2InitGuard> initGuard = std::make_unique<Libgt2InitGuard>();
    if (initGuard->status < 0) {
        SPDLOG_ERROR("Failed to init libgit2 {}", initGuard->errMsg.c_str());
        return StatusCode::HF_FAILED_TO_INIT_LIBGIT2;
    }

    this->hfSettings = config.getServerSettings().hfSettings;

    state = ModuleState::INITIALIZED;
    SPDLOG_INFO("{} started", HF_MODEL_PULL_MODULE_NAME);

    return StatusCode::OK;
}

Status HfPullModelModule::clone() const {
    std::unique_ptr<Libgt2InitGuard> initGuard = std::make_unique<Libgt2InitGuard>();
    if (initGuard->status < 0) {
        SPDLOG_ERROR("Failed to init libgit2 {}", initGuard->errMsg.c_str());
        return StatusCode::HF_FAILED_TO_INIT_LIBGIT2;
    }
    std::unique_ptr<HfDownloader> hfDownloader = std::make_unique<HfDownloader>(this->hfSettings.sourceModel, this->hfSettings.downloadPath, this->GetHfEndpoint(), this->GetHfToken(), this->GetProxy());
    // TODO: CVS-166568 Do we want to set timeout for this operation ?
    auto status = hfDownloader->cloneRepository();
    if (!status.ok()) {
        return status;
    }

    std::unique_ptr<GraphExport> graphExporter = std::make_unique<GraphExport>(this->hfSettings.graphSettings);
    status = graphExporter->createGraphFile(this->hfSettings.downloadPath);
    if (!status.ok()) {
        return status;
    }

    return StatusCode::OK;
}

const std::string HfPullModelModule::GetProxy() const {
    std::string proxy = "";
    const char* envCred = std::getenv("https_proxy");
    if (envCred)
        proxy = std::string(envCred);
    return proxy;
}

const std::string HfPullModelModule::GetHfToken() const {
    std::string token = "";
    const char* envCred = std::getenv("HF_TOKEN");
    if (envCred)
        token = std::string(envCred);
    return token;
}

const std::string HfPullModelModule::GetHfEndpoint() const {
    const char* envCred = std::getenv("HF_ENDPOINT");
    std::string hfEndpoint = "huggingface.co";
    if (envCred) {
        hfEndpoint = std::string(envCred);
    } else {
        SPDLOG_DEBUG("HF_ENDPOINT environment variable not set");
    }

    if (!endsWith(hfEndpoint, "/")) {
        hfEndpoint.append("/");
    }

    return hfEndpoint;
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

}  // namespace ovms

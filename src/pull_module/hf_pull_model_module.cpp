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

#include <iostream>
#include <string>
#include <utility>
#include <variant>

#include "../config.hpp"
#include "libgit2.hpp"
#include "optimum_export.hpp"
#include "gguf_export.hpp"
#include "../graph_export/graph_export.hpp"
#include "../logging.hpp"
#include "../module_names.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"

namespace ovms {
const std::string DEFAULT_EMPTY_ENV_VALUE{""};

const std::string HfPullModelModule::GIT_SERVER_CONNECT_TIMEOUT_ENV{"GIT_OPT_SET_SERVER_CONNECT_TIMEOUT"};
const std::string HfPullModelModule::GIT_SERVER_TIMEOUT_ENV{"GIT_OPT_SET_SERVER_TIMEOUT"};
const std::string HfPullModelModule::GIT_SSL_CERT_LOCATIONS_ENV{"GIT_OPT_SET_SSL_CERT_LOCATIONS"};
// GIT_OPT_SET_SERVER_TIMEOUT

static std::string getEnvReturnOrDefaultIfNotSet(const std::string& envName, const std::string& defaultValue = DEFAULT_EMPTY_ENV_VALUE) {
    std::string value = defaultValue;
    const char* envValue = std::getenv(envName.c_str());
    if (envValue) {
        value = std::string(envValue);
        SPDLOG_DEBUG("{} environment variable set. Using value: {};", envName, value);
    } else {
        SPDLOG_DEBUG("{} environment variable not set. Using default value: {};", envName, defaultValue);
    }
    return value;
}

HfPullModelModule::HfPullModelModule() {}

#define RETURN_IF_ERROR(StatusOr)                       \
    do {                                                \
        if (std::holds_alternative<Status>(StatusOr)) { \
            return std::get<Status>(StatusOr);          \
        }                                               \
    } while (0)

static std::variant<ovms::Status, Libgit2Options> prepareLibgit2Opts() {
    Libgit2Options opts;
    std::string timeoutString = getEnvReturnOrDefaultIfNotSet(HfPullModelModule::GIT_SERVER_CONNECT_TIMEOUT_ENV, "4000");
    auto timeoutOpt = ovms::stoi32(timeoutString);
    if (!timeoutOpt.has_value()) {
        SPDLOG_ERROR("Set invalid value for libgit2 server connection timeout:{}", timeoutString);
        return StatusCode::HF_FAILED_TO_INIT_LIBGIT2;
    }
    bool isHttpsProxyUsed = !getEnvReturnOrDefaultIfNotSet("https_proxy").empty();
    if (isHttpsProxyUsed) {
        if (timeoutOpt.value() != 0)
            SPDLOG_DEBUG("We are not able to set connection timeout when proxy is used");
    } else {
        opts.serverConnectTimeoutMs = timeoutOpt.value();
    }
    timeoutString = getEnvReturnOrDefaultIfNotSet(HfPullModelModule::GIT_SERVER_TIMEOUT_ENV, "4000");
    timeoutOpt = ovms::stoi32(timeoutString);
    if (!timeoutOpt.has_value()) {
        SPDLOG_ERROR("Set invalid value for libgit2 server timeout:{}", timeoutString);
        return StatusCode::HF_FAILED_TO_INIT_LIBGIT2;
    }
    opts.serverTimeoutMs = timeoutOpt.value();

    opts.sslCertificateLocation = getEnvReturnOrDefaultIfNotSet(HfPullModelModule::GIT_SSL_CERT_LOCATIONS_ENV, "");
    return opts;
}

std::variant<ovms::Status, std::unique_ptr<Libgt2InitGuard>> createGuard() {
    auto optsOrError = prepareLibgit2Opts();
    RETURN_IF_ERROR(optsOrError);
    auto initGuard = std::make_unique<Libgt2InitGuard>(std::get<Libgit2Options>(optsOrError));
    if (initGuard->status < 0) {
        SPDLOG_ERROR("Failed to init libgit2: {}", initGuard->errMsg.c_str());
        return StatusCode::HF_FAILED_TO_INIT_LIBGIT2;
    }
    return std::move(initGuard);
}

Status HfPullModelModule::start(const ovms::Config& config) {
    state = ModuleState::STARTED_INITIALIZE;
    SPDLOG_TRACE("{} starting", HF_MODEL_PULL_MODULE_NAME);
    if (config.getServerSettings().hfSettings.downloadType == GIT_CLONE_DOWNLOAD) {
        auto guardOrError = createGuard();
        RETURN_IF_ERROR(guardOrError);
    }
    this->hfSettings = config.getServerSettings().hfSettings;
    state = ModuleState::INITIALIZED;
    SPDLOG_TRACE("{} started", HF_MODEL_PULL_MODULE_NAME);
    return StatusCode::OK;
}

Status HfPullModelModule::clone() const {
    std::string graphDirectory = "";
    if (this->hfSettings.downloadType == GIT_CLONE_DOWNLOAD) {
        auto guardOrError = createGuard();
        if (std::holds_alternative<Status>(guardOrError)) {
            return std::get<Status>(guardOrError);
        }

        HfDownloader hfDownloader(this->hfSettings.sourceModel, this->hfSettings.downloadPath, this->GetHfEndpoint(), this->GetHfToken(), this->GetProxy(), this->hfSettings.overwriteModels);
        auto status = hfDownloader.cloneRepository();
        if (!status.ok()) {
            return status;
        }
        graphDirectory = hfDownloader.getGraphDirectory();
    } else if (this->hfSettings.downloadType == OPTIMUM_CLI_DOWNLOAD) {
        OptimumDownloader optimumDownloader(this->hfSettings);
        auto status = optimumDownloader.cloneRepository();
        if (!status.ok()) {
            return status;
        }
        graphDirectory = optimumDownloader.getGraphDirectory();
    } else if (this->hfSettings.downloadType == GGUF_DOWNLOAD) {
        GGUFDownloader ggufDownloader(this->GetHfEndpoint(), this->hfSettings);
        auto status = ggufDownloader.downloadModel();
        if (!status.ok()) {
            return status;
        }
        graphDirectory = ggufDownloader.getModelPath();
        SPDLOG_ERROR("GGUF graphDirectory: {}", graphDirectory);
        // FIXME need to use gguffilename
    } else {
        SPDLOG_ERROR("Unsupported download type: {}", enumToString(this->hfSettings.downloadType));
        return StatusCode::INTERNAL_ERROR;
    }
    std::cout << "Model: " << this->hfSettings.sourceModel << " downloaded to: " << graphDirectory << std::endl;
    GraphExport graphExporter;
    auto status = graphExporter.createServableConfig(graphDirectory, this->hfSettings);
    if (!status.ok()) {
        return status;
    }
    std::cout << "Graph: graph.pbtxt created in: " << graphDirectory << std::endl;

    return StatusCode::OK;
}

const std::string HfPullModelModule::GetProxy() const {
    return getEnvReturnOrDefaultIfNotSet("https_proxy");
}

const std::string HfPullModelModule::GetHfToken() const {
    return getEnvReturnOrDefaultIfNotSet("HF_TOKEN");
}

const std::string HfPullModelModule::GetHfEndpoint() const {
    std::string hfEndpoint = getEnvReturnOrDefaultIfNotSet("HF_ENDPOINT", "https://huggingface.co");
    if (!endsWith(hfEndpoint, "/")) {
        hfEndpoint.append("/");
    }
    return hfEndpoint;
}

void HfPullModelModule::shutdown() {
    if (state == ModuleState::SHUTDOWN)
        return;
    state = ModuleState::STARTED_SHUTDOWN;
    SPDLOG_TRACE("{} shutting down", HF_MODEL_PULL_MODULE_NAME);
    state = ModuleState::SHUTDOWN;
    SPDLOG_TRACE("{} shutdown", HF_MODEL_PULL_MODULE_NAME);
}

HfPullModelModule::~HfPullModelModule() {
    this->shutdown();
}

}  // namespace ovms

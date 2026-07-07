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
#include <filesystem>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "../config.hpp"
#include "src/filesystem/filesystem.hpp"
#include "libgit2.hpp"
#include "optimum_export.hpp"
#include "curl_downloader.hpp"
#include "gguf_downloader.hpp"
#include "../graph_export/graph_export_paths.hpp"
#include "../logging.hpp"
#include "../mediapipe_runtime_api.hpp"
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

std::variant<ovms::Status, std::unique_ptr<Libgt2InitGuard>> createLibGitGuard() {
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
        auto guardOrError = createLibGitGuard();
        RETURN_IF_ERROR(guardOrError);
    }
    this->hfSettings = config.getServerSettings().hfSettings;
    state = ModuleState::INITIALIZED;
    SPDLOG_TRACE("{} started", HF_MODEL_PULL_MODULE_NAME);
    return StatusCode::OK;
}

Status HfPullModelModule::resolveHfLoraFilenames() {
    if (!std::holds_alternative<ImageGenerationGraphSettingsImpl>(this->hfSettings.graphSettings)) {
        return StatusCode::OK;
    }
    auto& graphSettings = std::get<ImageGenerationGraphSettingsImpl>(this->hfSettings.graphSettings);
    for (auto& adapter : graphSettings.loraAdapters) {
        if (adapter.sourceType != LoraSourceType::HF_REPO) {
            continue;
        }
        if (adapter.safetensorsFile.has_value()) {
            continue;
        }
        if (adapter.resolvedSafetensorsFile.has_value()) {
            continue;
        }
        // Query HF API to find the .safetensors file in the LoRA repo
        std::string apiUrl = this->GetHfEndpoint() + "api/models/" + adapter.sourceLora;
        SPDLOG_DEBUG("Querying HF API for LoRA adapter files: {}", apiUrl);
        std::string responseBody;
        std::string hfToken = this->GetHfToken();
        auto status = fetchUrlToString(apiUrl, hfToken, responseBody);
        if (!status.ok()) {
            SPDLOG_ERROR("Failed to query HF API for LoRA adapter: {}", adapter.sourceLora);
            return status;
        }
        // Parse JSON response to find .safetensors files in siblings array
        // Example: { "siblings": [{"rfilename": "file1.safetensors"}, ...] }
        try {
            auto json = nlohmann::json::parse(responseBody);
            std::vector<std::string> safetensorsFiles;
            if (json.contains("siblings") && json["siblings"].is_array()) {
                for (const auto& sibling : json["siblings"]) {
                    if (sibling.contains("rfilename") && sibling["rfilename"].is_string()) {
                        const std::string& filename = sibling["rfilename"].get_ref<const std::string&>();
                        if (endsWith(filename, ".safetensors")) {
                            safetensorsFiles.push_back(filename);
                        }
                    }
                }
            }
            if (safetensorsFiles.empty()) {
                SPDLOG_ERROR("No .safetensors files found via HF API for LoRA adapter: {}", adapter.sourceLora);
                return StatusCode::PATH_INVALID;
            }
            if (safetensorsFiles.size() > 1) {
                SPDLOG_ERROR("Multiple .safetensors files found for LoRA adapter: {}. Use @filename to specify.", adapter.sourceLora);
                return StatusCode::PATH_INVALID;
            }
            adapter.resolvedSafetensorsFile = safetensorsFiles[0];
            SPDLOG_DEBUG("Resolved LoRA safetensors file for {}: {}", adapter.sourceLora, adapter.resolvedSafetensorsFile.value());
        } catch (const nlohmann::json::exception& e) {
            SPDLOG_ERROR("Failed to parse HF API JSON response for LoRA adapter {}: {}", adapter.sourceLora, e.what());
            return StatusCode::INTERNAL_ERROR;
        }
    }
    return StatusCode::OK;
}

Status HfPullModelModule::pullLoraAdapters(const std::string& graphDirectory) {
    if (!std::holds_alternative<ImageGenerationGraphSettingsImpl>(this->hfSettings.graphSettings)) {
        return StatusCode::OK;
    }
    auto status = this->resolveHfLoraFilenames();
    if (!status.ok()) {
        return status;
    }
    const auto& graphSettings = std::get<ImageGenerationGraphSettingsImpl>(this->hfSettings.graphSettings);
    for (const auto& adapter : graphSettings.loraAdapters) {
        if (adapter.sourceType == LoraSourceType::LOCAL_FILE) {
            std::cout << "LoRA adapter: " << adapter.alias << " using local file: " << adapter.sourceLora << std::endl;
            continue;
        }
        std::string loraDownloadPath;
        std::string loraUrl;
        std::string authTokenHF;
        if (adapter.sourceType == LoraSourceType::HF_REPO) {
            loraDownloadPath = FileSystem::joinPath({graphDirectory, "loras", adapter.sourceLora});
            loraUrl = this->GetHfEndpoint() + adapter.sourceLora + "/resolve/main/" + adapter.effectiveSafetensorsFile().value();
            authTokenHF = this->GetHfToken();
        } else if (adapter.sourceType == LoraSourceType::DIRECT_URL) {
            loraDownloadPath = FileSystem::joinPath({graphDirectory, "loras", adapter.alias});
            loraUrl = adapter.sourceLora;
        } else {
            SPDLOG_ERROR("Unknown LoRA source type for adapter: {}", adapter.alias);
            return StatusCode::INTERNAL_ERROR;
        }
        auto loraFilePath = FileSystem::joinPath({loraDownloadPath, adapter.effectiveSafetensorsFile().value()});
        if (!this->hfSettings.overwriteModels && std::filesystem::exists(loraFilePath)) {
            std::cout << "LoRA adapter: " << adapter.alias << " already exists, skipping download." << std::endl;
            continue;
        }
        if (!std::filesystem::exists(loraDownloadPath)) {
            if (!std::filesystem::create_directories(loraDownloadPath)) {
                SPDLOG_ERROR("Failed to create LoRA directory: {}", loraDownloadPath);
                return StatusCode::DIRECTORY_NOT_CREATED;
            }
        }
        auto loraTmpFilePath = loraFilePath + ".tmp";
        std::error_code ec;
        std::filesystem::remove(loraTmpFilePath, ec);
        status = downloadFileWithCurl(loraUrl, loraTmpFilePath, authTokenHF);
        if (!status.ok()) {
            SPDLOG_ERROR("Failed to download LoRA adapter: {} from: {}", adapter.alias, loraUrl);
            std::filesystem::remove(loraTmpFilePath, ec);
            return status;
        }
        std::filesystem::rename(loraTmpFilePath, loraFilePath, ec);
        if (ec) {
            SPDLOG_ERROR("Failed to rename LoRA temp file: {} -> {}: {}", loraTmpFilePath, loraFilePath, ec.message());
            std::filesystem::remove(loraTmpFilePath, ec);
            return StatusCode::INTERNAL_ERROR;
        }
        std::cout << "LoRA adapter: " << adapter.alias << " downloaded to: " << loraDownloadPath << std::endl;
    }
    return StatusCode::OK;
}

Status HfPullModelModule::clone() {
    std::string graphDirectory = "";
    std::unique_ptr<IModelDownloader> downloader;
    std::variant<ovms::Status, std::unique_ptr<Libgt2InitGuard>> guardOrError;
    if (this->hfSettings.downloadType == GIT_CLONE_DOWNLOAD) {
        guardOrError = createLibGitGuard();
        if (std::holds_alternative<Status>(guardOrError)) {
            return std::get<Status>(guardOrError);
        }

        downloader = std::make_unique<HfDownloader>(this->hfSettings.sourceModel, IModelDownloader::getGraphDirectory(this->hfSettings.downloadPath, this->hfSettings.sourceModel), this->GetHfEndpoint(), this->GetHfToken(), this->GetProxy(), this->hfSettings.overwriteModels);
    } else if (this->hfSettings.downloadType == OPTIMUM_CLI_DOWNLOAD) {
        downloader = std::make_unique<OptimumDownloader>(this->hfSettings.exportSettings, this->hfSettings.task, this->hfSettings.sourceModel, IModelDownloader::getGraphDirectory(this->hfSettings.downloadPath, this->hfSettings.sourceModel), this->hfSettings.overwriteModels);
    } else if (this->hfSettings.downloadType == GGUF_DOWNLOAD) {
        downloader = std::make_unique<GGUFDownloader>(this->hfSettings.sourceModel, IModelDownloader::getGraphDirectory(this->hfSettings.downloadPath, this->hfSettings.sourceModel), this->hfSettings.overwriteModels, this->hfSettings.ggufFilename, this->GetHfEndpoint());
    } else {
        SPDLOG_ERROR("Unsupported download type");
        return StatusCode::INTERNAL_ERROR;
    }

    auto status = downloader->downloadModel();
    if (!status.ok()) {
        return status;
    }
    graphDirectory = downloader->getGraphDirectory();
    std::cout << "Model: " << this->hfSettings.sourceModel << " downloaded to: " << graphDirectory << std::endl;

    // Text gen with draft source model case - downloads second model
    if (std::holds_alternative<TextGenGraphSettingsImpl>(this->hfSettings.graphSettings) && std::get<TextGenGraphSettingsImpl>(this->hfSettings.graphSettings).draftModelDirName.has_value()) {
        auto& graphSettings = std::get<TextGenGraphSettingsImpl>(this->hfSettings.graphSettings);
        std::unique_ptr<IModelDownloader> draftModelDownloader;
        draftModelDownloader = std::make_unique<HfDownloader>(graphSettings.draftModelDirName.value(), getDraftModelDirectoryPath(graphDirectory, graphSettings.draftModelDirName.value()), this->GetHfEndpoint(), this->GetHfToken(), this->GetProxy(), this->hfSettings.overwriteModels);
        status = draftModelDownloader->downloadModel();
        if (!status.ok()) {
            return status;
        }

        std::cout << "Draft model: " << getDraftModelDirectoryName(graphSettings.draftModelDirName.value()) << " downloaded to: " << getDraftModelDirectoryPath(graphDirectory, graphSettings.draftModelDirName.value()) << std::endl;
    }

    // Image gen with LoRA adapters case - resolve filenames and download safetensors files
    status = this->pullLoraAdapters(graphDirectory);
    if (!status.ok()) {
        return status;
    }

    MediapipeRuntimeApi runtimeApi(nullptr);
    status = runtimeApi.createServableConfig(graphDirectory, this->hfSettings, true);  // when downloading from HF we always create config file, but when using local model with --task we create config in memory without writing to file
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

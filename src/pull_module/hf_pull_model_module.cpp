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

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include "../config.hpp"
#include "src/filesystem/filesystem.hpp"
#include "libgit2.hpp"
#include "optimum_export.hpp"
#include "curl_downloader.hpp"
#include "gguf_downloader.hpp"
#include "../graph_export/graph_export.hpp"
#include "../logging.hpp"
#include "../module_names.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../version.hpp"

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
        if (!adapter.safetensorsFile.empty()) {
            continue;
        }
        // Query HF API to find the .safetensors file in the LoRA repo
        std::string apiUrl = this->GetHfEndpoint() + "api/models/" + adapter.sourceLora;
        SPDLOG_DEBUG("Querying HF API for LoRA adapter files: {}", apiUrl);
        std::string agentString = std::string(PROJECT_NAME) + "/" + std::string(PROJECT_VERSION);
        std::string responseBody;
        CURL* curl = curl_easy_init();
        if (!curl) {
            SPDLOG_ERROR("Failed to initialize cURL for HF API query");
            return StatusCode::INTERNAL_ERROR;
        }
        auto handleGuard = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>(curl, curl_easy_cleanup);
        auto writeCallback = +[](void* buffer, size_t size, size_t nmemb, void* userData) -> size_t {
            auto* body = static_cast<std::string*>(userData);
            body->append(static_cast<char*>(buffer), size * nmemb);
            return size * nmemb;
        };
        curl_easy_setopt(curl, CURLOPT_URL, apiUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, agentString.c_str());
        curl_easy_setopt(curl, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        std::string hfToken = this->GetHfToken();
        struct curl_slist* headers = nullptr;
        if (!hfToken.empty()) {
            std::string authHeader = "Authorization: Bearer " + hfToken;
            headers = curl_slist_append(headers, authHeader.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        }
        CURLcode res = curl_easy_perform(curl);
        if (headers) {
            curl_slist_free_all(headers);
        }
        if (res != CURLE_OK) {
            SPDLOG_ERROR("cURL error querying HF API for LoRA {}: {}", adapter.sourceLora, curl_easy_strerror(res));
            return StatusCode::INTERNAL_ERROR;
        }
        int32_t httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
        if (httpCode != 200) {
            SPDLOG_ERROR("HF API returned HTTP {} for LoRA adapter: {}", httpCode, adapter.sourceLora);
            return StatusCode::PATH_INVALID;
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
            adapter.safetensorsFile = safetensorsFiles[0];
            SPDLOG_DEBUG("Resolved LoRA safetensors file for {}: {}", adapter.sourceLora, adapter.safetensorsFile);
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
            loraUrl = this->GetHfEndpoint() + adapter.sourceLora + "/resolve/main/" + adapter.safetensorsFile;
            authTokenHF = this->GetHfToken();
        } else if (adapter.sourceType == LoraSourceType::DIRECT_URL) {
            loraDownloadPath = FileSystem::joinPath({graphDirectory, "loras", adapter.alias});
            loraUrl = adapter.sourceLora;
        } else {
            SPDLOG_ERROR("Unknown LoRA source type for adapter: {}", adapter.alias);
            return StatusCode::INTERNAL_ERROR;
        }
        auto loraFilePath = FileSystem::joinPath({loraDownloadPath, adapter.safetensorsFile});
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
        status = downloadFileWithCurl(loraUrl, loraFilePath, authTokenHF);
        if (!status.ok()) {
            SPDLOG_ERROR("Failed to download LoRA adapter: {} from: {}", adapter.alias, loraUrl);
            return status;
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
        draftModelDownloader = std::make_unique<HfDownloader>(graphSettings.draftModelDirName.value(), GraphExport::getDraftModelDirectoryPath(graphDirectory, graphSettings.draftModelDirName.value()), this->GetHfEndpoint(), this->GetHfToken(), this->GetProxy(), this->hfSettings.overwriteModels);
        status = draftModelDownloader->downloadModel();
        if (!status.ok()) {
            return status;
        }

        std::cout << "Draft model: " << GraphExport::getDraftModelDirectoryName(graphSettings.draftModelDirName.value()) << " downloaded to: " << GraphExport::getDraftModelDirectoryPath(graphDirectory, graphSettings.draftModelDirName.value()) << std::endl;
    }

    // Image gen with LoRA adapters case - resolve filenames and download safetensors files
    status = this->pullLoraAdapters(graphDirectory);
    if (!status.ok()) {
        return status;
    }

    GraphExport graphExporter;
    status = graphExporter.createServableConfig(graphDirectory, this->hfSettings);
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

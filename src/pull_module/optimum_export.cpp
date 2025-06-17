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
#include "optimum_export.hpp"

#include <string>
#include <memory>

#include "../capi_frontend/server_settings.hpp"
#include "../filesystem.hpp"
#include "../localfilesystem.hpp"
#include "../logging.hpp"
#include "../stringutils.hpp"
#include "../status.hpp"
#include "cmd_exec.hpp"

namespace ovms {

std::string getTextGenCmd(const std::string& directoryPath, TextGenGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // NPU specific settings
    if (graphSettings.targetDevice == "NPU" && !graphSettings.extra_quantization_params.has_value()){
        graphSettings.extra_quantization_params.value() = "--sym --ratio 1.0 --group-size -1";
    }
    // clang-format off
    oss << "optimum-cli export openvino --model " << graphSettings.modelName << " --trust-remote-code ";
    oss << " --weight-format " << graphSettings.precision;
    oss << graphSettings.extra_quantization_params.has_value() ? graphSettings.extra_quantization_params.value() : std::string();
    oss << " " << directoryPath;
    // clang-format on

    return oss.str();
}

std::string getTextGenCmd(const std::string& directoryPath, EmbeddingsGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << "optimum-cli export openvino --disable-convert-tokenizer --model " << graphSettings.modelName << " --trust-remote-code ";
    oss << " --weight-format " << graphSettings.precision;
    oss << " --task feature-extraction --library sentence_transformers ";
    oss << " " << directoryPath;
    // clang-format on

    return oss.str();
}

std::string getTextGenCmd(const std::string& directoryPath, RerankGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << "optimum-cli export openvino --disable-convert-tokenizer --model " << graphSettings.modelName << " --trust-remote-code ";
    oss << " --weight-format " << graphSettings.precision;
    oss << " --task text-classification ";
    oss << " " << directoryPath;
    // clang-format on

    return oss.str();
}

std::string getTextGenCmd(const std::string& directoryPath, ImageGenerationGraphSettingsImpl& graphSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << "optimum-cli export openvino --model " << graphSettings.modelName;
    oss << " --weight-format " << graphSettings.precision;
    oss << " " << directoryPath;
    // clang-format on

    return oss.str();
}

OptimumDownloader::OptimumDownloader() {
    this->sourceModel = "";
    this->downloadPath = "";
    this->hfSettings = {};
    this->overwriteModels = false;
}

std::string OptimumDownloader::getGraphDirectory() {
    return this->downloadPath;
}

OptimumDownloader::OptimumDownloader(const std::string& inSourceModel, const std::string& inDownloadPath, const HFSettingsImpl& inHfSettings, bool inOverwrite) {
    this->sourceModel = inSourceModel;
    this->downloadPath = HfDownloader::getGraphDirectory(inDownloadPath, inSourceModel);
    this->hfSettings = inHfSettings;
    this->overwriteModels = inOverwrite;
}

Status OptimumDownloader::checkRequiredToolsArePresent() {
    std::string cmd = "optimum-cli -h";
    std::string output = exec_cmd(cmd);
    if (output.find("usage: optimum-cli") == std::string::npos) {
        SPDLOG_DEBUG(output);
        SPDLOG_ERROR("optimum-cli executable is not present. Please install python and demos/common/export_models/requirements.txt");
        return StatusCode::HF_FAILED_TO_INIT_OPTIMUM_CLI;
    }

    SPDLOG_DEBUG("Optimum-cli executable is present");
    return StatusCode::OK;
}

Status OptimumDownloader::cloneRepository() {
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }

    // Repository exists and we do not want to overwrite
    if (std::filesystem::is_directory(this->downloadPath) && !this->overwriteModels) {
        SPDLOG_DEBUG("Path already exists on local filesystem. Not downloading to path: {}", this->downloadPath);
        return StatusCode::OK;
    }

    // Check optimum cli installed
    auto status = checkRequiredToolsArePresent();
    if (!status.ok()) {
        return status;
    }

    status = checkIfOverwriteAndRemove(this->downloadPath);
    if (!status.ok()) {
        return status;
    }

    std::string cmd = "";
    switch (this->hfSettings.task) {
        case TEXT_GENERATION_GRAPH: {
            if (std::holds_alternative<TextGenGraphSettingsImpl>(this->hfSettings.graphSettings)) {
                cmd = getTextGenCmd(this->downloadPath, std::get<TextGenGraphSettingsImpl>(this->hfSettings.graphSettings));
            } else {
                SPDLOG_ERROR("Text generation task options not initialised.");
                return StatusCode::INTERNAL_ERROR;
            }
            break;
        }
        case EMBEDDINGS_GRAPH: {
            if (std::holds_alternative<EmbeddingsGraphSettingsImpl>(this->hfSettings.graphSettings)) {
                cmd = getTextGenCmd(this->downloadPath, std::get<EmbeddingsGraphSettingsImpl>(this->hfSettings.graphSettings));
            } else {
                SPDLOG_ERROR("Embeddings task options not initialised.");
                return StatusCode::INTERNAL_ERROR;
            }
            break;
        }
        case RERANK_GRAPH: {
            if (std::holds_alternative<RerankGraphSettingsImpl>(this->hfSettings.graphSettings)) {
                cmd = getTextGenCmd(this->downloadPath, std::get<RerankGraphSettingsImpl>(this->hfSettings.graphSettings));
            } else {
                SPDLOG_ERROR("Rerank taskoptions not initialised.");
                return StatusCode::INTERNAL_ERROR;
            }
            break;
        }
        case IMAGE_GENERATION_GRAPH: {
            if (std::holds_alternative<ImageGenerationGraphSettingsImpl>(this->hfSettings.graphSettings)) {
                cmd = getTextGenCmd(this->downloadPath, std::get<ImageGenerationGraphSettingsImpl>(this->hfSettings.graphSettings));
            } else {
                SPDLOG_ERROR("Image generation task options not initialised.");
                return StatusCode::INTERNAL_ERROR;
            }
            break;
        }
        case UNKNOWN_GRAPH: {
            SPDLOG_ERROR("Optimum cli task options not initialised.");
            return StatusCode::INTERNAL_ERROR;
            break;
        }
    }

    SPDLOG_DEBUG("Executing command: {}", cmd);
    std::string output = exec_cmd(cmd);
    if (output.find("usage: optimum-cli") == std::string::npos) {
        SPDLOG_DEBUG(output);
        SPDLOG_ERROR("optimum-cli executable is not present. Please install python and demos/common/export_models/requirements.txt");
        return StatusCode::HF_FAILED_TO_INIT_OPTIMUM_CLI;
    }

    return StatusCode::OK;
}

}  // namespace ovms

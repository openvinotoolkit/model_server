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

std::string OptimumDownloader::getExportCmdText() {
    std::ostringstream oss;
    // NPU specific settings
    if (this->hfSettings.targetDevice == "NPU" && !this->hfSettings.extraQuantizationParams.has_value()) {
        this->hfSettings.extraQuantizationParams.value() = "--sym --ratio 1.0 --group-size -1";
    }
    // clang-format off
    oss << "optimum-cli export openvino --model " << this->hfSettings.sourceModel << " --trust-remote-code ";
    oss << " --weight-format " << this->hfSettings.precision << " ";
    if (this->hfSettings.extraQuantizationParams.has_value()) {
        oss << this->hfSettings.extraQuantizationParams.value() << " ";
    }
    oss << this->downloadPath;
    // clang-format on
    return oss.str();
}

std::string OptimumDownloader::getExportCmdEmbeddings() {
    std::ostringstream oss;
    // clang-format off
    oss << "optimum-cli export openvino --disable-convert-tokenizer --task feature-extraction --library sentence_transformers";
    oss << " --model " << this->hfSettings.sourceModel << " --trust-remote-code ";
    oss << " --weight-format " << this->hfSettings.precision;
    oss << " " << this->downloadPath;
    // clang-format on

    return oss.str();
}

std::string OptimumDownloader::getExportCmdRerank() {
    std::ostringstream oss;
    // clang-format off
    oss << "optimum-cli export openvino --disable-convert-tokenizer --model " << this->hfSettings.sourceModel << " --trust-remote-code ";
    oss << " --weight-format " << this->hfSettings.precision;
    oss << " --task text-classification ";
    oss << " " << this->downloadPath;
    // clang-format on

    return oss.str();
}

std::string OptimumDownloader::getExportCmdImage() {
    std::ostringstream oss;
    // clang-format off
    oss << "optimum-cli export openvino --model " << this->hfSettings.sourceModel;
    oss << " --weight-format " << this->hfSettings.precision;
    oss << " " << this->downloadPath;
    // clang-format on

    return oss.str();
}

std::string OptimumDownloader::getExportCmd() {
    std::string cmd = "";
    switch (this->hfSettings.task) {
        case TEXT_GENERATION_GRAPH: {
            cmd = getExportCmdText();
            break;
        }
        case EMBEDDINGS_GRAPH: {
            cmd = getExportCmdEmbeddings();
            break;
        }
        case RERANK_GRAPH: {
            cmd = getExportCmdRerank();
            break;
        }
        case IMAGE_GENERATION_GRAPH: {
            cmd = getExportCmdImage();
            break;
        }
        case UNKNOWN_GRAPH: {
            SPDLOG_ERROR("Optimum cli task options not initialised.");
            break;
        }
    }

    return cmd;
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

OptimumDownloader::OptimumDownloader(const HFSettingsImpl& inHfSettings) {
    this->sourceModel = inHfSettings.sourceModel;
    this->downloadPath = HfDownloader::getGraphDirectory(inHfSettings.downloadPath, inHfSettings.sourceModel);
    this->hfSettings = inHfSettings;
    this->overwriteModels = inHfSettings.overwriteModels;
}

Status OptimumDownloader::checkRequiredToolsArePresent() {
    std::string cmd = "optimum-cli -h";
    int retCode = 0;
    std::string output = exec_cmd(cmd, retCode);
    if (retCode != 0 || output.find(OPTIMUM_CLI_IS_PRESET_OUTPUT_STRING) == std::string::npos) {
        SPDLOG_DEBUG(output);
        SPDLOG_ERROR("optimum-cli executable is not present. Please install python and demos/common/export_models/requirements.txt");
        return StatusCode::HF_FAILED_TO_INIT_OPTIMUM_CLI;
    }

    SPDLOG_DEBUG("Optimum-cli executable is present");
    return StatusCode::OK;
}

Status OptimumDownloader::cloneRepository() {
    if (this->hfSettings.downloadType != OPTIMUM_CLI_DOWNLOAD) {
        SPDLOG_ERROR("Wrong download type selected. Expected optiumum-cli type.");
        return StatusCode::INTERNAL_ERROR;
    }
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

    std::string cmd = getExportCmd();
    if (cmd == "") {
        return StatusCode::INTERNAL_ERROR;
    }
    
    SPDLOG_DEBUG("Executing command: {}", cmd);
    int retCode = 0;
    std::string output = exec_cmd(cmd, retCode);
    if (retCode != 0 || output.find(OptimumDownloader::EXPORT_SUCCESS_OUTPUT_STRING) == std::string::npos) {
        SPDLOG_DEBUG(output);
        SPDLOG_ERROR("optimum-cli command failed.");
        return StatusCode::HF_RUN_OPTIMUM_CLI_EXPORT_FAILED;
    }

    return StatusCode::OK;
}

}  // namespace ovms

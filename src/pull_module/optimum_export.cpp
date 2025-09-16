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
    oss << this->OPTIMUM_CLI_EXPORT_COMMAND;
    oss << "--model " << this->hfSettings.sourceModel << " --trust-remote-code ";
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
    oss << this->OPTIMUM_CLI_EXPORT_COMMAND;
    oss << "--disable-convert-tokenizer --task feature-extraction --library sentence_transformers";
    oss << " --model " << this->hfSettings.sourceModel << " --trust-remote-code ";
    oss << " --weight-format " << this->hfSettings.precision;
    oss << " " << this->downloadPath;
    // clang-format on

    return oss.str();
}

std::string OptimumDownloader::getExportCmdRerank() {
    std::ostringstream oss;
    // clang-format off
    oss << this->OPTIMUM_CLI_EXPORT_COMMAND;
    oss << "--disable-convert-tokenizer --model " << this->hfSettings.sourceModel;
    oss << " --trust-remote-code ";
    oss << " --weight-format " << this->hfSettings.precision;
    oss << " --task text-classification ";
    oss << " " << this->downloadPath;
    // clang-format on

    return oss.str();
}

std::string OptimumDownloader::getExportCmdImageGeneration() {
    std::ostringstream oss;
    // clang-format off
    oss << this->OPTIMUM_CLI_EXPORT_COMMAND;
    oss << "--model " << this->hfSettings.sourceModel;
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
        cmd = getExportCmdImageGeneration();
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
    this->overwriteModels = false;
}

std::string OptimumDownloader::getGraphDirectory() {
    return this->downloadPath;
}

OptimumDownloader::OptimumDownloader(const HFSettingsImpl& inHFSettings, const std::string& inSourceModel, const std::string& inDownloadPath, bool inOverwrite, const std::string& cliExportCmd, const std::string& cliCheckCmd) {
    this->sourceModel = inSourceModel;
    this->downloadPath = HfDownloader::getGraphDirectory(inDownloadPath, inSourceModel);
    this->overwriteModels = overwriteModels;
    this->hfSettings = inHFSettings;
    this->OPTIMUM_CLI_CHECK_COMMAND = cliCheckCmd;
    this->OPTIMUM_CLI_EXPORT_COMMAND = cliExportCmd;
}

Status OptimumDownloader::checkRequiredToolsArePresent() {
    int retCode = -1;
    std::string output = exec_cmd(this->OPTIMUM_CLI_CHECK_COMMAND, retCode);
    if (retCode != 0) {
        SPDLOG_DEBUG("Command output {}", output);
        SPDLOG_ERROR("Target folder {} not found, trying to pull {} from HuggingFace but missing optimum-intel. Use the ovms package with optimum-intel.", this->hfSettings.downloadPath, this->hfSettings.sourceModel);
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

    std::string cmd = getExportCmd();
    if (cmd == "") {
        return StatusCode::INTERNAL_ERROR;
    }

    SPDLOG_DEBUG("Executing command: {}", cmd);
    int retCode = -1;
    std::string output = exec_cmd(cmd, retCode);
    if (retCode != 0) {
        SPDLOG_DEBUG("Command output {}", output);
        SPDLOG_ERROR("optimum-cli command failed.");
        return StatusCode::HF_RUN_OPTIMUM_CLI_EXPORT_FAILED;
    }

    return StatusCode::OK;
}

}  // namespace ovms

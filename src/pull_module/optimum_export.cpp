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
#include "model_downloader.hpp"

namespace ovms {

std::string OptimumDownloader::getExportCmdText() {
    std::ostringstream oss;
    // NPU specific settings
    if (this->exportSettings.targetDevice == "NPU" && !this->exportSettings.extraQuantizationParams.has_value()) {
        this->exportSettings.extraQuantizationParams.value() = "--sym --ratio 1.0 --group-size -1";
    }
    // clang-format off
    oss << this->OPTIMUM_CLI_EXPORT_COMMAND;
    oss << "--model " << this->sourceModel << " --trust-remote-code ";
    oss << " --weight-format " << this->exportSettings.precision << " ";
    if (this->exportSettings.extraQuantizationParams.has_value()) {
        oss << this->exportSettings.extraQuantizationParams.value() << " ";
    }
    oss << this->downloadPath;
    // clang-format on
    return oss.str();
}

std::string OptimumDownloader::getExportCmdEmbeddings() {
    std::ostringstream oss;
    // clang-format off
    oss << this->OPTIMUM_CLI_EXPORT_COMMAND;
    oss << "--task feature-extraction --library sentence_transformers";
    oss << " --model " << this->sourceModel << " --trust-remote-code ";
    oss << " --weight-format " << this->exportSettings.precision;
    oss << " " << this->downloadPath;
    // clang-format on

    return oss.str();
}

std::string OptimumDownloader::getExportCmdRerank() {
    std::ostringstream oss;
    // clang-format off
    oss << this->OPTIMUM_CLI_EXPORT_COMMAND;
    oss << "--model " << this->sourceModel;
    oss << " --trust-remote-code ";
    oss << " --weight-format " << this->exportSettings.precision;
    oss << " --task text-classification ";
    oss << " " << this->downloadPath;
    // clang-format on

    return oss.str();
}

std::string OptimumDownloader::getExportCmdImageGeneration() {
    std::ostringstream oss;
    // clang-format off
    oss << this->OPTIMUM_CLI_EXPORT_COMMAND;
    oss << "--model " << this->sourceModel;
    oss << " --weight-format " << this->exportSettings.precision;
    oss << " " << this->downloadPath;
    // clang-format on

    return oss.str();
}

std::string OptimumDownloader::getExportCmd() {
    std::string cmd = "";
    switch (this->task) {
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

std::string OptimumDownloader::getConvertCmd() {
    std::ostringstream oss;
    // clang-format off
    oss << this->CONVERT_TOKENIZER_EXPORT_COMMAND;
    oss << this->sourceModel;
    oss << " --with-detokenizer -o ";
    oss << this->downloadPath;
    // clang-format on

    return oss.str();
}

bool OptimumDownloader::checkIfDetokenizerFileIsExported() {
    return std::filesystem::exists(FileSystem::joinPath({this->downloadPath, "openvino_detokenizer.xml"}));
}

OptimumDownloader::OptimumDownloader(const ExportSettings& inExportSettings, const GraphExportType& inTask,
    const std::string& inSourceModel, const std::string& inDownloadPath, bool inOverwrite, const std::string& cliExportCmd,
    const std::string& cliCheckCmd, const std::string& convertExportCmd, const std::string& convertCheckCmd) :
    IModelDownloader(inSourceModel, inDownloadPath, inOverwrite),
    exportSettings(inExportSettings),
    task(inTask),
    OPTIMUM_CLI_CHECK_COMMAND(cliCheckCmd),
    OPTIMUM_CLI_EXPORT_COMMAND(cliExportCmd),
    CONVERT_TOKENIZER_CHECK_COMMAND(convertCheckCmd),
    CONVERT_TOKENIZER_EXPORT_COMMAND(convertExportCmd) {}

Status OptimumDownloader::checkRequiredToolsArePresent() {
    int retCode = -1;
    std::string output = exec_cmd(this->OPTIMUM_CLI_CHECK_COMMAND, retCode);
    if (retCode != 0) {
        SPDLOG_DEBUG("Command output {}", output);
        SPDLOG_ERROR("Trying to pull {} from HuggingFace but missing optimum-intel. Use the ovms package with optimum-intel.", this->sourceModel);
        return StatusCode::HF_FAILED_TO_INIT_OPTIMUM_CLI;
    }

    SPDLOG_DEBUG("Optimum-cli executable is present");

    output = exec_cmd(this->CONVERT_TOKENIZER_CHECK_COMMAND, retCode);
    if (retCode != 0) {
        SPDLOG_DEBUG("Command output {}", output);
        SPDLOG_ERROR("Trying to pull {} from HuggingFace but missing convert_tokenizer. Use the ovms package with convert_tokenizer.", this->sourceModel);
        return StatusCode::HF_FAILED_TO_INIT_OPTIMUM_CLI;
    }

    SPDLOG_DEBUG("Convert_tokenizer executable is present");
    return StatusCode::OK;
}

Status OptimumDownloader::downloadModel() {
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

    status = IModelDownloader::checkIfOverwriteAndRemove();
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

    if (!this->checkIfDetokenizerFileIsExported()) {
        SPDLOG_DEBUG("Detokenizer not found in the exported model. Exporting tokenizer and detokenizer from HF model.");
        cmd = getConvertCmd();
        retCode = -1;
        output = exec_cmd(cmd, retCode);
        if (retCode != 0) {
            SPDLOG_DEBUG("Command output {}", output);
            SPDLOG_ERROR("convert_tokenizer command failed.");
            return StatusCode::HF_RUN_CONVERT_TOKENIZER_EXPORT_FAILED;
        }
    } else {
        SPDLOG_DEBUG("Detokenizer is found in the exported model directory. Convert_tokenizer command not required.");
    }

    return StatusCode::OK;
}

}  // namespace ovms

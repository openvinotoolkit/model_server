//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "oci_downloader.hpp"

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "src/port/rapidjson_document.hpp"

#include "../capi_frontend/server_settings.hpp"
#include "../logging.hpp"
#include "../status.hpp"
#include "cmd_exec.hpp"
#include "model_downloader.hpp"
#include "optimum_export.hpp"
#include "src/filesystem/filesystem.hpp"

namespace ovms {

static const char* LLMMAN_BIN_ENV_VAR = "LLMMAN_BIN";
static const char* DEFAULT_LLMMAN_BINARY = "llmman";

std::string OciDownloader::resolveLlmmanBinary() {
    const char* override = std::getenv(LLMMAN_BIN_ENV_VAR);
    if (override != nullptr && std::string(override).length() > 0) {
        return override;
    }
    return DEFAULT_LLMMAN_BINARY;
}

OciDownloader::OciDownloader(const ExportSettings& inExportSettings, const GraphExportType& inTask,
    const std::string& inSourceModel, const std::string& inDownloadPath, bool inOverwrite,
    const std::string& inLlmmanBinary) :
    IModelDownloader(inSourceModel, inDownloadPath, inOverwrite),
    exportSettings(inExportSettings),
    task(inTask),
    llmmanBinary(inLlmmanBinary.empty() ? resolveLlmmanBinary() : inLlmmanBinary) {}

std::string OciDownloader::getVersionCmd() const {
    std::ostringstream oss;
    oss << this->llmmanBinary << " --version";
    return oss.str();
}

std::string OciDownloader::getResolveCmd() const {
    std::ostringstream oss;
    // Quoting keeps a reference containing shell-significant characters in a
    // single argv entry. exec_cmd() never spawns a shell, so this is only
    // about argument splitting, not injection.
    oss << this->llmmanBinary << " resolve \"" << stripOciScheme(this->sourceModel) << "\"";
    return oss.str();
}

Status OciDownloader::checkLlmmanIsPresent() {
    int retCode = -1;
    const std::string output = exec_cmd(this->getVersionCmd(), retCode);
    if (retCode != 0) {
        SPDLOG_DEBUG("Command output {}", output);
        SPDLOG_ERROR("Trying to pull {} but the llmman executable was not found. Install it from "
                     "https://github.com/llmmanorg/llmman or point {} at its full path.",
            this->sourceModel, LLMMAN_BIN_ENV_VAR);
        return StatusCode::OCI_LLMMAN_NOT_FOUND;
    }
    SPDLOG_DEBUG("llmman executable is present");
    return StatusCode::OK;
}

Status OciDownloader::parseResolveOutput(const std::string& output, std::string& outPath, std::string& outFormat) {
    std::vector<std::string> lines;
    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }

    for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
        rapidjson::Document document;
        if (document.Parse(it->c_str()).HasParseError() || !document.IsObject()) {
            continue;
        }
        if (!document.HasMember("path") || !document["path"].IsString()) {
            continue;
        }
        if (!document.HasMember("format") || !document["format"].IsString()) {
            continue;
        }
        outPath = document["path"].GetString();
        outFormat = document["format"].GetString();
        return StatusCode::OK;
    }

    SPDLOG_ERROR("Could not parse llmman resolve output. Expected a single line of JSON with \"path\" and \"format\" members.");
    SPDLOG_DEBUG("Command output {}", output);
    return StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID;
}

bool OciDownloader::containsOpenVinoIr(const std::string& directory) {
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec)) {
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const auto& path = entry.path();
        if (path.extension() != ".xml") {
            continue;
        }
        auto weights = path;
        weights.replace_extension(".bin");
        if (std::filesystem::exists(weights, ec)) {
            return true;
        }
    }
    return false;
}

Status OciDownloader::convertToOpenVinoIr(const std::string& resolvedPath) {
    SPDLOG_INFO("OCI model {} contains a HuggingFace-format checkout. Converting it to OpenVINO IR with optimum-cli.", this->sourceModel);
    // optimum-cli accepts a local directory for --model, so the checkout that
    // llmman produced is passed straight through as the export source. The
    // conversion output lands in the graph directory, which keeps models_path
    // at its default of "./".
    OptimumDownloader optimumDownloader(this->exportSettings, this->task, resolvedPath, this->downloadPath, this->overwriteModels);
    auto status = optimumDownloader.downloadModel();
    if (!status.ok()) {
        return status;
    }
    this->modelPath = "./";
    return StatusCode::OK;
}

Status OciDownloader::downloadModel() {
    if (FileSystem::isPathEscaped(this->downloadPath)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", this->downloadPath);
        return StatusCode::PATH_INVALID;
    }

    auto status = this->checkLlmmanIsPresent();
    if (!status.ok()) {
        return status;
    }

    status = IModelDownloader::checkIfOverwriteAndRemove();
    if (!status.ok()) {
        return status;
    }

    const std::string cmd = this->getResolveCmd();
    SPDLOG_DEBUG("Executing command: {}", cmd);
    int retCode = -1;
    const std::string output = exec_cmd(cmd, retCode);
    if (retCode != 0) {
        SPDLOG_ERROR("llmman resolve failed for {}: {}", this->sourceModel, output);
        return StatusCode::OCI_LLMMAN_RESOLVE_FAILED;
    }

    std::string resolvedPath;
    std::string format;
    status = parseResolveOutput(output, resolvedPath, format);
    if (!status.ok()) {
        return status;
    }

    std::error_code ec;
    if (!std::filesystem::exists(resolvedPath, ec)) {
        SPDLOG_ERROR("llmman resolved {} to {}, which does not exist.", this->sourceModel, resolvedPath);
        return StatusCode::OCI_LLMMAN_RESOLVE_OUTPUT_INVALID;
    }
    SPDLOG_DEBUG("llmman resolved {} to {} (format: {})", this->sourceModel, resolvedPath, format);

    if (format == "gguf") {
        // models_path must point at the GGUF file itself, which the graph
        // exporter builds by joining the directory with ggufFilename.
        const std::filesystem::path ggufPath(resolvedPath);
        this->modelPath = ggufPath.parent_path().string();
        this->ggufFilename = ggufPath.filename().string();
    } else if (format == "safetensors") {
        if (containsOpenVinoIr(resolvedPath)) {
            // Already an OpenVINO IR ModelPack - serve it straight from
            // llmman's store, no conversion and no second copy on disk.
            this->modelPath = resolvedPath;
        } else {
            status = this->convertToOpenVinoIr(resolvedPath);
            if (!status.ok()) {
                return status;
            }
        }
    } else {
        SPDLOG_ERROR("llmman reported unsupported format \"{}\" for {}. Supported formats: gguf, safetensors.", format, this->sourceModel);
        return StatusCode::OCI_UNSUPPORTED_MODEL_FORMAT;
    }

    // The graph directory holds graph.pbtxt even when the weights stay in
    // llmman's store, so it has to exist before the graph is exported.
    std::filesystem::create_directories(this->downloadPath, ec);
    if (ec) {
        SPDLOG_ERROR("Failed to create directory {}: {}", this->downloadPath, ec.message());
        return StatusCode::PATH_INVALID;
    }

    return StatusCode::OK;
}

}  // namespace ovms

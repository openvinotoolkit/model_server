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
#include "config_export.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "../capi_frontend/server_settings.hpp"
#include "src/filesystem.hpp"
#include "src/modelextensions.hpp"
#include "src/logging.hpp"

namespace ovms {

static Status createConfigTemplate(const std::string& directoryPath, const ModelSettingsImpl& modelSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << R"(
    {
        "model_config_list": [
            { "config":
                {
                    "name": ")" << graphSettings.modelName << R"(_tokenizer_model",
                    "base_path": "tokenizer"
                }
            },
            { "config":
                {
                    "name": ")" << graphSettings.modelName << R"(_embeddings_model",
                    "base_path": "embeddings",
                    "target_device": ")" << graphSettings.targetDevice << R"(",
                    "plugin_config": { "NUM_STREAMS": ")" << graphSettings.numStreams << R"(" }
                }
            }
        ]
    })";
    // clang-format on
    std::string fullPath = FileSystem::joinPath({directoryPath, "subconfig.json"});
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

Status createConfig(const std::string& directoryPath, const ModelSettingsImpl& modelSettings, const ConfigExportType& exportType) {
    if (directoryPath.empty() || !std::filesystem::exists(directoryPath)) {
        SPDLOG_ERROR("Directory path empty or does not exist: {}", directoryPath);
        return StatusCode::PATH_INVALID;
    }

    if (exportType == enable_model) {
            return createConfigTemplate(directoryPath, modelSettings);
    } else if (exportType == enable_model) {
        if (std::holds_alternative<EmbeddingsGraphSettingsImpl>(hfSettings.graphSettings)) {
            return createEmbeddingsGraphTemplate(directoryPath, std::get<EmbeddingsGraphSettingsImpl>(hfSettings.graphSettings));
        } else {
            SPDLOG_ERROR("Graph options not initialized for embeddings.");
            return StatusCode::INTERNAL_ERROR;
        }
    } else if (exportType == delete_model) {
        SPDLOG_ERROR("Delete not supported.");
        return StatusCode::INTERNAL_ERROR;
    } else if (exportType == unknown_model) {
        SPDLOG_ERROR("Graph options not initialized.");
        return StatusCode::INTERNAL_ERROR;
    }
}
}  // namespace ovms

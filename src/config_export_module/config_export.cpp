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
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#pragma warning(pop)

#include "../capi_frontend/server_settings.hpp"
#include "src/filesystem.hpp"
#include "src/logging.hpp"
#include "src/schema.hpp"
#include "src/status.hpp"

namespace ovms {

Status loadJsonConfig(const std::string& jsonFilename, rapidjson::Document& configJson) {
    //std::lock_guard<std::recursive_mutex> loadingLock(configMtx);
    std::string md5;
    Status status = parseConfig(jsonFilename, configJson, md5);
    if (!status.ok()) {
        return status;
    }
    SPDLOG_DEBUG("Parsing configuration file success: {}", jsonFilename);
    if (validateJsonAgainstSchema(configJson, MODELS_CONFIG_SCHEMA.c_str()) != StatusCode::OK) {
        SPDLOG_ERROR("Configuration file is not in valid configuration format: {}", jsonFilename);
        return status;
    }
    SPDLOG_DEBUG("Validating configuration file success: {}", jsonFilename);

    return StatusCode::OK;
}

Status createModelConfig(const std::string& fullPath, const ModelsSettingsImpl& modelSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << R"(
    {
        "model_config_list": [
            { "config":
                {
                    "name": ")" << modelSettings.modelName << R"(",
                    "base_path": ")" << modelSettings.modelPath << R"("
                }
            }
        ]
    })";
    // clang-format on
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

Status removeModelFromConfig(const std::string& fullPath, const ModelsSettingsImpl& modelSettings) {
    rapidjson::Document configJson;
    auto status = loadJsonConfig(fullPath, configJson);
    if (!status.ok()) {
        return status;
    }

    const auto modelsItr = configJson.FindMember("model_config_list");
    if (modelsItr == configJson.MemberEnd() || !modelsItr->value.IsArray()) {
        SPDLOG_ERROR("Configuration file doesn't have models property.");
        return StatusCode::JSON_INVALID;
    }

    bool erased = false;
    for (const auto& config : modelsItr->value.GetArray()) {
        auto checkItemDelete =  config.FindMember("config");
        if( checkItemDelete != config.MemberEnd() && config["name"].GetString() == modelSettings.modelName) 
        {
            configJson.Erase(&config);
            erased = true;
            break;
        }
    }

    if (!erased) {
        SPDLOG_ERROR("Configuration file doesn't have model with name {}.", modelSettings.modelName);
        return StatusCode::MODEL_NAME_MISSING;
    }

    SPDLOG_DEBUG("Model to be removed found in configuration file: {}", fullPath);
    std::string configString = "{ }";
    // Serialize the document to a JSON string
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    configJson.Accept(writer);

    // Output the JSON string
    configString = buffer.GetString();

    return FileSystem::createFileOverwrite(fullPath, configString);
}

Status updateConfigAddModel(const std::string& fullPath, const ModelsSettingsImpl& modelSettings) {
    std::ostringstream oss;
    // clang-format off
    oss << R"(
    {
        "model_config_list": [
            { "config":
                {
                    "name": ")" << modelSettings.modelName << R"(",
                    "base_path": ")" << modelSettings.modelPath << R"("
                }
            }
        ]
    })";
    // clang-format on
    return FileSystem::createFileOverwrite(fullPath, oss.str());
}

Status EnableModel(const std::string& configDirectoryPath, const ModelsSettingsImpl& modelSettings){
    std::string fullPath = FileSystem::joinPath({configDirectoryPath, "config.json"});
    if (std::filesystem::exists(fullPath)) {
        return updateConfigAddModel(fullPath, modelSettings);
    } else {
        return createModelConfig(fullPath, modelSettings);
    }
}

Status DisableModel(const std::string& configDirectoryPath, const ModelsSettingsImpl& modelSettings){
    std::string fullPath = FileSystem::joinPath({configDirectoryPath, "config.json"});
    if (std::filesystem::exists(fullPath)) {
        return removeModelFromConfig(fullPath, modelSettings);
    } else {
        SPDLOG_ERROR("Config path does not exist: {}", fullPath);
        return StatusCode::PATH_INVALID;
    }
}

Status createConfig(const ModelsSettingsImpl& modelSettings, const ConfigExportType& exportType) {
    if (modelSettings.configPath.empty()) {
        SPDLOG_ERROR("Directory path empty: {}", modelSettings.configPath);
        return StatusCode::PATH_INVALID;
    }

    if (exportType == enable_model) {
        return EnableModel(modelSettings.configPath, modelSettings);
    } else if (exportType == disable_model) {
        return DisableModel(modelSettings.configPath, modelSettings);
    } else if (exportType == delete_model) {
        SPDLOG_ERROR("Delete not supported.");
        return StatusCode::INTERNAL_ERROR;
    } else if (exportType == unknown_model) {
        SPDLOG_ERROR("Config creation options not initialized.");
        return StatusCode::INTERNAL_ERROR;
    }

    return StatusCode::INTERNAL_ERROR;
}
}  // namespace ovms

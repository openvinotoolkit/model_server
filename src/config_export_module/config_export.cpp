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

#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#pragma warning(pop)

#include "../capi_frontend/server_settings.hpp"
#include "src/filesystem.hpp"
#include "src/localfilesystem.hpp"
#include "src/logging.hpp"
#include "src/schema.hpp"
#include "src/status.hpp"

namespace ovms {

Status loadJsonConfig(const std::string& jsonFilename, rapidjson::Document& configJson) {
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
    oss << R"({
    "model_config_list": [
        { 
            "config": {
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

    auto modelsItr = configJson.FindMember("model_config_list");
    if (modelsItr == configJson.MemberEnd() || !modelsItr->value.IsArray()) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Configuration file doesn't have models property.");
        return StatusCode::JSON_INVALID;
    }

    bool erased = false;
    for (auto& config : modelsItr->value.GetArray()) {
        auto checkItemDelete = config.FindMember("config");
        if (checkItemDelete != config.MemberEnd() && config["config"].HasMember("name") && config["config"]["name"].GetString() == modelSettings.modelName) {
            SPDLOG_DEBUG("Erasing model from config: {}", modelSettings.modelName);
            modelsItr->value.Erase(&config);
            erased = true;
            break;
        }
    }

    if (!erased) {
        SPDLOG_ERROR("Configuration file doesn't have model with name: {}.", modelSettings.modelName);
        return StatusCode::MODEL_NAME_MISSING;
    }

    SPDLOG_DEBUG("Model to be removed found in configuration file: {}", fullPath);
    // Serialize the document to a JSON string
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    configJson.Accept(writer);

    // Output the JSON string
    std::string configString = buffer.GetString();

    return FileSystem::createFileOverwrite(fullPath, configString);
}

Status updateConfigAddModel(const std::string& fullPath, const ModelsSettingsImpl& modelSettings) {
    rapidjson::Document configJson;
    auto status = loadJsonConfig(fullPath, configJson);
    if (!status.ok()) {
        return status;
    }

    const auto modelsItr = configJson.FindMember("model_config_list");
    if (modelsItr == configJson.MemberEnd() || !modelsItr->value.IsArray()) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Configuration file doesn't have models property.");
        return StatusCode::JSON_INVALID;
    }

    bool alreadyAdded = false;
    for (const auto& config : modelsItr->value.GetArray()) {
        auto checkItemDelete = config.FindMember("config");
        if (checkItemDelete != config.MemberEnd() && config["config"].HasMember("name") && config["config"]["name"].GetString() == modelSettings.modelName) {
            alreadyAdded = true;
            break;
        }
    }

    if (alreadyAdded) {
        SPDLOG_ERROR("Could not add model to configuration file: {}. Model with the same name already exists.", modelSettings.modelName);
        return StatusCode::MODEL_NAME_OCCUPIED;
    }

    auto alloc = configJson.GetAllocator();

    rapidjson::Value newConfig;
    newConfig.SetObject();
    rapidjson::Value name;
    name.SetString(modelSettings.modelName.c_str(), alloc);
    newConfig.AddMember("name", name, alloc);
    rapidjson::Value path;
    path.SetString(modelSettings.modelPath.c_str(), alloc);
    newConfig.AddMember("base_path", path, alloc);

    rapidjson::Value newConfigItem;
    newConfigItem.SetObject();
    newConfigItem.AddMember("config", newConfig, alloc);

    rapidjson::Value& array = configJson["model_config_list"];
    array.PushBack(newConfigItem, alloc);

    SPDLOG_DEBUG("Model to be added to configuration file: {}", fullPath);
    // Serialize the document to a JSON string
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    configJson.Accept(writer);

    // Output the JSON string
    std::string configString = buffer.GetString();

    return FileSystem::createFileOverwrite(fullPath, configString);
}

Status EnableModel(const std::string& configFilePath, const ModelsSettingsImpl& modelSettings) {
    bool exists;
    auto status = LocalFileSystem::exists(configFilePath, &exists);
    if (!status.ok()) {
        return status;
    }
    if (exists) {
        return updateConfigAddModel(configFilePath, modelSettings);
    } else {
        return createModelConfig(configFilePath, modelSettings);
    }
}

Status DisableModel(const std::string& configFilePath, const ModelsSettingsImpl& modelSettings) {
    bool exists;
    auto status = LocalFileSystem::exists(configFilePath, &exists);
    if (!status.ok()) {
        return status;
    }
    if (exists) {
        return removeModelFromConfig(configFilePath, modelSettings);
    } else {
        SPDLOG_ERROR("Config path does not exist: {}", configFilePath);
        return StatusCode::PATH_INVALID;
    }
}

Status updateConfig(const ModelsSettingsImpl& modelSettings, const ConfigExportType& exportType) {
    std::string configFilePath = modelSettings.configPath;
    if (configFilePath.empty()) {
        SPDLOG_ERROR("Config path is empty.");
        return StatusCode::PATH_INVALID;
    }
    if (exportType == ENABLE_MODEL) {
        return EnableModel(configFilePath, modelSettings);
    } else if (exportType == DISABLE_MODEL) {
        return DisableModel(configFilePath, modelSettings);
    } else if (exportType == DELETE_MODEL) {
        SPDLOG_ERROR("Delete not supported.");
        return StatusCode::NOT_IMPLEMENTED;
    } else if (exportType == UNKNOWN_MODEL) {
        SPDLOG_ERROR("Config creation options not initialized.");
        return StatusCode::NOT_IMPLEMENTED;
    }

    return StatusCode::INTERNAL_ERROR;
}
}  // namespace ovms

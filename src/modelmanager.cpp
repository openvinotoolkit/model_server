//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <sys/stat.h>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <spdlog/spdlog.h>

#include "config.hpp"
#include "modelmanager.hpp"

namespace ovms {

const uint ModelManager::watcherIntervalSec = 1;

Status ModelManager::start() {
    auto& config = ovms::Config::instance();
    // start manager using config file
    if (config.configPath() != "") {
        return start(config.configPath());
    }

    // start manager using commandline parameters
    ModelConfig modelConfig {
        config.modelName(),
        config.modelPath(),
        config.targetDevice(),
        config.batchSize(),
        config.nireq()
    };
    auto status = modelConfig.parsePluginConfig(config.pluginConfig());
    if (status != Status::OK) {
        spdlog::error("Couldn't parse plugin config: {}", StatusDescription::getError(status));
        return status;
    }

    status = modelConfig.parseModelVersionPolicy(config.modelVersionPolicy());
    if (status != Status::OK) {
        spdlog::error("Couldn't parse model version policy: {}", StatusDescription::getError(status));
        return status;
    }

    return loadModelWithVersions(modelConfig);
}

Status ModelManager::start(const std::string& jsonFilename) {
    Status s = loadConfig(jsonFilename);
    if (s != Status::OK) {
        return s;
    }

    if (!monitor.joinable()) {
        std::future<void> exitSignal = exit.get_future();
        std::thread t(std::thread(&ModelManager::watcher, this, std::move(exitSignal)));
        monitor = std::move(t);
        monitor.detach();
    }

    return Status::OK;
}

Status ModelManager::loadConfig(const std::string& jsonFilename) {
    spdlog::info("Loading configuration from {}", jsonFilename);

    rapidjson::Document doc;
    std::ifstream ifs(jsonFilename.c_str());
    // Perform some basic checks on the config file
    if (!ifs.good()) {
        spdlog::error("File is invalid {}", jsonFilename);
        return Status::FILE_INVALID;
    }

    rapidjson::IStreamWrapper isw(ifs);
    if (doc.ParseStream(isw).HasParseError()) {
        spdlog::error("Configuration file is not a valid JSON file.");
        return Status::JSON_INVALID;
    }

    // TODO validate json against schema
    const auto itr = doc.FindMember("model_config_list");
    if (itr == doc.MemberEnd() || !itr->value.IsArray()) {
        spdlog::error("Configuration file doesn't have models property.");
        return Status::JSON_INVALID;
    }

    models.clear();
    configFilename = jsonFilename;
    for (const auto& configs : itr->value.GetArray()) {
        ModelConfig modelConfig;
        modelConfig.parseNode(configs["config"]);
        loadModelWithVersions(modelConfig);
    }

    return Status::OK;
}

void ModelManager::watcher(std::future<void> exit) {
    spdlog::info("Started config watcher thread");
    int64_t lastTime;
    struct stat statTime;

    stat(configFilename.c_str(), &statTime);
    lastTime = statTime.st_ctime;
    while (exit.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
        std::this_thread::sleep_for(std::chrono::seconds(watcherIntervalSec));
        stat(configFilename.c_str(), &statTime);
        if (lastTime != statTime.st_ctime) {
            lastTime = statTime.st_ctime;
            loadConfig(configFilename);
            spdlog::info("Model configuration changed");
        }
    }
    spdlog::info("Exited config watcher thread");
}

void ModelManager::join() {
    exit.set_value();
    if (monitor.joinable()) {
        monitor.join();
    }
}

Status ModelManager::readAvailableVersions(const std::string& path, std::vector<model_version_t>& versions) {
    std::filesystem::directory_iterator it;
    try {
        it = std::filesystem::directory_iterator(path);
    } catch (const std::filesystem::filesystem_error&) {
        return Status::PATH_INVALID;
    }

    for (const auto& entry : it) {
        if (!entry.is_directory()) {
            continue;
        }
        try {
            ovms::model_version_t version = std::stoll(entry.path().filename().string());
            versions.push_back(version);
        } catch(const std::exception& e) {
            continue;
        }
    }

    return Status::OK;
}

Status ModelManager::loadModelWithVersions(ModelConfig& config) {
    std::vector<model_version_t> versions;
    auto basePath = config.getBasePath();
    auto status = readAvailableVersions(basePath, versions);
    if  (status != Status::OK) {
        return status;
    }

    // TODO: In future, unload models
    if (versions.size() == 0) {
        return Status::OK;
    }

    models[config.getName()] = modelFactory();

    versions = config.getModelVersionPolicy()->filter(versions);
    for (const auto version : versions) {
        config.setVersion(version);
        config.setBasePath(basePath + "/" + std::to_string(version));
        auto status = config.parseModelMapping();
        if ((status != Status::OK) && (status != Status::FILE_INVALID)) {
            spdlog::error("Error while parsing model mapping for model {}", StatusDescription::getError(status));
        }

        status = models[config.getName()]->addVersion(config);
        if (status != Status::OK) {
            spdlog::error("Error while loading model: {}; version: {}; error: {}",
                config.getName(),
                version,
                StatusDescription::getError(status));
        }
    }

    return Status::OK;
}

}  // namespace ovms

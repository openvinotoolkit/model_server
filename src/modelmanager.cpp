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
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <spdlog/spdlog.h>

#include "config.hpp"
#include "modelmanager.hpp"

namespace ovms {

const uint ModelManager::watcherIntervalSec = 1;
const std::string MAPPING_CONFIG_JSON = "mapping_config.json";

Status ModelManager::start() {
    auto& config = ovms::Config::instance();

    // start manager using config file
    if (config.configPath() != "")
        return start(config.configPath());

    // start manager using commandline parameters
    ModelConfig modelConfig {
        config.modelName(),
        config.modelPath(),
        config.targetDevice(),
        config.batchSize(),
        config.nireq(),
        config.modelVersionPolicy()
    };

    if (config.pluginConfig() != "") {
        plugin_config_t pluginConfig;
        auto status = ModelManager::parsePluginConfig(config.pluginConfig(), pluginConfig);
        if (status != Status::OK) {
            return status;
        }
        modelConfig.setPluginConfig(pluginConfig);
    }

    return loadModelWithVersions(config.modelPath(), modelConfig);
}

Status ModelManager::start(const std::string& jsonFilename) {
    Status s = loadConfig(jsonFilename);
    if (s != Status::OK) {
        return s;
    }

    std::future<void> exitSignal = exit.get_future();
    std::thread t(std::thread(&ModelManager::watcher, this, std::move(exitSignal)));
    monitor = std::move(t);
    monitor.detach();

    return Status::OK;
}

Status ModelManager::loadConfig(const std::string& jsonFilename) {
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
        const auto& v = configs["config"].GetObject();
        ModelConfig modelConfig;
        modelConfig.setName(v["name"].GetString());


        // Check for optional parameters
        if (v.HasMember("batch_size")) {
            if (v["batch_size"].IsString()) {
                // Although batch size is in, in legacy python version it was string
                modelConfig.setBatchSize(std::atoi(v["batch_size"].GetString()));
            } else {
                modelConfig.setBatchSize(v["batch_size"].GetUint64());
            }
        }
        if (v.HasMember("target_device"))
            modelConfig.setBackend(v["target_device"].GetString());
        if (v.HasMember("version"))
            modelConfig.setVersion(v["version"].GetInt64());
        if (v.HasMember("nireq"))
            modelConfig.setNireq(v["nireq"].GetUint64());

        if (v.HasMember("shape")) {
            // Legacy format as string
            if (v["shape"].IsString()) {
                if (modelConfig.setShape(v["shape"].GetString()) != Status::OK) {
                    spdlog::error("There was an error parsing shape {}", v["shape"].GetString());
                }
            } else {
                if (v["shape"].IsArray()) {
                    // Shape for all inputs
                    shape_t shape;
                    for (auto& sh : v["shape"].GetArray()) {
                        shape.push_back(sh.GetUint64());
                    }
                    modelConfig.setShape(shape);
                } else {
                    // Map of shapes
                    for (auto& s : v["shape"].GetObject()) {
                        shape_t shape;
                        // check if legacy format is used
                        if (s.value.IsString()) {
                            if (ModelConfig::parseShape(shape, s.value.GetString()) != Status::OK) {
                                spdlog::error("There was an error parsing shape {}", v["shape"].GetString());
                            }
                        } else {
                            for (auto& sh : s.value.GetArray()) {
                                shape.push_back(sh.GetUint64());
                            }
                        }
                        modelConfig.addShape(s.name.GetString(), shape);
                    }
                }
            }
        }

        if (v.HasMember("layout")) {
            if (v["layout"].IsString()) {
                modelConfig.setLayout(v["layout"].GetString());
            } else {
                for (auto& s : v["layout"].GetObject()) {
                    modelConfig.addLayout(s.name.GetString(), s.value.GetString());
                }
            }
        }

        if (v.HasMember("plugin_config")) {
            plugin_config_t pluginConfig;
            auto status = ModelManager::parsePluginConfig(v["plugin_config"], pluginConfig);
            if (status != Status::OK) {
                spdlog::error("Couldn't parse plugin config: {}", StatusDescription::getError(status));
                return status;
            }
            modelConfig.setPluginConfig(pluginConfig);
        }

        if (v.HasMember("model_version_policy")) {
            rapidjson::StringBuffer buffer;
            buffer.Clear();
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
            v["model_version_policy"].Accept(writer);
            modelConfig.setModelVersionPolicy(buffer.GetString());
        }

        loadModelWithVersions(v["base_path"].GetString(), modelConfig);
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
    monitor.join();
}

Status ModelManager::parsePluginConfig(const rapidjson::Value& node, plugin_config_t& config) {
    if (!node.IsObject()) {
        return Status::PLUGIN_CONFIG_ERROR;
    }

    for (auto it = node.MemberBegin(); it != node.MemberEnd(); ++it) {
        if (!it->value.IsString()) {
            return Status::PLUGIN_CONFIG_ERROR;
        }
        config[it->name.GetString()] = it->value.GetString();
    }

    return Status::OK;
}

Status ModelManager::parsePluginConfig(std::string command, plugin_config_t& config) {
    rapidjson::Document node;
    if (node.Parse(command.c_str()).HasParseError()) {
        return Status::PLUGIN_CONFIG_ERROR;
    }
    return ModelManager::parsePluginConfig(node, config);
}

void ModelManager::parseModelMapping(const std::string& base,
                                     mapping_config_t& mappingInputs,
                                     mapping_config_t& mappingOutputs) {
    rapidjson::Document doc;

    std::filesystem::path path = base;
    path.append(MAPPING_CONFIG_JSON);

    std::ifstream ifs(path.c_str());
    if (!ifs.good()) {
        return;
    }

    rapidjson::IStreamWrapper isw(ifs);
    if (doc.ParseStream(isw).HasParseError()) {
        spdlog::warn("Couldn't load {} from dir {}", MAPPING_CONFIG_JSON, base);
        return;
    }

    // Process inputs
    const auto itr = doc.FindMember("inputs");
    if (itr == doc.MemberEnd() || !itr->value.IsObject()) {
        spdlog::warn("Couldn't load inputs object in file {} from dir {}", MAPPING_CONFIG_JSON, base);
    } else {
        for (const auto& key : itr->value.GetObject()) {
            spdlog::debug("Loaded mapping {} => {}", key.name.GetString(), key.value.GetString());
            mappingInputs[key.name.GetString()] = key.value.GetString();
        }
    }

    // Process outputs
    const auto it = doc.FindMember("outputs");
    if (it == doc.MemberEnd() || !it->value.IsObject()) {
        spdlog::warn("Couldn't load outputs object in file {} from dir {}", MAPPING_CONFIG_JSON, base);
    } else {
        for (const auto& key : it->value.GetObject()) {
            spdlog::debug("Loaded mapping {} => {}", key.name.GetString(), key.value.GetString());
            mappingOutputs[key.name.GetString()] = key.value.GetString();
        }
    }
}

Status ModelManager::parseModelVersionPolicy(std::string command, std::shared_ptr<ModelVersionPolicy>& policy) {
    rapidjson::Document node;
    if (node.Parse(command.c_str()).HasParseError()) {
        return Status::MODEL_VERSION_POLICY_ERROR;
    }

    if (!node.IsObject()) {
        return Status::MODEL_VERSION_POLICY_ERROR;
    }
    if (node.MemberCount() != 1) {
        return Status::MODEL_VERSION_POLICY_ERROR;
    }

    auto m = node.FindMember("all");
    if (m != node.MemberEnd()) {
        policy = std::make_shared<AllModelVersionPolicy>();
        return Status::OK;
    }

    m = node.FindMember("specific");
    if (m != node.MemberEnd()) {
        auto& specific = m->value;
        if (specific.MemberCount() != 1) {
            return Status::MODEL_VERSION_POLICY_ERROR;
        }
        m = specific.FindMember("versions");
        if (m == specific.MemberEnd()) {
            return Status::MODEL_VERSION_POLICY_ERROR;
        }
        std::vector<model_version_t> versions;
        for (auto& version : m->value.GetArray()) {
            versions.push_back(version.GetInt64());
        }
        policy = std::make_shared<SpecificModelVersionPolicy>(versions);
        return Status::OK;
    }

    m = node.FindMember("latest");
    if (m != node.MemberEnd()) {
        auto& latest = m->value;
        if (latest.MemberCount() != 1) {
            return Status::MODEL_VERSION_POLICY_ERROR;
        }
        m = latest.FindMember("num_versions");
        if (m == latest.MemberEnd()) {
            return Status::MODEL_VERSION_POLICY_ERROR;
        }
        policy = std::make_shared<LatestModelVersionPolicy>(m->value.GetInt64());
        return Status::OK;
    }

    return Status::MODEL_VERSION_POLICY_ERROR;
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

Status ModelManager::loadModelWithVersions(const std::string& basePath, ModelConfig& config) {
    std::vector<model_version_t> versions;
    auto status = readAvailableVersions(basePath, versions);
    if  (status != Status::OK) {
        return status;
    }

    // TODO: In future, unload models
    if (versions.size() == 0) {
        return Status::OK;
    }

    std::shared_ptr<ModelVersionPolicy> policy = nullptr;
    if (config.getModelVersionPolicy() == "") {
        policy = ModelVersionPolicy::getDefaultVersionPolicy();
    } else {
        status = parseModelVersionPolicy(config.getModelVersionPolicy(), policy);
        if (status != Status::OK) {
            return status;
        }
    }

    versions = policy->filter(versions);

    std::shared_ptr<Model> model = std::make_shared<Model>();
    models[config.getName()] = std::move(model);

    for (const auto version : versions) {
        config.setVersion(version);
        config.setBasePath(basePath + "/" + std::to_string(version));
        mapping_config_t mappingInputs, mappingOutputs = {};
        parseModelMapping(config.getBasePath(), mappingInputs, mappingOutputs);
        config.setMappingInputs(mappingInputs);
        config.setMappingOutputs(mappingOutputs);

        auto status = models[config.getName()]->addVersion(config);
        if (status != Status::OK) {
            spdlog::info("Error while loading model: {}; version: {}; error: {}",
                config.getName(),
                version,
                StatusDescription::getError(status));
        }
    }

    return Status::OK;
}

}  // namespace ovms

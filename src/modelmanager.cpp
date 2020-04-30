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
#include <fstream>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include "config.hpp"
#include "modelmanager.hpp"

namespace ovms {

const uint ModelManager::watcherIntervalSec = 1;

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
        config.nireq()
    };
    // Until we have filesystem, we need to provide version somehow
    modelConfig.setVersion(1);

    Status status;
    if (config.pluginConfig() != "") {
        plugin_config_t pluginConfig;
        status = ModelManager::parsePluginConfig(config.pluginConfig(), pluginConfig);
        if (status != Status::OK) {
            return status;
        }
        modelConfig.setPluginConfig(pluginConfig);
    }

    std::shared_ptr<Model> model = std::make_shared<Model>();
    models[modelConfig.getName()] = std::move(model);
    status = models[modelConfig.getName()]->addVersion(modelConfig);
    if (status != Status::OK) {
        // Logger(Log::Warning, "There was an error loading a model ", config.modelName());
        return status;
    }

    return Status::OK;
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
        // Logger(Log::Error, "File is invalid ", jsonFilename);
        return Status::FILE_INVALID;
    }

    rapidjson::IStreamWrapper isw(ifs);
    if (doc.ParseStream(isw).HasParseError()) {
        // Logger(Log::Error, "Configuration file is not a valid JSON file.");
        return Status::JSON_INVALID;
    }

    // TODO validate json against schema
    const auto itr = doc.FindMember("model_config_list");
    if (itr == doc.MemberEnd() || !itr->value.IsArray()) {
        // Logger(Log::Error, "Configuration file doesn't have models property.");
        return Status::JSON_INVALID;
    }
    
    models.clear();
    configFilename = jsonFilename;
    for (const auto& configs : itr->value.GetArray()) {
        const auto& v = configs["config"].GetObject();
        std::shared_ptr<Model> model = std::make_shared<Model>();
        ModelConfig modelConfig;
        modelConfig.setName(v["name"].GetString());
        std::string base_path = v["base_path"].GetString();
        // Until we have filesystem, we need to provide version somehow
        modelConfig.setVersion(1);
        base_path.append("/1");
        modelConfig.setBasePath(base_path);
        if (models.find(modelConfig.getName()) == models.end()) {
            models[modelConfig.getName()] = std::move(model);
        }

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
                    std::cout << "There was an error parsing shape: " << v["shape"].GetString() << std::endl;
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
                                std::cout << "There was an error parsing shape: " << v["shape"].GetString() << std::endl;
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
                std::cout << StatusDescription::getError(status) << std::endl;
                return status;
            }
            modelConfig.setPluginConfig(pluginConfig);
        }

        auto status = models[modelConfig.getName()]->addVersion(modelConfig);
        if (status != Status::OK) {
            // to be replaced with proper logger
            std::cout << "There was an error loading a model: " << v["name"].GetString() << std::endl;
            std::cout << "Reason: " << StatusDescription::getError(status) << std::endl;
        }
    }

    return Status::OK;
}

void ModelManager::watcher(std::future<void> exit) {
    // Logger(Log::Info, "Started config watcher thread");
    int64_t lastTime;
    struct stat statTime;

    stat(configFilename.c_str(), &statTime);
    lastTime = statTime.st_ctime;
	while (exit.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout)
	{
        std::this_thread::sleep_for(std::chrono::seconds(watcherIntervalSec));
        stat(configFilename.c_str(), &statTime);
        if (lastTime != statTime.st_ctime) {
            lastTime = statTime.st_ctime;
            loadConfig(configFilename);
            // Logger(Log::Info, "Configuration changed");
        }
	}
    // Logger(Log::Info, "Exited config watcher thread");
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
  
} // namespace ovms

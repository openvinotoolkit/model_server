//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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
#pragma once

#include <future>
#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <rapidjson/document.h>
#include <spdlog/spdlog.h>

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "customloaders.hpp"
#include "filesystem.hpp"
#include "global_sequences_viewer.hpp"
#include "model.hpp"
#include "pipeline.hpp"
#include "pipeline_factory.hpp"

namespace ovms {

const uint WAIT_FOR_MODEL_LOADED_TIMEOUT_MS = 10000;

class IVersionReader;
class CustomNodeLibraryManager;
/**
 * @brief Model manager is managing the list of model topologies enabled for serving and their versions.
 */
class ModelManager {
protected:
    /**
     * @brief A default constructor is private
     */
    ModelManager();

    std::shared_ptr<ovms::Model> getModelIfExistCreateElse(const std::string& name, const bool isStateful);

    /**
     * @brief A collection of models
     * 
     */
    std::map<std::string, std::shared_ptr<Model>> models;

    PipelineFactory pipelineFactory;

    std::unique_ptr<CustomNodeLibraryManager> customNodeLibraryManager;

    std::unique_ptr<GlobalSequencesViewer> sequenceViewer;

private:
    /**
     * @brief Private copying constructor
     */
    ModelManager(const ModelManager&) = delete;

    Status cleanupModelTmpFiles(ModelConfig& config);
    Status reloadModelVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<FileSystem>& fs, ModelConfig& config, std::shared_ptr<model_versions_t>& versionsToReload, std::shared_ptr<model_versions_t> versionsFailed);
    Status addModelVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<FileSystem>& fs, ModelConfig& config, std::shared_ptr<model_versions_t>& versionsToStart, std::shared_ptr<model_versions_t> versionsFailed);
    Status loadModelsConfig(rapidjson::Document& configJson, std::vector<ModelConfig>& gatedModelConfigs);
    Status tryReloadGatedModelConfigs(std::vector<ModelConfig>& gatedModelConfigs);
    Status loadCustomNodeLibrariesConfig(rapidjson::Document& configJson);
    Status loadPipelinesConfig(rapidjson::Document& configJson);
    Status loadCustomLoadersConfig(rapidjson::Document& configJson);

    /**
     * @brief creates customloader from the loader configuration
     */
    Status createCustomLoader(CustomLoaderConfig& loaderConfig);

    /**
     * @brief Watcher thread for monitor changes in config
     */
    void watcher(std::future<void> exit);

    /**
     * @brief A JSON configuration filename
     */
    std::string configFilename;

    /**
     * @brief A thread object used for monitoring changes in config
     */
    std::thread monitor;

    /**
     * @brief An exit signal to notify watcher thread to exit
     */
    std::promise<void> exit;

    /**
     * @brief A current configurations of models
     * 
     */
    std::unordered_map<std::string, ModelConfig> servedModelConfigs;

    /**
     * @brief Retires models non existing in config file
     *
     * @param modelsExistingInConfigFile
     */
    void retireModelsRemovedFromConfigFile(const std::set<std::string>& modelsExistingInConfigFile);

    /**
     * @brief Mutex for protecting concurrent reloading config
     */
    mutable std::recursive_mutex configMtx;

    /**
     * Time interval between each config file check
     */
    uint watcherIntervalSec = 1;

    /**
     * @brief Time of last config change
     */
    int64_t lastConfigChangeTime;

public:
    /**
     * @brief Mutex for blocking concurrent add & find of model
     */
    mutable std::shared_mutex modelsMtx;

    /**
     * @brief Gets the instance of ModelManager
     */
    static ModelManager& getInstance() {
        static ModelManager instance;
        return instance;
    }

    /**
     *  @brief Gets the watcher interval timestep in seconds
     */
    uint getWatcherIntervalSec() {
        return watcherIntervalSec;
    }

    /**
     * @brief Destroy the Model Manager object
     * 
     */
    virtual ~ModelManager();

    /**
     * @brief Gets config filename
     * 
     * @return config filename
     */
    const std::string& getConfigFilename() {
        return configFilename;
    }

    /**
     * @brief Gets models collection
     * 
     * @return models collection
     */
    const std::map<std::string, std::shared_ptr<Model>>& getModels() {
        return models;
    }

    void startSequenceWatcher();

    const PipelineFactory& getPipelineFactory() const {
        return pipelineFactory;
    }

    const CustomNodeLibraryManager& getCustomNodeLibraryManager() const;

    /**
     * @brief Finds model with specific name
     *
     * @param name of the model to search for
     *
     * @return pointer to Model or nullptr if not found 
     */
    const std::shared_ptr<Model> findModelByName(const std::string& name) const;

    Status getModelInstance(const std::string& modelName,
        ovms::model_version_t modelVersionId,
        std::shared_ptr<ovms::ModelInstance>& modelInstance,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr);

    Status getPipeline(std::unique_ptr<ovms::Pipeline>& pipelinePtr,
        const tensorflow::serving::PredictRequest* request,
        tensorflow::serving::PredictResponse* response);

    const bool modelExists(const std::string& name) const {
        if (findModelByName(name) == nullptr)
            return false;
        else
            return true;
    }

    /**
     * @brief Finds model instance with specific name and version, returns default if version not specified
     *
     * @param name of the model to search for
     * @param version of the model to search for or 0 if default
     *
     * @return pointer to ModelInstance or nullptr if not found 
     */
    const std::shared_ptr<ModelInstance> findModelInstance(const std::string& name, model_version_t version = 0) const {
        auto model = findModelByName(name);
        if (!model) {
            return nullptr;
        }

        if (version == 0) {
            return model->getDefaultModelInstance();
        } else {
            return model->getModelInstanceByVersion(version);
        }
    }

    Status createPipeline(std::unique_ptr<Pipeline>& pipeline,
        const std::string name,
        const tensorflow::serving::PredictRequest* request,
        tensorflow::serving::PredictResponse* response) {
        return pipelineFactory.create(pipeline, name, request, response, *this);
    }

    const bool pipelineDefinitionExists(const std::string& name) const {
        return pipelineFactory.definitionExists(name);
    }

    /**
     * @brief Starts model manager using provided config file
     * 
     * @param filename
     * @return status
     */
    Status startFromFile(const std::string& jsonFilename);

    /**
     * @brief Starts model manager using command line arguments
     * 
     * @return Status 
     */
    Status startFromConfig();

    /**
     * @brief Reload model versions located in base path
     * 
     * @param ModelConfig config
     * 
     * @return status
     */
    Status reloadModelWithVersions(ModelConfig& config);

    /**
     * @brief Starts model manager using ovms::Config
     * 
     * @return status
     */
    Status start();

    /**
     * @brief Starts monitoring as new thread
     * 
     */
    void startWatcher();

    /**
     * @brief Gracefully finish the thread
     */
    void join();

    /**
     * @brief Factory for creating a model
     * 
     * @return std::shared_ptr<Model> 
     */
    virtual std::shared_ptr<Model> modelFactory(const std::string& name, const bool isStateful) {
        return std::make_shared<Model>(name, isStateful);
    }

    /**
     * @brief Reads available versions from given filesystem
     * 
     * @param fs 
     * @param base 
     * @param versions 
     * @return Status 
     */
    virtual Status readAvailableVersions(
        std::shared_ptr<FileSystem>& fs,
        const std::string& base,
        model_versions_t& versions);

    /**
     * @brief Checks what versions needs to be started, reloaded, retired based on currently served ones
     *
     * @param modelVersionsInstances map with currently served versions
     * @param requestedVersions container with requested versions
     * @param versionsToRetireIn cointainer for versions to retire
     * @param versionsToReloadIn cointainer for versions to reload
     * @param versionsToStartIn cointainer for versions to start
     */
    static void getVersionsToChange(
        const ModelConfig& newModelConfig,
        const std::map<model_version_t, std::shared_ptr<ModelInstance>>& modelVersionsInstances,
        std::vector<model_version_t> requestedVersions,
        std::shared_ptr<model_versions_t>& versionsToRetireIn,
        std::shared_ptr<model_versions_t>& versionsToReloadIn,
        std::shared_ptr<model_versions_t>& versionsToStartIn);

    static std::shared_ptr<FileSystem> getFilesystem(const std::string& basePath);

    /**
     * @brief Check if configuration file reload is needed.
     */
    bool configFileReloadNeeded();

    /**
     * @brief Reads models from configuration file
     * 
     * @param jsonFilename configuration file
     * @return Status 
     */
    Status loadConfig(const std::string& jsonFilename);

    /**
     * @brief Updates OVMS configuration with cached configuration file. Will check for newly added model versions
     */
    void updateConfigurationWithoutConfigFile();
};

}  // namespace ovms

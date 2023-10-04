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
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include <rapidjson/document.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

#include "dags/pipeline_factory.hpp"
#include "global_sequences_viewer.hpp"
#if (MEDIAPIPE_DISABLE == 0)
#include "mediapipe_internal/mediapipefactory.hpp"
#endif
#include "metric_config.hpp"
#include "model.hpp"
#include "status.hpp"

namespace ovms {

const uint32_t DEFAULT_WAIT_FOR_MODEL_LOADED_TIMEOUT_MS = 10000;
extern const std::string DEFAULT_MODEL_CACHE_DIRECTORY;

class Config;
class CNLIMWrapper;
class CustomLoaderConfig;
class CustomNodeLibraryManager;
class MetricRegistry;
class ModelConfig;
class FileSystem;
class MediapipeGraphExecutor;
struct FunctorSequenceCleaner;
struct FunctorResourcesCleaner;
class PythonBackend;
/**
 * @brief Model manager is managing the list of model topologies enabled for serving and their versions.
 */
class ModelManager {
public:
    /**
     * @brief A default constructor is private
     */
    ModelManager(const std::string& modelCacheDirectory = "", MetricRegistry* registry = nullptr, PythonBackend* pythonBackend = nullptr);

protected:
    void logPluginConfiguration();

    Status checkStatefulFlagChange(const std::string& modelName, bool configStatefulFlag);

    std::shared_ptr<ovms::Model> getModelIfExistCreateElse(const std::string& name, const bool isStateful);

    /**
     * @brief A collection of models
     * 
     */
    std::map<std::string, std::shared_ptr<Model>> models;
    std::unique_ptr<ov::Core> ieCore;

    PipelineFactory pipelineFactory;
#if (MEDIAPIPE_DISABLE == 0)
    MediapipeFactory mediapipeFactory;
#endif
    std::unique_ptr<CustomNodeLibraryManager> customNodeLibraryManager;
    std::vector<std::shared_ptr<CNLIMWrapper>> resources = {};
    GlobalSequencesViewer globalSequencesViewer;
    uint32_t waitForModelLoadedTimeoutMs;

private:
    bool watcherStarted = false;
    bool cleanerStarted = false;

    ModelManager(const ModelManager&) = delete;

    Status lastLoadConfigStatus = StatusCode::OK;

    Status cleanupModelTmpFiles(ModelConfig& config);
    Status reloadModelVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<FileSystem>& fs, ModelConfig& config, std::shared_ptr<model_versions_t>& versionsToReload, std::shared_ptr<model_versions_t> versionsFailed);
    Status addModelVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<FileSystem>& fs, ModelConfig& config, std::shared_ptr<model_versions_t>& versionsToStart, std::shared_ptr<model_versions_t> versionsFailed);
    Status loadModels(const rapidjson::Value::MemberIterator& modelsConfigList, std::vector<ModelConfig>& gatedModelConfigs, std::set<std::string>& modelsInConfigFile, std::set<std::string>& modelsWithInvalidConfig, std::unordered_map<std::string, ModelConfig>& newModelConfigs, const std::string& rootDirectoryPath);
#if (MEDIAPIPE_DISABLE == 0)
    Status processMediapipeConfig(const MediapipeGraphConfig& config, std::set<std::string>& mediapipesInConfigFile, MediapipeFactory& factory);
    Status loadMediapipeGraphsConfig(std::vector<MediapipeGraphConfig>& mediapipesInConfigFile);
    Status loadModelsConfig(rapidjson::Document& configJson, std::vector<ModelConfig>& gatedModelConfigs, std::vector<ovms::MediapipeGraphConfig>& mediapipesInConfigFile);
#else
    Status loadModelsConfig(rapidjson::Document& configJson, std::vector<ModelConfig>& gatedModelConfigs);
#endif
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
    void watcher(std::future<void> exitSignal, bool watchConfigFile);

    /**
     * @brief Cleaner thread for sequence and resources cleanup
     */
    void cleanerRoutine(uint32_t resourcesCleanupIntervalSec, uint32_t sequenceCleanerIntervalMinutes, std::future<void> cleanerExitSignal);

    /**
     * @brief Mutex for blocking concurrent add & remove of resources
     */
    mutable std::shared_mutex resourcesMtx;

    /**
     * @brief A JSON configuration filename
     */
    std::string configFilename;

    /**
     * @brief A thread object used for monitoring changes in config
     */
    std::thread monitor;

    /**
     * @brief A thread object used for cleanup
     */
    std::thread cleanerThread;

    /**
         * @brief Metrics config
         */
    MetricConfig metricConfig;

    /**
         * @brief Metrics config was loaded flag
         */
    bool metricConfigLoadedOnce = false;

    /**
     * @brief An exit trigger to notify watcher thread to exit
     */
    std::promise<void> exitTrigger;

    /**
     * @brief An exit trigger to notify cleaner thread to exit
     */
    std::promise<void> cleanerExitTrigger;

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
    void retireModelsRemovedFromConfigFile(const std::set<std::string>& modelsExistingInConfigFile, const std::set<std::string>& modelsWithInvalidConfig);

    /**
     * @brief Mutex for protecting concurrent reloading config
     */
    mutable std::recursive_mutex configMtx;

protected:
    /**
     * Time interval between each config file check
     */
    uint watcherIntervalMillisec = 1000;
    const int WRONG_CONFIG_FILE_RETRY_DELAY_MS = 10;

private:
    /**
     * Time interval between two consecutive sequence cleanup scans (in minutes)
     */
    uint32_t sequenceCleaupIntervalMinutes = 5;

    /**
     * Time interval between two consecutive resources cleanup scans (in seconds)
     */
    uint32_t resourcesCleanupIntervalSec = 1;

    /**
      * @brief last md5sum of configfile
      */
    std::string lastConfigFileMD5;

    /**
     * @brief Directory for OpenVINO to store cache files.
     */
    std::string modelCacheDirectory;

    MetricRegistry* metricRegistry;

#if (PYTHON_DISABLE == 0)
    PythonBackend* pythonBackend;
#endif

    /**
     * @brief Json config directory path
     *
     */
    std::string rootDirectoryPath;

    /**
     * @brief Set json config directory path
     *
     * @param configFileFullPath
     */
    void setRootDirectoryPath(const std::string& configFileFullPath);

public:
    /**
     * @brief Get the full path from relative or full path
     *
     * @return const std::string&
     */
    const std::string getFullPath(const std::string& pathToCheck) const;

    /**
     * @brief Get the config root path 
     *
     * @return const std::string&
     */
    const std::string getRootDirectoryPath() const {
        return rootDirectoryPath;
    }

    /**
     * @brief Mutex for blocking concurrent add & find of model
     */
    mutable std::shared_mutex modelsMtx;

    /**
     *  @brief Gets the watcher interval timestep in seconds
     */
    uint getWatcherIntervalMillisec() {
        return watcherIntervalMillisec;
    }

    /**
     *  @brief Gets the cleaner resources interval timestep in seconds
     */
    uint32_t getResourcesCleanupIntervalSec() {
        return resourcesCleanupIntervalSec;
    }

    /**
     *  @brief Adds new resource to watch by the cleaner thread
     */
    void addResourceToCleaner(std::shared_ptr<CNLIMWrapper> resource) {
        std::unique_lock resourcesLock(resourcesMtx);
        resources.emplace(resources.end(), std::move(resource));
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

    /**
     * @brief Starts monitoring cleanup as new thread
     */
    void startCleaner();

    const PipelineFactory& getPipelineFactory() const {
        return pipelineFactory;
    }

#if (MEDIAPIPE_DISABLE == 0)
    const MediapipeFactory& getMediapipeFactory() const {
        return mediapipeFactory;
    }
#endif

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
        std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) const;

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

    template <typename RequestType, typename ResponseType>
    Status createPipeline(std::unique_ptr<Pipeline>& pipeline,
        const std::string& name,
        const RequestType* request,
        ResponseType* response) {
        return pipelineFactory.create(pipeline, name, request, response, *this);
    }
    Status createPipeline(std::shared_ptr<MediapipeGraphExecutor>& graph,
        const std::string& name,
        const KFSRequest* request,
        KFSResponse* response);

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
         * @brief Get the metric config
         * 
         * @return const std::string&
         */
    const MetricConfig& getMetricConfig() const {
        return this->metricConfig;
    }

    Status loadMetricsConfig(rapidjson::Document& configJson);

    /**
         * @brief Set the metric config
         * 
         * @param metricConfig 
         */
    void setMetricConfig(const MetricConfig& metricConfig) {
        this->metricConfig = metricConfig;
    }

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
    Status start(const Config& config);

    /**
     * @brief Starts monitoring as new thread
     * 
     */
    void startWatcher(bool watchConfigFile);

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
        return std::make_shared<Model>(name, isStateful, &this->globalSequencesViewer);
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
    Status configFileReloadNeeded(bool& isNeeded);

    Status parseConfig(const std::string& jsonFilename, rapidjson::Document& configJson);

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
    Status updateConfigurationWithoutConfigFile();

    /**
     * @brief Cleaner thread procedure to cleanup resources that are not used
     */
    void cleanupResources();

    MetricRegistry* getMetricRegistry() const { return this->metricRegistry; }
};

void cleanerRoutine(uint32_t resourcesCleanupInterval, FunctorResourcesCleaner& functorResourcesCleaner, uint32_t sequenceCleanerInterval, FunctorSequenceCleaner& functorSequenceCleaner, std::future<void>& cleanerExitSignal);

}  // namespace ovms

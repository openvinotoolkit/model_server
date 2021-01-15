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
#include "modelmanager.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <dlfcn.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <sys/stat.h>

#include "azurefilesystem.hpp"
#include "config.hpp"
#include "customloaders.hpp"
#include "filesystem.hpp"
#include "gcsfilesystem.hpp"
#include "localfilesystem.hpp"
#include "logging.hpp"
#include "pipeline.hpp"
#include "pipeline_factory.hpp"
#include "s3filesystem.hpp"
#include "schema.hpp"

namespace ovms {

static bool watcherStarted = false;

Status ModelManager::start() {
    auto& config = ovms::Config::instance();
    watcherIntervalSec = config.filesystemPollWaitSeconds();
    Status status;
    if (config.configPath() != "") {
        status = startFromFile(config.configPath());
    } else {
        status = startFromConfig();
    }
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Couldn't start model manager");
        return status;
    }
    startWatcher();
    return status;
}

void ModelManager::startWatcher() {
    if ((!watcherStarted) && (watcherIntervalSec > 0)) {
        std::future<void> exitSignal = exit.get_future();
        std::thread t(std::thread(&ModelManager::watcher, this, std::move(exitSignal)));
        watcherStarted = true;
        monitor = std::move(t);
    }
}

Status ModelManager::startFromConfig() {
    auto& config = ovms::Config::instance();

    auto [it, success] = servedModelConfigs.emplace(
        config.modelName(),
        ModelConfig{
            config.modelName(),
            config.modelPath(),
            config.targetDevice(),
            config.batchSize(),
            config.nireq(),
            config.stateful(),
            config.lowLatencyTransformation(),
            config.sequenceTimeoutSeconds(),
            config.maxSequenceNumber()});

    if (!success) {
        return StatusCode::UNKNOWN_ERROR;
    }

    ModelConfig& modelConfig = it->second;

    auto status = modelConfig.parsePluginConfig(config.pluginConfig());
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Couldn't parse plugin config");
        return status;
    }

    status = modelConfig.parseModelVersionPolicy(config.modelVersionPolicy());
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Couldn't parse model version policy. {}", status.string());
        return status;
    }

    status = modelConfig.parseShapeParameter(config.shape());
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Couldn't parse shape parameter");
        return status;
    }

    bool batchSizeSet = (modelConfig.getBatchingMode() != FIXED || modelConfig.getBatchSize() != 0);
    bool shapeSet = (modelConfig.getShapes().size() > 0);

    SPDLOG_DEBUG("Batch size set: {}, shape set: {}", batchSizeSet, shapeSet);
    if (batchSizeSet && shapeSet) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Both shape and batch size have been defined. Batch size parameter will be ignored.");
        modelConfig.setBatchingMode(FIXED);
        modelConfig.setBatchSize(0);
    }

    return reloadModelWithVersions(modelConfig);
}

Status ModelManager::startFromFile(const std::string& jsonFilename) {
    Status status = loadConfig(jsonFilename);
    if (!status.ok()) {
        return status;
    }

    return StatusCode::OK;
}

void processNodeInputs(const std::string nodeName, const rapidjson::Value::ConstMemberIterator& itro, pipeline_connections_t& connections) {
    for (const auto& nodeInput : itro->value.GetArray()) {
        for (const auto& objectNameValue : nodeInput.GetObject()) {
            const std::string inputName = objectNameValue.name.GetString();
            const std::string sourceNodeName = objectNameValue.value.GetObject()["node_name"].GetString();
            const std::string sourceOutputName = objectNameValue.value.GetObject()["data_item"].GetString();
            SPDLOG_DEBUG("Creating node dependencies mapping request. Node: {} input: {} <- SourceNode: {} output: {}",
                nodeName, inputName, sourceNodeName, sourceOutputName);
            if (connections.find(nodeName) == connections.end()) {
                connections[nodeName] = {
                    {sourceNodeName,
                        {{sourceOutputName, inputName}}}};
            } else {
                if (connections[nodeName].find(sourceNodeName) == connections[nodeName].end()) {
                    connections[nodeName].insert({sourceNodeName,
                        {{sourceOutputName, inputName}}});
                } else {
                    connections[nodeName][sourceNodeName].push_back({sourceOutputName, inputName});
                }
            }
        }
    }
}

void processPipelineInputs(const rapidjson::Value::ConstMemberIterator& pipelineInputsPtr, const std::string& nodeName, std::unordered_map<std::string, std::string>& nodeOutputNameAlias, const std::string& pipelineName) {
    for (const auto& pipelineInput : pipelineInputsPtr->value.GetArray()) {
        const std::string pipelineInputName = pipelineInput.GetString();
        SPDLOG_DEBUG("Mapping node:{} output:{}, under alias:{}",
            nodeName, pipelineInputName, pipelineInputName);
        auto result = nodeOutputNameAlias.insert({pipelineInputName, pipelineInputName});
        if (!result.second) {
            SPDLOG_ERROR("Pipeline {} has duplicated input declaration", pipelineName);
        }
    }
}

void processNodeOutputs(const rapidjson::Value::ConstMemberIterator& nodeOutputsItr, const std::string& nodeName, const std::string& modelName, std::unordered_map<std::string, std::string>& nodeOutputNameAlias) {
    for (const auto& nodeOutput : nodeOutputsItr->value.GetArray()) {
        const std::string modelOutputName = nodeOutput.GetObject()["data_item"].GetString();
        const std::string nodeOutputName = nodeOutput.GetObject()["alias"].GetString();
        SPDLOG_DEBUG("Mapping node: {} model_name: {} output: {}, under alias: {}",
            nodeName, modelName, modelOutputName, nodeOutputName);
        nodeOutputNameAlias[nodeOutputName] = modelOutputName;
    }
}

void processPipelineConfig(rapidjson::Document& configJson, const rapidjson::Value& pipelineConfig, std::set<std::string>& pipelinesInConfigFile, PipelineFactory& factory, ModelManager& manager) {
    const std::string pipelineName = pipelineConfig["name"].GetString();
    if (pipelinesInConfigFile.find(pipelineName) != pipelinesInConfigFile.end()) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Duplicated pipeline names: {} defined in config file. Only first definition will be loaded.", pipelineName);
        return;
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Reading pipeline: {} configuration", pipelineName);
    auto itr2 = pipelineConfig.FindMember("nodes");

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, ENTRY_NODE_NAME}};
    processPipelineInputs(pipelineConfig.FindMember("inputs"), ENTRY_NODE_NAME, info[0].outputNameAliases, pipelineName);
    pipeline_connections_t connections;
    for (const auto& nodeConfig : itr2->value.GetArray()) {
        std::string nodeName;
        nodeName = nodeConfig["name"].GetString();

        std::string modelName;
        modelName = nodeConfig["model_name"].GetString();

        const std::string nodeKindStr = nodeConfig["type"].GetString();
        auto nodeOutputsItr = nodeConfig.FindMember("outputs");
        if (nodeOutputsItr == nodeConfig.MemberEnd() || !nodeOutputsItr->value.IsArray()) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Pipeline: {} does not have valid outputs configuration", pipelineName);
            return;
        }
        std::unordered_map<std::string, std::string> nodeOutputNameAlias;  // key:alias, value realName
        processNodeOutputs(nodeOutputsItr, nodeName, modelName, nodeOutputNameAlias);
        std::optional<model_version_t> modelVersion;
        if (nodeConfig.HasMember("version")) {
            modelVersion = nodeConfig["version"].GetUint64();
        } else {
            modelVersion = std::nullopt;
        }
        NodeKind nodeKind;
        auto status = toNodeKind(nodeKindStr, nodeKind);
        if (!status.ok()) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Parsing node kind failed: {}", nodeKindStr);
            return;
        }
        SPDLOG_DEBUG("Creating node: {} type: {} model_name: {} modelVersion: {}",
            nodeName, nodeKindStr, modelName, modelVersion.value_or(0));
        info.emplace_back(std::move(NodeInfo{nodeKind, nodeName, modelName, modelVersion, nodeOutputNameAlias}));
        auto nodeInputItr = nodeConfig.FindMember("inputs");
        processNodeInputs(nodeName, nodeInputItr, connections);
    }
    const auto iteratorOutputs = pipelineConfig.FindMember("outputs");
    // pipeline outputs are node exit inputs
    processNodeInputs(EXIT_NODE_NAME, iteratorOutputs, connections);
    info.emplace_back(std::move(NodeInfo(NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {})));
    if (!factory.definitionExists(pipelineName)) {
        SPDLOG_DEBUG("Pipeline:{} was not loaded so far. Triggering load", pipelineName);
        auto status = factory.createDefinition(pipelineName, info, connections, manager);
        pipelinesInConfigFile.insert(pipelineName);
        return;
    }
    SPDLOG_DEBUG("Pipeline:{} is already loaded. Triggering reload", pipelineName);
    auto status = factory.reloadDefinition(pipelineName,
        std::move(info),
        std::move(connections),
        manager);
    pipelinesInConfigFile.insert(pipelineName);
}

Status ModelManager::loadPipelinesConfig(rapidjson::Document& configJson) {
    const auto itrp = configJson.FindMember("pipeline_config_list");
    if (itrp == configJson.MemberEnd() || !itrp->value.IsArray()) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Configuration file doesn't have pipelines property.");
        pipelineFactory.retireOtherThan({}, *this);
        return StatusCode::OK;
    }
    std::set<std::string> pipelinesInConfigFile;
    for (const auto& pipelineConfig : itrp->value.GetArray()) {
        processPipelineConfig(configJson, pipelineConfig, pipelinesInConfigFile, pipelineFactory, *this);
    }
    pipelineFactory.retireOtherThan(std::move(pipelinesInConfigFile), *this);
    return ovms::StatusCode::OK;
}

Status ModelManager::createCustomLoader(CustomLoaderConfig& loaderConfig) {
    auto& customloaders = ovms::CustomLoaders::instance();
    std::string loaderName = loaderConfig.getLoaderName();
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Check if loader is already loaded");
    if (customloaders.find(loaderName) == nullptr) {
        // this is where library or custom loader is loaded
        void* handleCL = dlopen(const_cast<char*>(loaderConfig.getLibraryPath().c_str()), RTLD_LAZY | RTLD_LOCAL);
        if (!handleCL) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot open library:  {} {}", loaderConfig.getLibraryPath(), dlerror());
            return StatusCode::CUSTOM_LOADER_LIBRARY_INVALID;
        }

        // load the symbols
        createCustomLoader_t* customObj = (createCustomLoader_t*)dlsym(handleCL, "createCustomLoader");
        const char* dlsym_error = dlerror();
        if (dlsym_error || (customObj == nullptr)) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot load symbol create:  {} ", dlsym_error);
            return StatusCode::CUSTOM_LOADER_LIBRARY_LOAD_FAILED;
        }

        std::shared_ptr<CustomLoaderInterface> customLoaderIfPtr{customObj()};
        try {
            customLoaderIfPtr->loaderInit(loaderConfig.getLoaderConfigFile());
        } catch (std::exception& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot create or initialize the custom loader. Failed with error {}", e.what());
            return StatusCode::CUSTOM_LOADER_INIT_FAILED;
        } catch (...) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Cannot create or initialize the custom loader");
            return StatusCode::CUSTOM_LOADER_INIT_FAILED;
        }
        customloaders.add(loaderName, customLoaderIfPtr, handleCL);
    } else {
        // Loader is already in the existing loaders. Move it to new loaders.
        // Reload of customloader is not supported yet
        customloaders.move(loaderName);
    }
    return StatusCode::OK;
}

Status ModelManager::loadCustomLoadersConfig(rapidjson::Document& configJson) {
    const auto itrp = configJson.FindMember("custom_loader_config_list");
    if (itrp == configJson.MemberEnd() || !itrp->value.IsArray()) {
        return StatusCode::OK;
    }

    // Load Customer Loaders as per the configuration
    SPDLOG_DEBUG("Using Customloader");
    for (const auto& configs : itrp->value.GetArray()) {
        const std::string loaderName = configs["config"]["loader_name"].GetString();
        SPDLOG_INFO("Reading Custom Loader: {} configuration", loaderName);

        CustomLoaderConfig loaderConfig;
        auto status = loaderConfig.parseNode(configs["config"]);
        if (status != StatusCode::OK) {
            SPDLOG_ERROR("Parsing loader: {} config failed", loaderName);
            return status;
        }

        auto retVal = createCustomLoader(loaderConfig);
        if (retVal != StatusCode::OK) {
            SPDLOG_ERROR("Creation of loader: {} failed", loaderName);
        }
    }
    // All loaders are the done. Finalize the list by deleting removed loaders in config
    auto& customloaders = ovms::CustomLoaders::instance();
    customloaders.finalize();
    return ovms::StatusCode::OK;
}

Status ModelManager::loadModelsConfig(rapidjson::Document& configJson, std::vector<ModelConfig>& gatedModelConfigs) {
    const auto itr = configJson.FindMember("model_config_list");
    if (itr == configJson.MemberEnd() || !itr->value.IsArray()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Configuration file doesn't have models property.");
        return StatusCode::JSON_INVALID;
    }
    std::set<std::string> modelsInConfigFile;
    std::unordered_map<std::string, ModelConfig> newModelConfigs;
    for (const auto& configs : itr->value.GetArray()) {
        ModelConfig modelConfig;
        auto status = modelConfig.parseNode(configs["config"]);
        const auto modelName = modelConfig.getName();
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Parsing model: {} config failed", modelName);
            continue;
        }
        if (pipelineDefinitionExists(modelName)) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Model name: {} is already occupied by pipeline definition.", modelName);
            continue;
        }
        if (modelsInConfigFile.find(modelName) != modelsInConfigFile.end()) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Duplicated model names: {} defined in config file. Only first definition will be loaded.", modelName);
            continue;
        }
        status = reloadModelWithVersions(modelConfig);
        modelsInConfigFile.emplace(modelName);

        if (!status.ok()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Cannot reload model: {} with versions due to error: {}", modelName, status.string());
        }
        if (status != StatusCode::REQUESTED_DYNAMIC_PARAMETERS_ON_SUBSCRIBED_MODEL) {
            newModelConfigs.emplace(modelName, std::move(modelConfig));
        } else {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Will retry to reload model({}) after pipelines are revalidated", modelName);
            auto it = this->servedModelConfigs.find(modelName);
            if (it == this->servedModelConfigs.end()) {
                continue;
            }
            gatedModelConfigs.emplace_back(std::move(modelConfig));
            newModelConfigs.emplace(modelName, std::move(it->second));
            this->servedModelConfigs.erase(modelName);
        }
    }
    this->servedModelConfigs = std::move(newModelConfigs);
    retireModelsRemovedFromConfigFile(modelsInConfigFile);
    return ovms::StatusCode::OK;
}

Status ModelManager::tryReloadGatedModelConfigs(std::vector<ModelConfig>& gatedModelConfigs) {
    for (auto& modelConfig : gatedModelConfigs) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Trying to reload model({}) configuration", modelConfig.getName());
        auto status = reloadModelWithVersions(modelConfig);
        if (!status.ok()) {
            continue;
        }
        auto it = this->servedModelConfigs.find(modelConfig.getName());
        if (it == this->servedModelConfigs.end()) {
            continue;
        }
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Successfully retried to load new model({}) configuration after unsubscribed from pipeline", modelConfig.getName());
        this->servedModelConfigs.at(modelConfig.getName()) = std::move(modelConfig);
    }
    return StatusCode::OK;
}

Status ModelManager::loadConfig(const std::string& jsonFilename) {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Loading configuration from {}", jsonFilename);
    std::ifstream ifs(jsonFilename.c_str());
    if (!ifs.good()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "File is invalid {}", jsonFilename);
        return StatusCode::FILE_INVALID;
    }
    rapidjson::Document configJson;
    rapidjson::IStreamWrapper isw(ifs);
    if (configJson.ParseStream(isw).HasParseError()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Configuration file is not a valid JSON file.");
        return StatusCode::JSON_INVALID;
    }

    if (validateJsonAgainstSchema(configJson, MODELS_CONFIG_SCHEMA) != StatusCode::OK) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Configuration file is not in valid configuration format");
        return StatusCode::JSON_INVALID;
    }
    configFilename = jsonFilename;
    Status status;
    // load the custom loader config, if available
    status = loadCustomLoadersConfig(configJson);
    if (status != StatusCode::OK) {
        return status;
    }
    std::vector<ModelConfig> gatedModelConfigs;
    status = loadModelsConfig(configJson, gatedModelConfigs);
    if (status != StatusCode::OK) {
        return status;
    }
    status = loadPipelinesConfig(configJson);
    tryReloadGatedModelConfigs(gatedModelConfigs);
    return StatusCode::OK;
}

void ModelManager::retireModelsRemovedFromConfigFile(const std::set<std::string>& modelsExistingInConfigFile) {
    std::set<std::string> modelsCurrentlyLoaded;
    for (auto& nameModelPair : getModels()) {
        modelsCurrentlyLoaded.insert(nameModelPair.first);
    }
    std::vector<std::string> modelsToUnloadAllVersions(getModels().size());
    auto it = std::set_difference(
        modelsCurrentlyLoaded.begin(), modelsCurrentlyLoaded.end(),
        modelsExistingInConfigFile.begin(), modelsExistingInConfigFile.end(),
        modelsToUnloadAllVersions.begin());
    modelsToUnloadAllVersions.resize(it - modelsToUnloadAllVersions.begin());
    for (auto& modelName : modelsToUnloadAllVersions) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Retiring all versions of model: {}", modelName);
        try {
            models.at(modelName)->retireAllVersions();
        } catch (const std::out_of_range& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Unknown error occurred when tried to retire all versions of model: {}", modelName);
        }
    }
}

void ModelManager::updateConfigurationWithoutConfigFile() {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Checking if something changed with model versions");
    for (auto& [name, config] : servedModelConfigs) {
        reloadModelWithVersions(config);
    }
    pipelineFactory.revalidatePipelines(*this);
}

void ModelManager::watcher(std::future<void> exit) {
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Started config watcher thread");
    int64_t lastTime;
    struct stat statTime;
    stat(configFilename.c_str(), &statTime);
    lastTime = statTime.st_ctime;
    while (exit.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
        std::this_thread::sleep_for(std::chrono::seconds(watcherIntervalSec));
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Watcher thread check cycle begin");
        stat(configFilename.c_str(), &statTime);
        if (lastTime != statTime.st_ctime) {
            lastTime = statTime.st_ctime;
            loadConfig(configFilename);
        }
        updateConfigurationWithoutConfigFile();
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Watcher thread check cycle end");
    }
    SPDLOG_LOGGER_ERROR(modelmanager_logger, "Exited config watcher thread");
}

void ModelManager::join() {
    if (watcherStarted) {
        exit.set_value();
        if (monitor.joinable()) {
            monitor.join();
            watcherStarted = false;
        }
    }
}

void ModelManager::getVersionsToChange(
    const ModelConfig& newModelConfig,
    const std::map<model_version_t, std::shared_ptr<ModelInstance>>& modelVersionsInstances,
    model_versions_t requestedVersions,
    std::shared_ptr<model_versions_t>& versionsToStartIn,
    std::shared_ptr<model_versions_t>& versionsToReloadIn,
    std::shared_ptr<model_versions_t>& versionsToRetireIn) {
    std::sort(requestedVersions.begin(), requestedVersions.end());
    model_versions_t registeredModelVersions;
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Currently registered model: {} versions count: {}", newModelConfig.getName(), modelVersionsInstances.size());
    for (const auto& [version, versionInstance] : modelVersionsInstances) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "model: {} version: {} state: {}", newModelConfig.getName(), version, ovms::ModelVersionStateToString(versionInstance->getStatus().getState()));
        registeredModelVersions.push_back(version);
    }

    model_versions_t alreadyRegisteredVersionsWhichAreRequested(requestedVersions.size());
    model_versions_t::iterator it = std::set_intersection(
        requestedVersions.begin(), requestedVersions.end(),
        registeredModelVersions.begin(), registeredModelVersions.end(),
        alreadyRegisteredVersionsWhichAreRequested.begin());
    alreadyRegisteredVersionsWhichAreRequested.resize(it - alreadyRegisteredVersionsWhichAreRequested.begin());

    std::shared_ptr<model_versions_t> versionsToReload = std::make_shared<model_versions_t>();
    for (const auto& version : alreadyRegisteredVersionsWhichAreRequested) {
        try {
            if (modelVersionsInstances.at(version)->getStatus().willEndUnloaded() ||
                modelVersionsInstances.at(version)->getModelConfig().isReloadRequired(newModelConfig)) {
                versionsToReload->push_back(version);
            }
        } catch (std::out_of_range& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Data race occured during versions update. Could not found version. Details: {}", e.what());
        }
    }

    std::shared_ptr<model_versions_t> versionsToRetire = std::make_shared<model_versions_t>(registeredModelVersions.size());
    it = std::set_difference(
        registeredModelVersions.begin(), registeredModelVersions.end(),
        requestedVersions.begin(), requestedVersions.end(),
        versionsToRetire->begin());
    versionsToRetire->resize(it - versionsToRetire->begin());
    try {
        it = std::remove_if(versionsToRetire->begin(),
            versionsToRetire->end(),
            [&modelVersionsInstances](model_version_t version) {
                return modelVersionsInstances.at(version)->getStatus().willEndUnloaded();
            });
    } catch (std::out_of_range& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Data race occured during versions update. Could not found version. Details: {}", e.what());
    }
    versionsToRetire->resize(it - versionsToRetire->begin());

    std::shared_ptr<model_versions_t> versionsToStart = std::make_shared<model_versions_t>(requestedVersions.size());
    it = std::set_difference(
        requestedVersions.begin(), requestedVersions.end(),
        registeredModelVersions.begin(), registeredModelVersions.end(),
        versionsToStart->begin());
    versionsToStart->resize(it - versionsToStart->begin());

    versionsToStartIn = std::move(versionsToStart);
    versionsToReloadIn = std::move(versionsToReload);
    versionsToRetireIn = std::move(versionsToRetire);
}

std::shared_ptr<ovms::Model> ModelManager::getModelIfExistCreateElse(const std::string& modelName, const bool isStateful) {
    std::unique_lock modelsLock(modelsMtx);
    auto modelIt = models.find(modelName);
    if (models.end() == modelIt) {
        models.insert({modelName, modelFactory(modelName, isStateful)});
    }
    return models[modelName];
}

std::shared_ptr<FileSystem> ModelManager::getFilesystem(const std::string& basePath) {
    if (basePath.rfind(S3FileSystem::S3_URL_PREFIX, 0) == 0) {
        Aws::SDKOptions options;
        Aws::InitAPI(options);
        return std::make_shared<S3FileSystem>(options, basePath);
    }
    if (basePath.rfind(GCSFileSystem::GCS_URL_PREFIX, 0) == 0) {
        return std::make_shared<ovms::GCSFileSystem>();
    }
    if (basePath.rfind(AzureFileSystem::AZURE_URL_FILE_PREFIX, 0) == 0) {
        return std::make_shared<ovms::AzureFileSystem>();
    }
    if (basePath.rfind(AzureFileSystem::AZURE_URL_BLOB_PREFIX, 0) == 0) {
        return std::make_shared<ovms::AzureFileSystem>();
    }
    return std::make_shared<LocalFileSystem>();
}

Status ModelManager::readAvailableVersions(std::shared_ptr<FileSystem>& fs, const std::string& base, model_versions_t& versions) {
    files_list_t dirs;

    bool is_directory = false;
    if (FileSystem::isPathEscaped(base)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Path {} escape with .. is forbidden.", base);
        return StatusCode::PATH_INVALID;
    }

    auto status = fs->isDirectory(base, &is_directory);
    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Couldn't check directory: {}", base);
        return status;
    }
    if (!is_directory) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Directory does not exist: {}", base);
        return StatusCode::PATH_INVALID;
    }

    status = fs->getDirectorySubdirs(base, &dirs);

    if (status != StatusCode::OK) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Couldn't list directories in path: {}", base);
        return status;
    }

    for (const auto& entry : dirs) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Detected version folder: {}", entry);
        try {
            ovms::model_version_t version = std::stoll(entry);
            if (version <= 0) {
                SPDLOG_LOGGER_WARN(modelmanager_logger, "Expected version directory name to be a number greater than 0. Got: {}", version);
                continue;
            }
            versions.push_back(version);
        } catch (const std::invalid_argument& e) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Expected version directory name to be in number format. Got: {}", entry);
        } catch (const std::out_of_range& e) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Directory name is out of range for supported version format. Got: {}", entry);
        }
    }

    if (0 == versions.size()) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "No version found for model in path: {}", base);
        return StatusCode::NO_MODEL_VERSION_AVAILABLE;
    }

    return StatusCode::OK;
}

Status ModelManager::addModelVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<FileSystem>& fs, ModelConfig& config, std::shared_ptr<model_versions_t>& versionsToStart, std::shared_ptr<model_versions_t> versionsFailed) {
    Status status = StatusCode::OK;
    try {
        status = model->addVersions(versionsToStart, config, fs, versionsFailed);
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error occurred while loading model: {} versions; error: {}",
                config.getName(),
                status.string());
        }
    } catch (std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Exception occurred while loading model: {};", e.what());
    }
    return status;
}

Status ModelManager::reloadModelVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<FileSystem>& fs, ModelConfig& config, std::shared_ptr<model_versions_t>& versionsToReload, std::shared_ptr<model_versions_t> versionsFailed) {
    Status status = StatusCode::OK;
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Reloading model versions");
    try {
        auto status = model->reloadVersions(versionsToReload, config, fs, versionsFailed);
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error occurred while reloading model: {}; versions; error: {}",
                config.getName(),
                status.string());
        }
    } catch (std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Exception occurred while reloading model: {};", e.what());
    }

    return status;
}

Status ModelManager::reloadModelWithVersions(ModelConfig& config) {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Started applying config changes to model: {}", config.getName());
    auto model = getModelIfExistCreateElse(config.getName(), config.isStateful());
    if (model->isAnyVersionSubscribed() && config.isDynamicParameterEnabled()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Requested setting dynamic parameters for model {} but it is used in pipeline. Cannot reload model configuration.", config.getName());
        return StatusCode::REQUESTED_DYNAMIC_PARAMETERS_ON_SUBSCRIBED_MODEL;
    }

    auto fs = ModelManager::getFilesystem(config.getBasePath());
    std::vector<model_version_t> availableVersions;
    Status blocking_status = readAvailableVersions(fs, config.getBasePath(), availableVersions);
    if (!blocking_status.ok()) {
        return blocking_status;
    }
    auto requestedVersions = config.getModelVersionPolicy()->filter(availableVersions);
    std::shared_ptr<model_versions_t> versionsToStart;
    std::shared_ptr<model_versions_t> versionsToReload;
    std::shared_ptr<model_versions_t> versionsToRetire;
    std::shared_ptr<model_versions_t> versionsFailed = std::make_shared<model_versions_t>();
    // first reset custom loader name to empty string so that any changes to name can be captured
    model->resetCustomLoaderName();

    if (config.isCustomLoaderRequiredToLoadModel()) {
        custom_loader_options_config_t customLoaderOptionsConfig = config.getCustomLoaderOptionsConfigMap();
        const std::string loaderName = customLoaderOptionsConfig["loader_name"];

        auto& customloaders = ovms::CustomLoaders::instance();
        auto loaderPtr = customloaders.find(loaderName);
        if (loaderPtr != nullptr) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Custom Loader to be used : {}", loaderName);
            model->setCustomLoaderName(loaderName);

            // check existing version for blacklist
            for (const auto& [version, versionInstance] : model->getModelVersions()) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "The model {} checking for blacklist", versionInstance->getName());
                CustomLoaderStatus bres = loaderPtr->getModelBlacklistStatus(versionInstance->getName(), version);
                if (bres != CustomLoaderStatus::OK) {
                    SPDLOG_LOGGER_INFO(modelmanager_logger, "The model {} is blacklisted", versionInstance->getName());
                    requestedVersions.erase(std::remove(requestedVersions.begin(), requestedVersions.end(), version), requestedVersions.end());
                }
            }
        } else {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Specified custom loader {} not found. In case any models are loaded, will be unloading them", loaderName);
            model->retireAllVersions();
            return StatusCode::OK;
        }
    }

    getVersionsToChange(config, model->getModelVersions(), requestedVersions, versionsToStart, versionsToReload, versionsToRetire);

    while (versionsToStart->size() > 0) {
        blocking_status = addModelVersions(model, fs, config, versionsToStart, versionsFailed);
        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Adding new versions. Status: {};", blocking_status.string());
        if (!blocking_status.ok()) {
            for (const auto version : *versionsFailed) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Removing available version {} due to load failure; ", version);
                if (std::binary_search(availableVersions.begin(), availableVersions.end(), version)) {
                    availableVersions.erase(std::remove(availableVersions.begin(), availableVersions.end(), version), availableVersions.end());
                }
            }
            requestedVersions = config.getModelVersionPolicy()->filter(availableVersions);
            getVersionsToChange(config, model->getModelVersions(), requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
        } else {
            break;
        }
    }

    if (versionsToReload->size() > 0) {
        reloadModelVersions(model, fs, config, versionsToReload, versionsFailed);
    }
    for (const auto version : *versionsFailed) {
        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Removing available version {} due to load failure.", version);
        if (std::binary_search(availableVersions.begin(), availableVersions.end(), version)) {
            availableVersions.erase(std::remove(availableVersions.begin(), availableVersions.end(), version), availableVersions.end());
        }
    }
    // refresh versions to retire based on failed reloads
    requestedVersions = config.getModelVersionPolicy()->filter(availableVersions);
    getVersionsToChange(config, model->getModelVersions(), requestedVersions, versionsToStart, versionsToReload, versionsToRetire);

    if (versionsToRetire->size() > 0) {
        auto status = model->retireVersions(versionsToRetire);
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error occurred while unloading model: {}; versions; error: {}",
                config.getName(),
                status.string());
        }
    }
    return blocking_status;
}

const std::shared_ptr<Model> ModelManager::findModelByName(const std::string& name) const {
    std::shared_lock lock(modelsMtx);
    auto it = models.find(name);
    return it != models.end() ? it->second : nullptr;
}

Status ModelManager::getModelInstance(const std::string& modelName,
    ovms::model_version_t modelVersionId,
    std::shared_ptr<ovms::ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) {
    SPDLOG_DEBUG("Requesting model: {}; version: {}.", modelName, modelVersionId);

    auto model = findModelByName(modelName);
    if (model == nullptr) {
        return StatusCode::MODEL_NAME_MISSING;
    }
    if (modelVersionId != 0) {
        modelInstance = model->getModelInstanceByVersion(modelVersionId);
        if (modelInstance == nullptr) {
            return StatusCode::MODEL_VERSION_MISSING;
        }
    } else {
        modelInstance = model->getDefaultModelInstance();
        if (modelInstance == nullptr) {
            return StatusCode::MODEL_VERSION_MISSING;
        }
    }

    return modelInstance->waitForLoaded(WAIT_FOR_MODEL_LOADED_TIMEOUT_MS, modelInstanceUnloadGuardPtr);
}

Status ModelManager::getPipeline(std::unique_ptr<ovms::Pipeline>& pipelinePtr,
    const tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response) {

    SPDLOG_DEBUG("Requesting pipeline: {};", request->model_spec().name());
    auto status = createPipeline(pipelinePtr, request->model_spec().name(), request, response);
    return status;
}

}  // namespace ovms

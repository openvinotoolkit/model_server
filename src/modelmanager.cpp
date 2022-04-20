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
#include <errno.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <sys/stat.h>
#include <unistd.h>

#include "azurefilesystem.hpp"
#include "cleaner_utils.hpp"
#include "config.hpp"
#include "custom_node_library_manager.hpp"
#include "customloaders.hpp"
#include "entry_node.hpp"  // need for ENTRY_NODE_NAME
#include "exit_node.hpp"   // need for EXIT_NODE_NAME
#include "filesystem.hpp"
#include "gcsfilesystem.hpp"
#include "localfilesystem.hpp"
#include "logging.hpp"
#include "node_library.hpp"
#include "openssl/md5.h"
#include "pipeline.hpp"
#include "pipeline_factory.hpp"
#include "pipelinedefinition.hpp"
#include "s3filesystem.hpp"
#include "schema.hpp"
#include "stringutils.hpp"
#include "ov_utils.hpp"

namespace ovms {

static uint16_t MAX_CONFIG_JSON_READ_RETRY_COUNT = 2;
static bool watcherStarted = false;
static bool cleanerStarted = false;

ModelManager::ModelManager(const std::string& modelCacheDirectory) :
    ieCore(std::make_unique<ov::Core>()),
    waitForModelLoadedTimeoutMs(DEFAULT_WAIT_FOR_MODEL_LOADED_TIMEOUT_MS),
    modelCacheDirectory(modelCacheDirectory) {
    // Take --cache_dir from CLI
    if (this->modelCacheDirectory.empty()) {
        this->modelCacheDirectory = ovms::Config::instance().cacheDir();
    }
    // If not enabled via CLI, check for /opt/cache existence.
    if (this->modelCacheDirectory.empty()) {
        if (std::filesystem::exists(DEFAULT_MODEL_CACHE_DIRECTORY)) {
            this->modelCacheDirectory = DEFAULT_MODEL_CACHE_DIRECTORY;
        }
    }
    // If cache dir enabled, check for write access.
    if (!this->modelCacheDirectory.empty()) {
        // Create directory if does not exist
        if (!std::filesystem::exists(this->modelCacheDirectory)) {
            std::filesystem::create_directories(this->modelCacheDirectory);
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Cache directory {} did not exist, created", this->modelCacheDirectory);
        }
        int result = access(this->modelCacheDirectory.c_str(), W_OK);
        if (result != 0) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Cache directory {} is not writable; access() result: {}", this->modelCacheDirectory, result);
        } else {
            SPDLOG_LOGGER_INFO(modelmanager_logger, "Model cache is enabled: {}", this->modelCacheDirectory);
        }
    }
    this->customNodeLibraryManager = std::make_unique<CustomNodeLibraryManager>();
    if (ovms::Config::instance().cpuExtensionLibraryPath() != "") {
        SPDLOG_INFO("Loading custom CPU extension from {}", ovms::Config::instance().cpuExtensionLibraryPath());
        try {
            ieCore->add_extension(ovms::Config::instance().cpuExtensionLibraryPath());
            SPDLOG_INFO("Extension added.");
        } catch (std::exception& ex) {
            SPDLOG_CRITICAL("Custom CPU extension loading has failed! Reason: {}", ex.what());
            throw;
        } catch (...) {
            SPDLOG_CRITICAL("Custom CPU extension loading has failed with an unknown error!");
            throw;
        }
    }
    this->logPluginConfiguration();
}

void ModelManager::logPluginConfiguration() {
    auto availableDevices = ieCore->get_available_devices();
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Available devices for Open VINO: {}", joins(availableDevices, std::string(", ")));
    auto availablePlugins = availableDevices;
    const std::string supportedConfigKey = METRIC_KEY(SUPPORTED_CONFIG_KEYS);
    for (const auto& plugin : availablePlugins) {
        std::vector<std::string> supportedConfigKeys;
        try {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Logging plugin: {}; configuration", plugin);
            std::vector<std::string> supportedConfigKeys2 = ieCore->get_property(plugin, supportedConfigKey).as<std::vector<std::string>>();
            supportedConfigKeys = std::move(supportedConfigKeys2);
        } catch (std::exception& e) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting plugin: {}; key: {}; value. Error: {}", plugin, supportedConfigKey, e.what());
        } catch (...) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting plugin: {}; key: {}; value.", plugin, supportedConfigKey);
        }
        for (auto& key : supportedConfigKeys) {
            std::string value;
            try {
                auto paramValue = ieCore->get_property(plugin, key);
                value = paramValue.as<std::string>();
            } catch (std::exception& e) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting plugin: {}; config key: {}; Error: {}", plugin, key, e.what());
                continue;
            } catch (...) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Exception thrown from IE when requesting plugin: {}; config key: {}", plugin, key);
                continue;
            }
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Global plugin: {}; key: {}; value: {}", plugin, key, value);
        }
    }
}

ModelManager::~ModelManager() = default;

Status ModelManager::start() {
    auto& config = ovms::Config::instance();
    watcherIntervalSec = config.filesystemPollWaitSeconds();
    sequenceCleaupIntervalMinutes = config.sequenceCleanerPollWaitMinutes();
    resourcesCleanupIntervalSec = config.resourcesCleanerPollWaitSeconds();
    if (resourcesCleanupIntervalSec < 1) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Parameter: custom_node_resources_cleaner_interval has to be greater than 0. Applying default value(1 second)");
        resourcesCleanupIntervalSec = 1;
    }
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
    startCleaner();
    return status;
}

void ModelManager::startWatcher() {
    if ((!watcherStarted) && (watcherIntervalSec > 0)) {
        std::future<void> exitSignal = exitTrigger.get_future();
        std::thread t(std::thread(&ModelManager::watcher, this, std::move(exitSignal)));
        watcherStarted = true;
        monitor = std::move(t);
    }
}

void ModelManager::startCleaner() {
    if ((!cleanerStarted)) {
        std::future<void> exitSignal = cleanerExitTrigger.get_future();
        std::thread t(std::thread(&ModelManager::cleanerRoutine, this, resourcesCleanupIntervalSec, sequenceCleaupIntervalMinutes, std::move(exitSignal)));
        cleanerStarted = true;
        cleanerThread = std::move(t);
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
            config.idleSequenceCleanup(),
            config.lowLatencyTransformation(),
            config.maxSequenceNumber(),
            this->modelCacheDirectory});

    if (!success) {
        return StatusCode::UNKNOWN_ERROR;
    }

    ModelConfig& modelConfig = it->second;

    auto status = modelConfig.parsePluginConfig(config.pluginConfig());
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Couldn't parse plugin config");
        return status;
    }

    status = validatePluginConfiguration(modelConfig.getPluginConfig(), modelConfig.getTargetDevice(), *ieCore.get());
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Plugin config contains unsupported keys");
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

    status = modelConfig.parseLayoutParameter(config.layout());
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Couldn't parse layout parameter");
        return status;
    }

    bool batchSizeSet = (modelConfig.getBatchingMode() != FIXED || modelConfig.getBatchSize() != 0);
    bool shapeSet = (modelConfig.getShapes().size() > 0);

    SPDLOG_DEBUG("Batch size set: {}, shape set: {}", batchSizeSet, shapeSet);
    if (batchSizeSet && shapeSet) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Both shape and batch size have been defined. Batch size parameter will be ignored.");
        modelConfig.setBatchingMode(FIXED);
        modelConfig.setBatchSize(std::nullopt);
    }

    return reloadModelWithVersions(modelConfig);
}

Status ModelManager::startFromFile(const std::string& jsonFilename) {
    Status status = loadConfig(jsonFilename);
    if (status == StatusCode::CONFIG_FILE_INVALID || status == StatusCode::JSON_INVALID) {
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

void processDLNodeConfig(const rapidjson::Value& nodeConfig, DLNodeInfo& info) {
    info.modelName = nodeConfig["model_name"].GetString();
    if (nodeConfig.HasMember("version")) {
        info.modelVersion = nodeConfig["version"].GetUint64();
    }
}

#define IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status) \
    if (firstErrorStatus.ok()) {                                   \
        firstErrorStatus = status;                                 \
    }

Status processCustomNodeConfig(const rapidjson::Value& nodeConfig, CustomNodeInfo& info, const std::string& pipelineName, ModelManager& manager) {
    std::string libraryName = nodeConfig["library_name"].GetString();
    auto status = manager.getCustomNodeLibraryManager().getLibrary(libraryName, info.library);
    if (!status.ok()) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Pipeline: {} refers to non existing custom node library: {}", pipelineName, libraryName);
    }
    if (nodeConfig.HasMember("params")) {
        for (const auto& param : nodeConfig["params"].GetObject()) {
            info.parameters.emplace(param.name.GetString(), param.value.GetString());
        }
    }
    return StatusCode::OK;
}

Status processPipelineConfig(rapidjson::Document& configJson, const rapidjson::Value& pipelineConfig, std::set<std::string>& pipelinesInConfigFile, PipelineFactory& factory, ModelManager& manager) {
    const std::string pipelineName = pipelineConfig["name"].GetString();
    if (pipelinesInConfigFile.find(pipelineName) != pipelinesInConfigFile.end()) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Duplicated pipeline names: {} defined in config file. Only first definition will be loaded.", pipelineName);
        return StatusCode::OK;
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Reading pipeline: {} configuration", pipelineName);
    std::set<std::string> demultiplexerNodes;
    std::set<std::string> gatheredDemultiplexerNodes;
    std::optional<int32_t> demultiplyCountEntry = std::nullopt;
    auto demultiplyCountEntryIt = pipelineConfig.FindMember("demultiply_count");
    if (demultiplyCountEntryIt != pipelineConfig.MemberEnd()) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Pipeline: {} does have demultiply at entry node", pipelineName);
        int32_t parsedDemultiplyCount = pipelineConfig["demultiply_count"].GetInt();
        if (parsedDemultiplyCount == 0) {
            parsedDemultiplyCount = -1;
            SPDLOG_LOGGER_WARN(modelmanager_logger, "demultiply_count 0 will be deprecated. For dynamic count use -1.");
        }
        demultiplyCountEntry = parsedDemultiplyCount;
        demultiplexerNodes.insert(ENTRY_NODE_NAME);
    } else {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Pipeline: {} does not have demultiply at entry node", pipelineName);
    }

    std::vector<NodeInfo> info;
    NodeInfo entryInfo{NodeKind::ENTRY, ENTRY_NODE_NAME, "", std::nullopt, {}, demultiplyCountEntry};
    info.emplace_back(std::move(entryInfo));
    processPipelineInputs(pipelineConfig.FindMember("inputs"), ENTRY_NODE_NAME, info[0].outputNameAliases, pipelineName);
    pipeline_connections_t connections;

    auto nodesItr = pipelineConfig.FindMember("nodes");
    for (const auto& nodeConfig : nodesItr->value.GetArray()) {
        std::string nodeName;
        nodeName = nodeConfig["name"].GetString();

        const std::string nodeKindStr = nodeConfig["type"].GetString();
        NodeKind nodeKind;
        auto status = toNodeKind(nodeKindStr, nodeKind);
        if (!status.ok()) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Parsing node kind failed: {} for pipeline: {}", nodeKindStr, pipelineName);
            return status;
        }

        DLNodeInfo dlNodeInfo;
        CustomNodeInfo customNodeInfo;
        if (nodeKind == NodeKind::DL) {
            processDLNodeConfig(nodeConfig, dlNodeInfo);
        } else if (nodeKind == NodeKind::CUSTOM) {
            status = processCustomNodeConfig(nodeConfig, customNodeInfo, pipelineName, manager);
            if (!status.ok()) {
                return status;
            }
        } else {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Pipeline {} contains unknown node kind", pipelineName);
            throw std::invalid_argument("unknown node kind");
        }

        auto nodeOutputsItr = nodeConfig.FindMember("outputs");
        if (nodeOutputsItr == nodeConfig.MemberEnd() || !nodeOutputsItr->value.IsArray()) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Pipeline: {} does not have valid outputs configuration", pipelineName);
            return status;
        }
        std::unordered_map<std::string, std::string> nodeOutputNameAlias;  // key:alias, value realName
        processNodeOutputs(nodeOutputsItr, nodeName, dlNodeInfo.modelName, nodeOutputNameAlias);
        std::optional<int32_t> demultiplyCount;
        if (nodeConfig.HasMember("demultiply_count")) {
            int32_t parsedDemultiplyCount = nodeConfig["demultiply_count"].GetInt();
            if (parsedDemultiplyCount == 0) {
                parsedDemultiplyCount = -1;
                SPDLOG_LOGGER_WARN(modelmanager_logger, "demultiply_count 0 will be deprecated. For dynamic count use -1.");
            }
            demultiplyCount = parsedDemultiplyCount;
            demultiplexerNodes.insert(nodeName);
        }
        std::set<std::string> gatherFromNode;
        if (nodeConfig.HasMember("gather_from_node")) {
            std::string nodeToGatherFrom = nodeConfig["gather_from_node"].GetString();
            gatherFromNode.insert(nodeToGatherFrom);
            gatheredDemultiplexerNodes.insert(nodeToGatherFrom);
        }
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Creating node: {} type: {} model_name: {} modelVersion: {}",
            nodeName, nodeKindStr, dlNodeInfo.modelName, dlNodeInfo.modelVersion.value_or(0));
        info.emplace_back(
            nodeKind,
            nodeName,
            dlNodeInfo.modelName,
            dlNodeInfo.modelVersion,
            nodeOutputNameAlias,
            demultiplyCount,
            gatherFromNode,
            customNodeInfo.library,
            customNodeInfo.parameters);
        auto nodeInputItr = nodeConfig.FindMember("inputs");
        processNodeInputs(nodeName, nodeInputItr, connections);
    }
    const auto iteratorOutputs = pipelineConfig.FindMember("outputs");
    // pipeline outputs are node exit inputs
    processNodeInputs(EXIT_NODE_NAME, iteratorOutputs, connections);
    std::set<std::string> nonGatheredDemultiplexerNodes;
    std::set_difference(demultiplexerNodes.begin(), demultiplexerNodes.end(),
        gatheredDemultiplexerNodes.begin(), gatheredDemultiplexerNodes.end(),
        std::inserter(nonGatheredDemultiplexerNodes, nonGatheredDemultiplexerNodes.begin()));
    info.emplace_back(std::move(NodeInfo(NodeKind::EXIT, EXIT_NODE_NAME, "", std::nullopt, {}, std::nullopt, nonGatheredDemultiplexerNodes)));
    if (!factory.definitionExists(pipelineName)) {
        SPDLOG_DEBUG("Pipeline:{} was not loaded so far. Triggering load", pipelineName);
        auto status = factory.createDefinition(pipelineName, info, connections, manager);
        pipelinesInConfigFile.insert(pipelineName);
        return status;
    }
    SPDLOG_DEBUG("Pipeline:{} is already loaded. Triggering reload", pipelineName);
    auto status = factory.reloadDefinition(pipelineName,
        std::move(info),
        std::move(connections),
        manager);
    pipelinesInConfigFile.insert(pipelineName);
    return status;
}

Status ModelManager::loadCustomNodeLibrariesConfig(rapidjson::Document& configJson) {
    const auto doc = configJson.FindMember("custom_node_library_config_list");
    if (doc == configJson.MemberEnd()) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Configuration file doesn't have custom node libraries property.");
        return StatusCode::OK;
    }
    std::set<std::string> librariesInConfig;
    for (const auto& libraryConfig : doc->value.GetArray()) {
        librariesInConfig.emplace(libraryConfig.FindMember("name")->value.GetString());
        this->customNodeLibraryManager->loadLibrary(
            libraryConfig.FindMember("name")->value.GetString(),
            libraryConfig.FindMember("base_path")->value.GetString());
    }
    this->customNodeLibraryManager->unloadLibrariesRemovedFromConfig(librariesInConfig);
    return StatusCode::OK;
}

Status ModelManager::loadPipelinesConfig(rapidjson::Document& configJson) {
    const auto itrp = configJson.FindMember("pipeline_config_list");
    if (itrp == configJson.MemberEnd() || !itrp->value.IsArray()) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Configuration file doesn't have pipelines property.");
        pipelineFactory.retireOtherThan({}, *this);
        return StatusCode::OK;
    }
    std::set<std::string> pipelinesInConfigFile;
    Status firstErrorStatus = StatusCode::OK;
    for (const auto& pipelineConfig : itrp->value.GetArray()) {
        auto status = processPipelineConfig(configJson, pipelineConfig, pipelinesInConfigFile, pipelineFactory, *this);
        if (status != StatusCode::OK) {
            IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
        }
    }
    pipelineFactory.retireOtherThan(std::move(pipelinesInConfigFile), *this);
    return firstErrorStatus;
}

Status ModelManager::createCustomLoader(CustomLoaderConfig& loaderConfig) {
    auto& customloaders = ovms::CustomLoaders::instance();
    std::string loaderName = loaderConfig.getLoaderName();
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Check if loader is already loaded");
    if (customloaders.find(loaderName) == nullptr) {
        // this is where library or custom loader is loaded
        if (FileSystem::isPathEscaped(loaderConfig.getLibraryPath())) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Path {} escape with .. is forbidden.", loaderConfig.getLibraryPath());
            return StatusCode::PATH_INVALID;
        }
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

    Status firstErrorStatus = StatusCode::OK;
    // Load Customer Loaders as per the configuration
    SPDLOG_DEBUG("Using Customloader");
    for (const auto& configs : itrp->value.GetArray()) {
        const std::string loaderName = configs["config"]["loader_name"].GetString();
        SPDLOG_INFO("Reading Custom Loader: {} configuration", loaderName);

        CustomLoaderConfig loaderConfig;
        auto status = loaderConfig.parseNode(configs["config"]);
        if (status != StatusCode::OK) {
            IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
            SPDLOG_ERROR("Parsing loader: {} config failed", loaderName);
        }

        auto retVal = createCustomLoader(loaderConfig);
        if (retVal != StatusCode::OK) {
            IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(retVal);
            SPDLOG_ERROR("Creation of loader: {} failed", loaderName);
        }
    }
    // All loaders are the done. Finalize the list by deleting removed loaders in config
    auto& customloaders = ovms::CustomLoaders::instance();
    customloaders.finalize();
    return firstErrorStatus;
}

Status ModelManager::loadModelsConfig(rapidjson::Document& configJson, std::vector<ModelConfig>& gatedModelConfigs) {
    const auto itr = configJson.FindMember("model_config_list");
    if (itr == configJson.MemberEnd() || !itr->value.IsArray()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Configuration file doesn't have models property.");
        return StatusCode::JSON_INVALID;
    }
    Status firstErrorStatus = StatusCode::OK;
    std::set<std::string> modelsInConfigFile;
    std::set<std::string> modelsWithInvalidConfig;
    std::unordered_map<std::string, ModelConfig> newModelConfigs;
    for (const auto& configs : itr->value.GetArray()) {
        ModelConfig modelConfig;
        auto status = modelConfig.parseNode(configs["config"]);
        if (!status.ok()) {
            IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(StatusCode::MODEL_CONFIG_INVALID);
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Parsing model: {} config failed due to error: {}", modelConfig.getName(), status.string());
            modelsWithInvalidConfig.emplace(modelConfig.getName());
            continue;
        }

        status = validatePluginConfiguration(modelConfig.getPluginConfig(), modelConfig.getTargetDevice(), *ieCore.get());
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Plugin config contains unsupported keys");
            return status;
        }
        modelConfig.setCacheDir(this->modelCacheDirectory);

        const auto modelName = modelConfig.getName();
        if (pipelineDefinitionExists(modelName)) {
            IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(StatusCode::MODEL_NAME_OCCUPIED);
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Model name: {} is already occupied by pipeline definition.", modelName);
            continue;
        }
        if (modelsInConfigFile.find(modelName) != modelsInConfigFile.end()) {
            IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(StatusCode::MODEL_NAME_OCCUPIED);
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Duplicated model names: {} defined in config file. Only first definition will be loaded.", modelName);
            continue;
        }
        status = reloadModelWithVersions(modelConfig);
        IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);

        modelsInConfigFile.emplace(modelName);
        if (!status.ok()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Cannot reload model: {} with versions due to error: {}", modelName, status.string());
        }
        if (status == StatusCode::REQUESTED_DYNAMIC_PARAMETERS_ON_SUBSCRIBED_MODEL || status == StatusCode::REQUESTED_STATEFUL_PARAMETERS_ON_SUBSCRIBED_MODEL) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Will retry to reload model({}) after pipelines are revalidated", modelName);
            auto it = this->servedModelConfigs.find(modelName);
            if (it == this->servedModelConfigs.end()) {
                continue;
            }
            gatedModelConfigs.emplace_back(std::move(modelConfig));
            newModelConfigs.emplace(modelName, std::move(it->second));
            this->servedModelConfigs.erase(modelName);
        } else {
            newModelConfigs.emplace(modelName, std::move(modelConfig));
        }
    }
    this->servedModelConfigs = std::move(newModelConfigs);
    retireModelsRemovedFromConfigFile(modelsInConfigFile, modelsWithInvalidConfig);
    return firstErrorStatus;
}

Status ModelManager::tryReloadGatedModelConfigs(std::vector<ModelConfig>& gatedModelConfigs) {
    Status firstErrorStatus = StatusCode::OK;
    for (auto& modelConfig : gatedModelConfigs) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Trying to reload model({}) configuration", modelConfig.getName());
        auto status = reloadModelWithVersions(modelConfig);
        if (!status.ok()) {
            IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
            continue;
        }
        auto it = this->servedModelConfigs.find(modelConfig.getName());
        if (it == this->servedModelConfigs.end()) {
            IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
            continue;
        }
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Successfully retried to load new model({}) configuration after unsubscribed from pipeline", modelConfig.getName());
        this->servedModelConfigs.at(modelConfig.getName()) = std::move(modelConfig);
    }
    return firstErrorStatus;
}

class LoudFileInfoReporter {
    std::stringstream ss;

public:
    LoudFileInfoReporter(const std::string& filename, std::ifstream& file) {
        struct stat statTime;

        if (stat(filename.c_str(), &statTime) != 0) {
            SPDLOG_ERROR("Failed to debug-read fileconfig");
            return;
        }
        ss << "FileInfoReporter: " << filename
           << " time modification [s]: " << statTime.st_ctim.tv_sec
           << " [ns]: " << statTime.st_ctim.tv_nsec << std::endl;
        std::string some;
        file.clear();
        file.seekg(0);
        while (file) {
            file >> some;
            ss << some << std::endl;
        }
        file.clear();
        file.seekg(0);
    }
    void log() {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, ss.str());
    }
};

Status ModelManager::loadConfig(const std::string& jsonFilename) {
    std::lock_guard<std::recursive_mutex> loadingLock(configMtx);
    configFilename = jsonFilename;
    lastConfigFileMD5 = getConfigFileMD5();
    rapidjson::Document configJson;

    uint16_t counter = 0;
    Status intermediateStatus;
    do {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Loading configuration from {} for: {} time", jsonFilename, counter + 1);
        std::ifstream ifs(jsonFilename.c_str());
        LoudFileInfoReporter loud(jsonFilename, ifs);
        if (!ifs.good()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Configuration file is invalid {}", jsonFilename);
            intermediateStatus = StatusCode::CONFIG_FILE_INVALID;
            loud.log();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::ParseResult parseResult = configJson.ParseStream(isw);
        if (!parseResult) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Configuration file is not a valid JSON file. Error: {}",
                rapidjson::GetParseError_En(parseResult.Code()));
            intermediateStatus = StatusCode::JSON_INVALID;
            loud.log();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        intermediateStatus = StatusCode::OK;
        break;
    } while (++counter < MAX_CONFIG_JSON_READ_RETRY_COUNT && !intermediateStatus.ok());
    if (!intermediateStatus.ok()) {
        lastLoadConfigStatus = intermediateStatus;
        return lastLoadConfigStatus;
    }

    if (validateJsonAgainstSchema(configJson, MODELS_CONFIG_SCHEMA) != StatusCode::OK) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Configuration file is not in valid configuration format");
        lastLoadConfigStatus = StatusCode::JSON_INVALID;
        return lastLoadConfigStatus;
    }
    Status status;
    Status firstErrorStatus = StatusCode::OK;
    // load the custom loader config, if available
    status = loadCustomLoadersConfig(configJson);
    if (!status.ok()) {
        IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
    }
    std::vector<ModelConfig> gatedModelConfigs;
    status = loadModelsConfig(configJson, gatedModelConfigs);
    if (!status.ok()) {
        IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
    }
    status = loadCustomNodeLibrariesConfig(configJson);
    if (!status.ok()) {
        IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
    }
    status = loadPipelinesConfig(configJson);
    if (!status.ok()) {
        IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
    }
    status = tryReloadGatedModelConfigs(gatedModelConfigs);
    if (!status.ok()) {
        IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
    }

    lastLoadConfigStatus = firstErrorStatus;
    return firstErrorStatus;
}

void ModelManager::retireModelsRemovedFromConfigFile(const std::set<std::string>& modelsExistingInConfigFile, const std::set<std::string>& modelsWithInvalidConfig) {
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
        if (modelsWithInvalidConfig.find(modelName) == modelsWithInvalidConfig.end()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Retiring all versions of model: {}", modelName);
            try {
                models.at(modelName)->retireAllVersions();
            } catch (const std::out_of_range& e) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Unknown error occurred when tried to retire all versions of model: {}", modelName);
            }
        } else {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Cleaning up all versions of model: {}", modelName);
            try {
                models.at(modelName)->cleanupAllVersions();
            } catch (const std::out_of_range& e) {
                SPDLOG_LOGGER_ERROR(modelmanager_logger, "Unknown error occurred when tried to clean up all versions of model: {}", modelName);
            }
        }
    }
}

Status ModelManager::updateConfigurationWithoutConfigFile() {
    std::lock_guard<std::recursive_mutex> loadingLock(configMtx);
    SPDLOG_LOGGER_TRACE(modelmanager_logger, "Checking if something changed with model versions");
    bool reloadNeeded = false;
    Status firstErrorStatus = StatusCode::OK;
    Status status;
    for (auto& [name, config] : servedModelConfigs) {
        status = reloadModelWithVersions(config);
        if (!status.ok()) {
            IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
        } else if (status == StatusCode::OK_RELOADED) {
            reloadNeeded = true;
        }
    }
    status = pipelineFactory.revalidatePipelines(*this);
    if (!status.ok()) {
        IF_ERROR_NOT_OCCURRED_EARLIER_THEN_SET_FIRST_ERROR(status);
    }

    if (!firstErrorStatus.ok()) {
        return firstErrorStatus;
    }

    if (reloadNeeded) {
        return StatusCode::OK_RELOADED;
    } else {
        return StatusCode::OK_NOT_RELOADED;
    }
}

std::string ModelManager::getConfigFileMD5() {
    std::ifstream ifs;
    ifs.open(configFilename);
    std::stringstream strStream;
    strStream << ifs.rdbuf();
    std::string str = strStream.str();
    ifs.close();

    unsigned char result[MD5_DIGEST_LENGTH];
    MD5((unsigned char*)str.c_str(), str.size(), result);
    std::string md5sum(reinterpret_cast<char*>(result), MD5_DIGEST_LENGTH);
    return (md5sum);
}

Status ModelManager::configFileReloadNeeded(bool& isNeeded) {
    std::lock_guard<std::recursive_mutex> loadingLock(configMtx);

    if (!std::ifstream(configFilename)) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Config file not found or cannot open.");
        isNeeded = false;
        return StatusCode::CONFIG_FILE_TIMESTAMP_READING_FAILED;
    }

    std::string newmd5 = getConfigFileMD5();
    bool configFileModified = false;
    if (lastConfigFileMD5 != newmd5) {
        configFileModified = true;
    }

    if (configFilename == "" || !configFileModified) {
        isNeeded = false;
        return lastLoadConfigStatus;
    } else {
        isNeeded = true;
    }

    return StatusCode::OK;
}

void ModelManager::watcher(std::future<void> exitSignal) {
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Started model manager thread");

    while (exitSignal.wait_for(std::chrono::seconds(watcherIntervalSec)) == std::future_status::timeout) {
        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Models configuration and filesystem check cycle begin");
        std::lock_guard<std::recursive_mutex> loadingLock(configMtx);
        bool isNeeded;
        configFileReloadNeeded(isNeeded);
        if (isNeeded) {
            loadConfig(configFilename);
        }
        updateConfigurationWithoutConfigFile();

        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Models configuration and filesystem check cycle end");
    }
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Stopped model manager thread");
}

void ModelManager::cleanerRoutine(uint32_t resourcesCleanupIntervalSec, uint32_t sequenceCleanerIntervalMinutes, std::future<void> cleanerExitSignal) {
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Started cleaner thread");

    uint32_t resourcesCleanupIntervalMiliseconds = resourcesCleanupIntervalSec * 1000;
    uint32_t sequenceCleanerIntervalMiliseconds = sequenceCleanerIntervalMinutes * 60 * 1000;

    FunctorResourcesCleaner functorResourcesCleaner{*this};
    FunctorSequenceCleaner functorSequenceCleaner{globalSequencesViewer};

    ovms::cleanerRoutine(resourcesCleanupIntervalMiliseconds, functorResourcesCleaner, sequenceCleanerIntervalMiliseconds, functorSequenceCleaner, cleanerExitSignal);

    SPDLOG_LOGGER_INFO(modelmanager_logger, "Stopped cleaner thread");
}

void cleanerRoutine(uint32_t resourcesCleanupInterval, FunctorResourcesCleaner& functorResourcesCleaner, uint32_t sequenceCleanerInterval, FunctorSequenceCleaner& functorSequenceCleaner, std::future<void>& cleanerExitSignal) {
    uint32_t currentResourcesWaitTime = resourcesCleanupInterval;
    uint32_t currentSequenceWaitTime = sequenceCleanerInterval;
    bool shouldCheckForSequenceCleanup = sequenceCleanerInterval != 0;
    uint32_t currentWaitTime = (!shouldCheckForSequenceCleanup || currentResourcesWaitTime < currentSequenceWaitTime) ? currentResourcesWaitTime : currentSequenceWaitTime;

    while (cleanerExitSignal.wait_for(std::chrono::milliseconds(currentWaitTime)) == std::future_status::timeout) {
        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Cleanup check cycle begin");

        currentResourcesWaitTime = (currentResourcesWaitTime - currentWaitTime) == 0 ? resourcesCleanupInterval : currentResourcesWaitTime - currentWaitTime;
        if (shouldCheckForSequenceCleanup)
            currentSequenceWaitTime = (currentSequenceWaitTime - currentWaitTime) == 0 ? sequenceCleanerInterval : currentSequenceWaitTime - currentWaitTime;
        currentWaitTime = (!shouldCheckForSequenceCleanup || currentResourcesWaitTime < currentSequenceWaitTime) ? currentResourcesWaitTime : currentSequenceWaitTime;

        if (currentResourcesWaitTime == resourcesCleanupInterval)
            functorResourcesCleaner.cleanup();
        if (currentSequenceWaitTime == sequenceCleanerInterval && shouldCheckForSequenceCleanup)
            functorSequenceCleaner.cleanup();

        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Cleanup check cycle end");
    }
}

void ModelManager::cleanupResources() {
    std::vector<std::shared_ptr<CNLIMWrapper>> toBeRemoved;
    std::unique_lock resourcesLock(resourcesMtx);
    // Move all resources that should be destroyed to temporary container
    std::copy_if(resources.begin(),
        resources.end(),
        std::back_inserter(toBeRemoved),
        [](auto& resource) { return resource.use_count() == 1; });
    resources.erase(
        std::remove_if(
            resources.begin(),
            resources.end(),
            [toBeRemoved](auto& resource) { return std::find(toBeRemoved.begin(), toBeRemoved.end(), resource) != toBeRemoved.end(); }),
        resources.end());
    // Unlock mutex so new resources can be put into container owned by ModelManager
    resourcesLock.unlock();
    // Temporary container will fall out of scope and therefore deinitialize should be called on every resource inside of it
}

void ModelManager::join() {
    if (watcherStarted) {
        exitTrigger.set_value();
        if (monitor.joinable()) {
            monitor.join();
            watcherStarted = false;
            SPDLOG_INFO("Shutdown model manager");
        }
    }

    if (cleanerStarted) {
        cleanerExitTrigger.set_value();
        if (cleanerThread.joinable()) {
            cleanerThread.join();
            cleanerStarted = false;
            SPDLOG_INFO("Shutdown cleaner thread");
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
    SPDLOG_LOGGER_TRACE(modelmanager_logger, "Currently registered model: {} versions count: {}", newModelConfig.getName(), modelVersionsInstances.size());
    for (const auto& [version, versionInstance] : modelVersionsInstances) {
        SPDLOG_LOGGER_TRACE(modelmanager_logger, "model: {} version: {} state: {}", newModelConfig.getName(), version, ovms::ModelVersionStateToString(versionInstance->getStatus().getState()));
        registeredModelVersions.push_back(version);
    }

    if (newModelConfig.isCustomLoaderRequiredToLoadModel()) {
        custom_loader_options_config_t customLoaderOptionsConfig = newModelConfig.getCustomLoaderOptionsConfigMap();
        const std::string loaderName = customLoaderOptionsConfig["loader_name"];

        auto& customloaders = ovms::CustomLoaders::instance();
        auto loaderPtr = customloaders.find(loaderName);
        if (loaderPtr != nullptr) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Custom Loader to be used : {}", loaderName);

            // check existing version for blacklist
            for (const auto& [version, versionInstance] : modelVersionsInstances) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "The model {} checking for blacklist", versionInstance->getName());
                CustomLoaderStatus bres = loaderPtr->getModelBlacklistStatus(versionInstance->getName(), version);
                if (bres != CustomLoaderStatus::OK) {
                    SPDLOG_LOGGER_INFO(modelmanager_logger, "The model {} is blacklisted", versionInstance->getName());
                    requestedVersions.erase(std::remove(requestedVersions.begin(), requestedVersions.end(), version), requestedVersions.end());
                }
            }
        }
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
                modelVersionsInstances.at(version)->getStatus().isFailedLoading() ||
                modelVersionsInstances.at(version)->getModelConfig().isReloadRequired(newModelConfig)) {
                if (modelVersionsInstances.at(version)->getModelConfig().isCustomLoaderConfigChanged(newModelConfig)) {
                    modelVersionsInstances.at(version)->setCustomLoaderConfigChangeFlag();
                }
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

Status ModelManager::checkStatefulFlagChange(const std::string& modelName, bool configStatefulFlag) {
    std::unique_lock modelsLock(modelsMtx);
    auto modelIt = models.find(modelName);
    if (models.end() == modelIt)
        return StatusCode::OK;  // Model has not been loaded yet, so there are no restrictions regarding stateful flag setup

    auto model = models[modelName];
    if (model->isStateful() != configStatefulFlag)
        return StatusCode::REQUESTED_MODEL_TYPE_CHANGE;
    return StatusCode::OK;
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
        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Detected version folder: {}", entry);
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
    }

    return StatusCode::OK;
}

Status ModelManager::addModelVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<FileSystem>& fs, ModelConfig& config, std::shared_ptr<model_versions_t>& versionsToStart, std::shared_ptr<model_versions_t> versionsFailed) {
    Status status = StatusCode::OK;
    try {
        status = model->addVersions(versionsToStart, config, fs, *ieCore, versionsFailed);
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error occurred while loading model: {} versions; error: {}",
                config.getName(),
                status.string());
            return status;
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
        auto status = model->reloadVersions(versionsToReload, config, fs, *ieCore, versionsFailed);
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error occurred while reloading model: {}; versions; error: {}",
                config.getName(),
                status.string());

            return status;
        }
    } catch (std::exception& e) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Exception occurred while reloading model: {};", e.what());
    }

    return status;
}

Status ModelManager::reloadModelWithVersions(ModelConfig& config) {
    SPDLOG_LOGGER_TRACE(modelmanager_logger, "Started applying config changes to model: {}", config.getName());

    if (config.isStateful() && config.isDynamicParameterEnabled()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Requested setting dynamic parameters for stateful model {}. Dynamic shape and dynamic batch size not supported for stateful models.", config.getName());
        return StatusCode::REQUESTED_DYNAMIC_PARAMETERS_ON_STATEFUL_MODEL;
    }
    if (!config.isStateful()) {
        if (config.isLowLatencyTransformationUsed()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Requested low latency transformation parameter for non stateful model {}.", config.getName());
            return StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER;
        }
    }

    auto status = checkStatefulFlagChange(config.getName(), config.isStateful());
    if (!status.ok()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Requested model type change on model: {}. Stateful flag cannot be changed after model is loaded", config.getName());
        return StatusCode::REQUESTED_MODEL_TYPE_CHANGE;
    }

    auto model = getModelIfExistCreateElse(config.getName(), config.isStateful());
    if (model->isAnyVersionSubscribed()) {
        if (config.isDynamicParameterEnabled()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Requested setting dynamic parameters for model {} but it is used in pipeline. Cannot reload model configuration.", config.getName());
            return StatusCode::REQUESTED_DYNAMIC_PARAMETERS_ON_SUBSCRIBED_MODEL;
        }
        if (config.isStateful()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Requested using stateful model {} but it is used in pipeline. Stateful model cannot be subscribed to pipeline.", config.getName());
            return StatusCode::REQUESTED_STATEFUL_PARAMETERS_ON_SUBSCRIBED_MODEL;
        }
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
        } else {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Specified custom loader {} not found. In case any models are loaded, will be unloading them", loaderName);
            model->retireAllVersions();
            return StatusCode::OK;
        }
    }
    getVersionsToChange(config, model->getModelVersions(), requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    bool reloadNeeded = false;
    if (versionsToStart->size() > 0 || versionsToReload->size() > 0 || versionsToRetire->size() > 0) {
        reloadNeeded = true;
    }
    std::set<ovms::model_version_t> allFailedVersions;
    while (versionsToStart->size() > 0) {
        blocking_status = addModelVersions(model, fs, config, versionsToStart, versionsFailed);
        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Adding new versions. Status: {};", blocking_status.string());
        if (!blocking_status.ok()) {
            for (const auto version : *versionsFailed) {
                SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Removing available version {} due to load failure; ", version);
                if (std::binary_search(availableVersions.begin(), availableVersions.end(), version)) {
                    availableVersions.erase(std::remove(availableVersions.begin(), availableVersions.end(), version), availableVersions.end());
                }
                allFailedVersions.insert(version);
            }
            requestedVersions = config.getModelVersionPolicy()->filter(availableVersions);
            getVersionsToChange(config, model->getModelVersions(), requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
        } else {
            break;
        }
    }

    if (versionsToReload->size() > 0) {
        auto status = reloadModelVersions(model, fs, config, versionsToReload, versionsFailed);
        if (!status.ok()) {
            blocking_status = status;
        }
    }

    for (const auto version : *versionsFailed) {
        SPDLOG_LOGGER_TRACE(modelmanager_logger, "Removing available version {} due to load failure.", version);
        if (std::binary_search(availableVersions.begin(), availableVersions.end(), version)) {
            availableVersions.erase(std::remove(availableVersions.begin(), availableVersions.end(), version), availableVersions.end());
        }
        allFailedVersions.insert(version);
    }
    // refresh versions to retire based on failed reloads
    requestedVersions = config.getModelVersionPolicy()->filter(availableVersions);
    getVersionsToChange(config, model->getModelVersions(), requestedVersions, versionsToStart, versionsToReload, versionsToRetire);
    std::shared_ptr<model_versions_t> versionsToCleanup = std::make_shared<model_versions_t>();
    std::copy_if(versionsToRetire->begin(), versionsToRetire->end(), std::back_inserter(*versionsToCleanup), [&](auto& version) { return allFailedVersions.find(version) != allFailedVersions.end(); });
    versionsToRetire->erase(std::remove_if(versionsToRetire->begin(), versionsToRetire->end(), [&](auto& version) { return allFailedVersions.find(version) != allFailedVersions.end(); }), versionsToRetire->end());
    if (versionsToRetire->size() > 0) {
        auto status = model->retireVersions(versionsToRetire);
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error occurred while unloading model: {}; versions; error: {}",
                config.getName(),
                status.string());
            return status;
        }
    }
    if (versionsToCleanup->size() > 0) {
        auto status = model->cleanupFailedLoad(versionsToCleanup);
        if (!status.ok()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Error occurred while cleaning up model that failed to load: {}; versions; error: {}",
                config.getName(),
                status.string());
            return status;
        }
    }

    if (blocking_status.ok() && reloadNeeded) {
        return StatusCode::OK_RELOADED;
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

    return modelInstance->waitForLoaded(waitForModelLoadedTimeoutMs, modelInstanceUnloadGuardPtr);
}

const CustomNodeLibraryManager& ModelManager::getCustomNodeLibraryManager() const {
    return *customNodeLibraryManager;
}

}  // namespace ovms

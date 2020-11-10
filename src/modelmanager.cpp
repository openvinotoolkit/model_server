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

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

#include "azurefilesystem.hpp"
#include "config.hpp"
#include "filesystem.hpp"
#include "gcsfilesystem.hpp"
#include "localfilesystem.hpp"
#include "pipeline.hpp"
#include "pipeline_factory.hpp"
#include "s3filesystem.hpp"
#include "schema.hpp"

namespace ovms {

static uint watcherIntervalSec = 1;
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
        spdlog::error("Couldn't start model manager");
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

    auto& modelConfig = servedModelConfigs.emplace_back(
        config.modelName(),
        config.modelPath(),
        config.targetDevice(),
        config.batchSize(),
        config.nireq());

    auto status = modelConfig.parsePluginConfig(config.pluginConfig());
    if (!status.ok()) {
        spdlog::error("Couldn't parse plugin config");
        return status;
    }

    status = modelConfig.parseModelVersionPolicy(config.modelVersionPolicy());
    if (!status.ok()) {
        spdlog::error("Couldn't parse model version policy. {}", status.string());
        return status;
    }

    status = modelConfig.parseShapeParameter(config.shape());
    if (!status.ok()) {
        spdlog::error("Couldn't parse shape parameter");
        return status;
    }

    bool batchSizeSet = (modelConfig.getBatchingMode() != FIXED || modelConfig.getBatchSize() != 0);
    bool shapeSet = (modelConfig.getShapes().size() > 0);

    spdlog::debug("Batch size set: {}, shape set: {}", batchSizeSet, shapeSet);
    if (batchSizeSet && shapeSet) {
        spdlog::warn("Both shape and batch size have been defined. Batch size parameter will be ignored.");
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
            SPDLOG_INFO("Creating node dependencies mapping request. Node:{} input:{} <- SourceNode:{} output:{}",
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

void processNodeOutputs(const rapidjson::Value::ConstMemberIterator& nodeOutputsItr, const std::string& nodeName, const std::string& modelName, std::unordered_map<std::string, std::string>& nodeOutputNameAlias) {
    for (const auto& nodeOutput : nodeOutputsItr->value.GetArray()) {
        const std::string modelOutputName = nodeOutput.GetObject()["data_item"].GetString();
        const std::string nodeOutputName = nodeOutput.GetObject()["alias"].GetString();
        SPDLOG_INFO("Alliasing node:{} model_name:{} output:{}, under alias:{}",
            nodeName, modelName, modelOutputName, nodeOutputName);
        nodeOutputNameAlias[nodeOutputName] = modelOutputName;
    }
}

void processPipelineConfig(rapidjson::Document& configJson, const rapidjson::Value& pipelineConfig, std::set<std::string>& pipelinesInConfigFile, PipelineFactory& factory, ModelManager& manager) {
    const std::string pipelineName = pipelineConfig["name"].GetString();
    SPDLOG_INFO("Reading pipeline:{} configuration", pipelineName);
    auto itr2 = pipelineConfig.FindMember("nodes");

    std::vector<NodeInfo> info{
        {NodeKind::ENTRY, "request"}};
    pipeline_connections_t connections;
    for (const auto& nodeConfig : itr2->value.GetArray()) {
        std::string nodeName;
        nodeName = nodeConfig["name"].GetString();

        std::string modelName;
        modelName = nodeConfig["model_name"].GetString();

        const std::string nodeKindStr = nodeConfig["type"].GetString();
        auto nodeOutputsItr = nodeConfig.FindMember("outputs");
        if (nodeOutputsItr == nodeConfig.MemberEnd() || !nodeOutputsItr->value.IsArray()) {
            SPDLOG_INFO("Pipeline:{} does not have valid outputs configuration", pipelineName);
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
            SPDLOG_ERROR("There was error while parsing node kind:{}", nodeKindStr);
            return;
        }
        SPDLOG_INFO("Creating node:{} type:{} model_name:{} modelVersion:{}",
            nodeName, nodeKindStr, modelName, modelVersion.value_or(0));
        info.emplace_back(std::move(NodeInfo{nodeKind, nodeName, modelName, modelVersion, nodeOutputNameAlias}));
        auto nodeInputItr = nodeConfig.FindMember("inputs");
        processNodeInputs(nodeName, nodeInputItr, connections);
    }
    const auto iteratorOutputs = pipelineConfig.FindMember("outputs");
    const std::string nodeName = "response";
    // pipeline outputs are node exit inputs
    processNodeInputs(nodeName, iteratorOutputs, connections);
    info.emplace_back(std::move(NodeInfo(NodeKind::EXIT, nodeName, "", std::nullopt, {})));
    auto status = factory.createDefinition(pipelineName, info, connections, manager);
    if (!status.ok()) {
        return;
    }
    pipelinesInConfigFile.insert(pipelineName);
}

Status ModelManager::loadPipelinesConfig(rapidjson::Document& configJson) {
    const auto itrp = configJson.FindMember("pipeline_config_list");
    if (itrp == configJson.MemberEnd() || !itrp->value.IsArray()) {
        SPDLOG_INFO("Configuration file doesn't have pipelines property.");
        return StatusCode::OK;
    }
    std::set<std::string> pipelinesInConfigFile;
    for (const auto& pipelineConfig : itrp->value.GetArray()) {
        processPipelineConfig(configJson, pipelineConfig, pipelinesInConfigFile, pipelineFactory, *this);
    }
    return ovms::StatusCode::OK;
}

Status ModelManager::loadModelsConfig(rapidjson::Document& configJson) {
    const auto itr = configJson.FindMember("model_config_list");
    if (itr == configJson.MemberEnd() || !itr->value.IsArray()) {
        SPDLOG_ERROR("Configuration file doesn't have models property.");
        return StatusCode::JSON_INVALID;
    }
    std::set<std::string> modelsInConfigFile;
    servedModelConfigs.clear();
    for (const auto& configs : itr->value.GetArray()) {
        ModelConfig& modelConfig = servedModelConfigs.emplace_back();
        auto status = modelConfig.parseNode(configs["config"]);
        if (!status.ok()) {
            SPDLOG_ERROR("Parsing model:{} config failed",
                modelConfig.getName());
            servedModelConfigs.pop_back();
            continue;
        }
        reloadModelWithVersions(modelConfig);
        modelsInConfigFile.emplace(modelConfig.getName());
    }
    retireModelsRemovedFromConfigFile(modelsInConfigFile);
    return ovms::StatusCode::OK;
}

Status ModelManager::loadConfig(const std::string& jsonFilename) {
    spdlog::info("Loading configuration from {}", jsonFilename);
    std::ifstream ifs(jsonFilename.c_str());
    if (!ifs.good()) {
        SPDLOG_ERROR("File is invalid {}", jsonFilename);
        return StatusCode::FILE_INVALID;
    }
    rapidjson::Document configJson;
    rapidjson::IStreamWrapper isw(ifs);
    if (configJson.ParseStream(isw).HasParseError()) {
        SPDLOG_ERROR("Configuration file is not a valid JSON file.");
        return StatusCode::JSON_INVALID;
    }

    if (validateJsonAgainstSchema(configJson, MODELS_CONFIG_SCHEMA) != StatusCode::OK) {
        SPDLOG_ERROR("Configuration file is not in valid configuration format");
        return StatusCode::JSON_INVALID;
    }
    configFilename = jsonFilename;
    Status status;
    status = loadModelsConfig(configJson);
    if (status != StatusCode::OK) {
        return status;
    }
    status = loadPipelinesConfig(configJson);
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
        try {
            models.at(modelName)->retireAllVersions();
        } catch (const std::out_of_range& e) {
            SPDLOG_ERROR("Unknown error occured when tried to retire all versions of model:{}", modelName);
        }
    }
}

void ModelManager::watcher(std::future<void> exit) {
    SPDLOG_INFO("Started config watcher thread");
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
        }
        for (auto& config : servedModelConfigs) {
            reloadModelWithVersions(config);
        }
    }
    spdlog::info("Exited config watcher thread");
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
    spdlog::debug("Currently registered versions count:{}", modelVersionsInstances.size());
    for (const auto& [version, versionInstance] : modelVersionsInstances) {
        spdlog::debug("version:{} state:{}", version, ovms::ModelVersionStateToString(versionInstance->getStatus().getState()));
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
            spdlog::error("Data race occured during versions update. Could not found version. Details:{}", e.what());
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
        spdlog::error("Data race occured during versions update. Could not found version. Details:{}", e.what());
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

std::shared_ptr<ovms::Model> ModelManager::getModelIfExistCreateElse(const std::string& modelName) {
    std::unique_lock modelsLock(modelsMtx);
    auto modelIt = models.find(modelName);
    if (models.end() == modelIt) {
        models.insert({modelName, modelFactory(modelName)});
    }
    return models[modelName];
}

std::shared_ptr<FileSystem> getFilesystem(const std::string& basePath) {
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
    auto status = fs->isDirectory(base, &is_directory);
    if (status != StatusCode::OK) {
        spdlog::error("Couldn't check directory: {}", base);
        return status;
    }
    if (!is_directory) {
        spdlog::error("Directory does not exist: {}", base);
        return StatusCode::PATH_INVALID;
    }

    status = fs->getDirectorySubdirs(base, &dirs);

    if (status != StatusCode::OK) {
        spdlog::error("Couldn't list directories in path: {}", base);
        return status;
    }

    for (const auto& entry : dirs) {
        try {
            ovms::model_version_t version = std::stoll(entry);
            if (version <= 0) {
                SPDLOG_WARN("Expected version directory name to be a number greater than 0. Got:{}", version);
                continue;
            }
            versions.push_back(version);
        } catch (const std::invalid_argument& e) {
            spdlog::warn("Expected version directory name to be in number format. Got:{}", entry);
        } catch (const std::out_of_range& e) {
            spdlog::error("Directory name is out of range for supported version format. Got:{}", entry);
        }
    }

    if (0 == versions.size()) {
        spdlog::error("No version found for model in path:{}", base);
        return StatusCode::NO_MODEL_VERSION_AVAILABLE;
    }

    return StatusCode::OK;
}

StatusCode downloadModels(std::shared_ptr<FileSystem>& fs, ModelConfig& config, std::shared_ptr<model_versions_t> versions) {
    if (versions->size() == 0) {
        return StatusCode::OK;
    }
    std::string localPath;
    spdlog::info("Getting model from {}", config.getBasePath());
    auto sc = fs->downloadModelVersions(config.getBasePath(), &localPath, *versions);
    if (sc != StatusCode::OK) {
        spdlog::error("Couldn't download model from {}", config.getBasePath());
        return sc;
    }
    config.setLocalPath(localPath);
    spdlog::info("Model downloaded to {}", config.getLocalPath());

    return StatusCode::OK;
}

Status ModelManager::cleanupModelTmpFiles(ModelConfig& config) {
    auto lfstatus = StatusCode::OK;

    if (config.getLocalPath().compare(config.getBasePath())) {
        LocalFileSystem lfs;
        lfstatus = lfs.deleteFileFolder(config.getLocalPath());
        if (lfstatus != StatusCode::OK) {
            spdlog::error("Error occurred while deleting local copy of cloud model: {} reason {}",
                config.getLocalPath(),
                lfstatus);
        } else {
            spdlog::info("Model removed from {}", config.getLocalPath());
        }
    }

    return lfstatus;
}

Status ModelManager::addModelVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<FileSystem>& fs, ModelConfig& config, std::shared_ptr<model_versions_t>& versionsToStart) {
    Status status = StatusCode::OK;
    try {
        downloadModels(fs, config, versionsToStart);
        status = model->addVersions(versionsToStart, config);
        if (!status.ok()) {
            spdlog::error("Error occurred while loading model: {} versions; error: {}",
                config.getName(),
                status.string());
        }
    } catch (std::exception& e) {
        spdlog::error("Exception occurred while loading model: {};", e.what());
    }

    cleanupModelTmpFiles(config);
    return status;
}

Status ModelManager::reloadModelVersions(std::shared_ptr<ovms::Model>& model, std::shared_ptr<FileSystem>& fs, ModelConfig& config, std::shared_ptr<model_versions_t>& versionsToReload) {
    Status status = StatusCode::OK;

    try {
        downloadModels(fs, config, versionsToReload);
        auto status = model->reloadVersions(versionsToReload, config);
        if (!status.ok()) {
            spdlog::error("Error occurred while reloading model: {}; versions; error: {}",
                config.getName(),
                status.string());
        }
    } catch (std::exception& e) {
        spdlog::error("Exception occurred while reloading model: {};", e.what());
    }

    cleanupModelTmpFiles(config);
    return status;
}

Status ModelManager::reloadModelWithVersions(ModelConfig& config) {
    auto fs = getFilesystem(config.getBasePath());
    std::vector<model_version_t> requestedVersions;
    auto blocking_status = readAvailableVersions(fs, config.getBasePath(), requestedVersions);
    if (!blocking_status.ok()) {
        return blocking_status;
    }
    requestedVersions = config.getModelVersionPolicy()->filter(requestedVersions);

    std::shared_ptr<model_versions_t> versionsToStart;
    std::shared_ptr<model_versions_t> versionsToReload;
    std::shared_ptr<model_versions_t> versionsToRetire;

    auto model = getModelIfExistCreateElse(config.getName());
    getVersionsToChange(config, model->getModelVersions(), requestedVersions, versionsToStart, versionsToReload, versionsToRetire);

    if (versionsToStart->size() > 0) {
        auto blocking_status = addModelVersions(model, fs, config, versionsToStart);
        if (!blocking_status.ok()) {
            return blocking_status;
        }
    }

    if (versionsToReload->size() > 0) {
        reloadModelVersions(model, fs, config, versionsToReload);
    }

    auto status = model->retireVersions(versionsToRetire);
    if (!status.ok()) {
        spdlog::error("Error occurred while unloading model: {}; versions; error: {}",
            config.getName(),
            status.string());
    }

    return blocking_status;
}

const std::shared_ptr<Model> ModelManager::findModelByName(const std::string& name) const {
    std::shared_lock lock(modelsMtx);
    auto it = models.find(name);
    return it != models.end() ? it->second : nullptr;
}

}  // namespace ovms

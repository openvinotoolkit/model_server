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
#include "model.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <utility>

namespace ovms {

void Model::subscribe(PipelineDefinition& pd) {
    subscriptionManager.subscribe(pd);
}

void Model::unsubscribe(PipelineDefinition& pd) {
    subscriptionManager.unsubscribe(pd);
}

const std::map<model_version_t, const ModelInstance&> Model::getModelVersionsMapCopy() const {
    std::shared_lock lock(modelVersionsMtx);
    std::map<model_version_t, const ModelInstance&> modelInstancesMapCopy;
    for (auto& [modelVersion, modelInstancePtr] : modelVersions) {
        modelInstancesMapCopy.insert({modelVersion, *modelInstancePtr});
    }
    return std::move(modelInstancesMapCopy);
}

const std::map<model_version_t, std::shared_ptr<ModelInstance>>& Model::getModelVersions() const {
    return modelVersions;
}

void Model::updateDefaultVersion() {
    model_version_t newDefaultVersion = 0;
    SPDLOG_INFO("Updating default version for model:{}, from:{}", getName(), defaultVersion);
    for (const auto& [version, versionInstance] : modelVersions) {
        if (version > newDefaultVersion &&
            ModelVersionState::AVAILABLE == versionInstance->getStatus().getState()) {
            newDefaultVersion = version;
        }
    }
    defaultVersion = newDefaultVersion;
    if (newDefaultVersion) {
        SPDLOG_INFO("Updated default version for model:{}, to:{}", getName(), newDefaultVersion);
    } else {
        SPDLOG_INFO("Model:{} will not have default version since no version is available.", getName());
    }
}

const std::shared_ptr<ModelInstance> Model::getDefaultModelInstance() const {
    std::shared_lock lock(modelVersionsMtx);
    auto defaultVersion = getDefaultVersion();
    const auto modelInstanceIt = modelVersions.find(defaultVersion);

    if (modelVersions.end() == modelInstanceIt) {
        SPDLOG_WARN("Default version:{} for model:{} not found", defaultVersion, getName());
        return nullptr;
    }
    return modelInstanceIt->second;
}

std::shared_ptr<ovms::ModelInstance> Model::modelInstanceFactory(const std::string& modelName, const model_version_t modelVersion) {
    SPDLOG_DEBUG("Producing new ModelInstance");
    return std::move(std::make_shared<ModelInstance>(modelName, modelVersion));
}

Status Model::addVersion(const ModelConfig& config) {
    const auto& version = config.getVersion();
    std::shared_ptr<ModelInstance> modelInstance = modelInstanceFactory(config.getName(), version);
    std::shared_ptr<ModelInstance> modelInstance = modelInstanceFactory();
    if (config.isCustomLoaderRequiredToLoadModel()) {
        // if model requires custom loader, set the custom loader interface pointer object to modelInstance
        modelInstance->setCustomLoaderInterfacePtr(customLoaderInterfacePtr);
    }
    auto status = modelInstance->loadModel(config);
    if (!status.ok()) {
        return status;
    }
    std::unique_lock lock(modelVersionsMtx);
    modelVersions[version] = std::move(modelInstance);
    lock.unlock();
    updateDefaultVersion();
    subscriptionManager.notifySubscribers();
    return StatusCode::OK;
}

Status Model::addVersions(std::shared_ptr<model_versions_t> versionsToStart, ovms::ModelConfig& config) {
    Status result = StatusCode::OK;
    for (const auto version : *versionsToStart) {
        SPDLOG_INFO("Will add model: {}; version: {} ...", getName(), version);
        config.setVersion(version);
        config.parseModelMapping();
        auto status = addVersion(config);
        if (!status.ok()) {
            SPDLOG_ERROR("Error occurred while loading model: {}; version: {}; error: {}",
                getName(),
                version,
                status.string());
            result = status;
        }
    }
    return result;
}

Status Model::retireVersions(std::shared_ptr<model_versions_t> versionsToRetire) {
    Status result = StatusCode::OK;
    for (const auto version : *versionsToRetire) {
        SPDLOG_INFO("Will unload model: {}; version: {} ...", getName(), version);
        auto modelVersion = getModelInstanceByVersion(version);
        if (!modelVersion) {
            Status status = StatusCode::UNKNOWN_ERROR;
            SPDLOG_ERROR("Error occurred while unloading model: {}; version: {}; error: {}",
                getName(),
                version,
                status.string());
            result = status;
            continue;
        }
        modelVersion->unloadModel();
        updateDefaultVersion();
    }
    subscriptionManager.notifySubscribers();
    return result;
}

void Model::retireAllVersions() {
    for (const auto versionModelInstancePair : modelVersions) {
        SPDLOG_INFO("Will unload model: {}; version: {} ...", getName(), versionModelInstancePair.first);
        versionModelInstancePair.second->unloadModel();
        updateDefaultVersion();
    }
    subscriptionManager.notifySubscribers();
}

Status Model::reloadVersions(std::shared_ptr<model_versions_t> versionsToReload, ovms::ModelConfig& config) {
    Status result = StatusCode::OK;
    for (const auto version : *versionsToReload) {
        SPDLOG_INFO("Will reload model: {}; version: {} ...", getName(), version);
        config.setVersion(version);
        auto status = config.parseModelMapping();
        if ((!status.ok()) && (status != StatusCode::FILE_INVALID)) {
            SPDLOG_ERROR("Error while parsing model mapping for model {}", status.string());
        }

        auto modelVersion = getModelInstanceByVersion(version);
        if (!modelVersion) {
            SPDLOG_ERROR("Error occurred while reloading model: {}; version: {}; error: {}",
                getName(),
                version,
                status.string());
            result = StatusCode::UNKNOWN_ERROR;
            continue;
        }
        status = modelVersion->reloadModel(config);
        if (!status.ok()) {
            SPDLOG_ERROR("Error occurred while loading model: {}; version: {}; error: {}",
                getName(),
                version,
                status.string());
            result = status;
            continue;
        }
        updateDefaultVersion();
    }
    subscriptionManager.notifySubscribers();
    return result;
}

}  // namespace ovms

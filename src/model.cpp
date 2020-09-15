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
#include <utility>

namespace ovms {
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
    spdlog::info("Updating default version for model:{}, from:{}", getName(), defaultVersion);
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
        return nullptr;
    }
    return modelInstanceIt->second;
}

std::shared_ptr<ovms::ModelInstance> Model::modelInstanceFactory() {
    SPDLOG_DEBUG("Producing new ModelInstance");
    return std::move(std::make_shared<ModelInstance>());
}

Status Model::addVersion(const ModelConfig& config) {
    std::shared_ptr<ModelInstance> modelInstance = modelInstanceFactory();
    auto status = modelInstance->loadModel(config);
    if (!status.ok()) {
        return status;
    }
    const auto& version = config.getVersion();
    std::unique_lock lock(modelVersionsMtx);
    modelVersions[version] = std::move(modelInstance);
    lock.unlock();
    updateDefaultVersion();
    return StatusCode::OK;
}

Status Model::addVersions(std::shared_ptr<model_versions_t> versionsToStart, ovms::ModelConfig& config) {
    Status status;
    for (const auto version : *versionsToStart) {
        spdlog::info("Will add model: {}; version: {} ...", getName(), version);
        config.setVersion(version);
        config.parseModelMapping();
        status = addVersion(config);
        if (!status.ok()) {
            spdlog::error("Error occurred while loading model: {}; version: {}; error: {}",
                getName(),
                version,
                status.string());
            return status;
        }
    }
    return status;
}

Status Model::retireVersions(std::shared_ptr<model_versions_t> versionsToRetire) {
    Status status;
    for (const auto version : *versionsToRetire) {
        spdlog::info("Will unload model: {}; version: {} ...", getName(), version);
        auto modelVersion = getModelInstanceByVersion(version);
        if (!modelVersion) {
            status = StatusCode::UNKNOWN_ERROR;
            spdlog::error("Error occurred while unloading model: {}; version: {}; error: {}",
                getName(),
                version,
                status.string());
            return status;
        }
        modelVersion->unloadModel();
        updateDefaultVersion();
    }
    return StatusCode::OK;
}

void Model::retireAllVersions() {
    for (const auto versionModelInstancePair : modelVersions) {
        spdlog::info("Will unload model: {}; version: {} ...", getName(), versionModelInstancePair.first);
        versionModelInstancePair.second->unloadModel();
        updateDefaultVersion();
    }
}

Status Model::reloadVersions(std::shared_ptr<model_versions_t> versionsToReload, ovms::ModelConfig& config) {
    Status status;
    for (const auto version : *versionsToReload) {
        spdlog::info("Will reload model: {}; version: {} ...", getName(), version);
        config.setVersion(version);
        status = config.parseModelMapping();
        if ((!status.ok()) && (status != StatusCode::FILE_INVALID)) {
            spdlog::error("Error while parsing model mapping for model {}", status.string());
        }

        auto modelVersion = getModelInstanceByVersion(version);
        if (!modelVersion) {
            spdlog::error("Error occurred while reloading model: {}; version: {}; error: {}",
                getName(),
                version,
                status.string());
            return StatusCode::UNKNOWN_ERROR;
        }
        status = modelVersion->reloadModel(config);
        if (!status.ok()) {
            spdlog::error("Error occurred while loading model: {}; version: {}; error: {}",
                getName(),
                version,
                status.string());
            return status;
        }
        updateDefaultVersion();
    }
    return StatusCode::OK;
}

}  // namespace ovms

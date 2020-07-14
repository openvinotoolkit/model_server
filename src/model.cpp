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
    modelVersions[version] = std::move(modelInstance);
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

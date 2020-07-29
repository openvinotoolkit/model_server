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
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "modelinstance.hpp"

namespace ovms {
/**
     * @brief This class represent inference models
     */
class Model {
protected:
    /**
         * @brief Model name
         */
    std::string name;

    /**
         * @brief Holds different versions of model
         */
    std::map<model_version_t, std::shared_ptr<ModelInstance>> modelVersions;

    /**
         * @brief Model default version
         *
         */
    model_version_t defaultVersion = 0;

    /**
         * @brief Get default version
         *
         * @return default version
         */
    const model_version_t getDefaultVersion() const {
        spdlog::debug("Getting default version for model:{}, {}", getName(), defaultVersion);
        return defaultVersion;
    }

    /**
         * @brief Adds a new version of ModelInstance to the list of versions
         *
         * @param config model configuration
         *
         * @return status
         */
    virtual Status addVersion(const ModelConfig& config);

    /**
         * @brief ModelInstances factory
         *
         * @return modelInstance
         */
    virtual std::shared_ptr<ovms::ModelInstance> modelInstanceFactory();

public:
    /**
         * @brief Constructor
         */
    Model(const std::string& name) :
        name(name),
        defaultVersion(0) {}

    /**
         * @brief Destroy the Model object
         * 
         */
    virtual ~Model() {}

    /**
         * @brief Gets the model name
         * 
         * @return model name
         */
    const std::string& getName() const {
        return name;
    }

    /**
         * @brief Gets the default ModelInstance
         *
         * @return ModelInstance
         */
    const std::shared_ptr<ModelInstance> getDefaultModelInstance() const {
        auto defaultVersion = getDefaultVersion();
        const auto modelInstanceIt = modelVersions.find(defaultVersion);

        if (modelVersions.end() == modelInstanceIt) {
            return nullptr;
        }
        return modelVersions.at(defaultVersion);
    }

    /**
         * @brief Gets model versions
         *
         * @return model versions
         */
    const std::map<model_version_t, std::shared_ptr<ModelInstance>>& getModelVersions() const {
        return modelVersions;
    }

    /**
         * @brief Finds ModelInstance with specific version
         *
         * @param version of the model to search for
         *
         * @return specific model version
         */
    const std::shared_ptr<ModelInstance> getModelInstanceByVersion(const model_version_t& version) const {
        auto it = modelVersions.find(version);
        return it != modelVersions.end() ? it->second : nullptr;
    }

    /**
         * @brief Update default version
         */
    void updateDefaultVersion() {
        model_version_t newDefaultVersion = 0;
        spdlog::info("Updating default version for model:{}, from:{}", getName(), getDefaultVersion());
        for (const auto& [version, versionInstance] : getModelVersions()) {
            if (version > newDefaultVersion &&
                ModelVersionState::AVAILABLE == versionInstance->getStatus().getState()) {
                newDefaultVersion = version;
            }
        }
        spdlog::info("Updated default version for model:{}, to:{}", getName(), newDefaultVersion);
        defaultVersion = newDefaultVersion;
    }

    /**
         * @brief Adds new versions of ModelInstance
         *
         * @param config model configuration
         *
         * @return status
         */
    Status addVersions(std::shared_ptr<model_versions_t> versions, ovms::ModelConfig& config);

    /**
         * @brief Retires versions of Model
         *
         * @param config model configuration
         *
         * @return status
         */
    Status retireVersions(std::shared_ptr<model_versions_t> versions);

    /**
         * @brief Retires all versions of Model
         */
    void retireAllVersions();

    /**
         * @brief Reloads versions of Model
         *
         * @param config model configuration
         *
         * @return status
         */
    Status reloadVersions(std::shared_ptr<model_versions_t> versions, ovms::ModelConfig& config);
};
}  // namespace ovms

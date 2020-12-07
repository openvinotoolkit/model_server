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
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "filesystem.hpp"
#include "modelchangesubscription.hpp"
#include "modelinstance.hpp"

namespace ovms {
class PipelineDefinition;
/*     * @brief This class represent inference models
     */
class Model {
private:
    /**
     * @brief Mutex for protecting concurrent modfying and accessing modelVersions
     */
    mutable std::shared_mutex modelVersionsMtx;

    /**
      * @brief Update default version
      *
      * @param ignoredVersion Version to exclude from being selected as the default version
      */
    void updateDefaultVersion(int ignoredVersion = 0);

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
        SPDLOG_DEBUG("Getting default version for model: {}, {}", getName(), defaultVersion);
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
    virtual std::shared_ptr<ovms::ModelInstance> modelInstanceFactory(const std::string& modelName, const model_version_t modelVersion);

    ModelChangeSubscription subscriptionManager;

    /**
         * @brief Holds the custom loader interface Name
         *
         */
    std::string customLoaderName;

public:
    /**
         * @brief Constructor
         */
    Model(const std::string& name) :
        name(name),
        defaultVersion(0),
        subscriptionManager(std::string("model: ") + name) {}

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
    const std::shared_ptr<ModelInstance> getDefaultModelInstance() const;

    /**
     * @brief Gets model versions instances
     *
     * @return model versions instances
     */
    const std::map<model_version_t, std::shared_ptr<ModelInstance>>& getModelVersions() const;

    /**
     * @brief Gets model versions instances
     *
     * @return model versions instances
     */
    const std::map<model_version_t, const ModelInstance&> getModelVersionsMapCopy() const;

    /**
         * @brief Finds ModelInstance with specific version
         *
         * @param version of the model to search for
         *
         * @return specific model version
         */
    const std::shared_ptr<ModelInstance> getModelInstanceByVersion(const model_version_t& version) const {
        std::shared_lock lock(modelVersionsMtx);
        auto it = modelVersions.find(version);
        return it != modelVersions.end() ? it->second : nullptr;
    }

    /**
         * @brief Adds new versions of ModelInstance
         *
         * @param config model configuration
         *
         * @return status
         */
    Status addVersions(std::shared_ptr<model_versions_t> versions, ovms::ModelConfig& config, std::shared_ptr<FileSystem>& fs);

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
    Status reloadVersions(std::shared_ptr<model_versions_t> versions, ovms::ModelConfig& config, std::shared_ptr<FileSystem>& fs);

    void subscribe(PipelineDefinition& pd);
    void unsubscribe(PipelineDefinition& pd);
    /**
         * @brief Set the custom loader name
         *
         * @param custom loader name
         *
         */

    bool isAnyVersionSubscribed() const;

    void setCustomLoaderName(const std::string name) {
        customLoaderName = name;
    }

    /**
         * @brief Reset the custom loader name
         *
         */
    void resetCustomLoaderName() {
        customLoaderName.clear();
    }

    /**
     * @brief Delete temporary model files
     *
     */
    static Status cleanupModelTmpFiles(const ModelConfig& config);
};
}  // namespace ovms

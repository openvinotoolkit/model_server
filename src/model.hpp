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
         * @brief Default version of the model
         */
        model_version_t defaultVersion;

        /**
         * @brief Holds different versions of model
         */
        std::map<model_version_t, std::shared_ptr<ModelInstance>> modelVersions;

    public:
        /**
         * @brief A default constructor
         */
        Model() = default;

        /**
         * @brief Gets the model name
         * 
         * @return model name
         */
        const std::string& getName() {
            return name;
        }

        /**
         * @brief Gets model default version
         * 
         * @return default model version
         */
        const model_version_t& getDefaultVersion() {
            return defaultVersion;
        }

        /**
         * @brief Gets the default ModelInstance
         *
         * @return ModelInstance
         */
        const std::shared_ptr<ModelInstance>& getDefaultModelInstance() const {
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
         * @brief Adds a new version of ModelInstance to the list of versions
         *
         * @param config model configuration
         *
         * @return status
         */
        Status addVersion(const ModelConfig& config);

        /**
         * @brief Removes model version from the list
         *
         * @param model version
         *
         * @return status
         */
        Status dropVersion(const model_version_t& version);
    };
}  // namespace ovms

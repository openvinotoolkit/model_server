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

#include <string>
#include <vector>

#include "modelversion.h"

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
            unsigned int defaultVersion;

            /**
             * @brief Model input
             */
            std::vector<std::shared_ptr<ModelVersion>> modelVersions;

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
            unsigned int getDefaultVersion() {
                return defaultVersion;
            }

            /**
             * @brief Gets model versions
             *
             * @return model versions
             */
            const std::vector<std::shared_ptr<ModelVersion>>& getModelVersions() const {
                return modelVersions;
            }

            /**
             * @brief Finds ModelVersion instance with specific version
             *
             * @param version of the model to search for
             *
             * @return specific model version
             */
            std::vector<std::shared_ptr<ModelVersion>>::iterator findModelVersionByVersion(const int64_t version) {
                return std::find_if(modelVersions.begin(), modelVersions.end(), [version](const std::shared_ptr<ModelVersion>& modelVersion) {
                    return modelVersion->getVersion() == version;
                });
            }

            /**
             * @brief Adds a new version of ModelVersion to the list of versions
             * 
             * @param name model name
             * @param path to the model
             * @param backend
             * @param version 
             * @param batchSize
             * @param shape
             *  
             * @return status
             */
            Status addVersion(  const std::string& name,
                                const std::string& path,
                                const std::string& backend,
                                const int64_t version,
                                const size_t batchSize,
                                const std::vector<size_t>& shape);

            /**
             * @brief Removes model version from the list
             * 
             * @param modelVersion object
             * 
             * @return status
             */
            Status dropVersion(const ModelVersion& modelVersion);
    };
} // namespace ovms

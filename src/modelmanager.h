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
#include <thread>

#include "model.h"

namespace ovms {
    /**
     * @brief Model manager is managing the list of model topologies enabled for serving and their versions.
     */
    class ModelManager {
        private:
            /**
             * @brief A default constructor is private
             */
            ModelManager() = default;

            /**
             * @brief Private copying constructor
             */
            ModelManager(const ModelManager&);

            /**
             * @brief A JSON configuration filename
             */
            std::string configFilename;

            /**
             * @brief A collection of models
             */
            std::map<std::string, Model> models;

            /**
             * @brief A thread object used for monitoring changes in config
             */
            std::thread monitor;
        public:
            /**
             * @brief Gets the instance of ModelManager
             */
            static ModelManager& getInstance() {
                static ModelManager instance;

                return instance;
            }

            /**
             * @brief Gets config filename
             * 
             * @return config filename
             */
            const std::string& getConfigFilename() {
                return configFilename;
            }

            /**
             * @brief Gets models collection
             * 
             * @return models collection
             */
            const std::map<std::string, Model>& getModels() {
                return models;
            }

            /**
             * @brief Finds model with specific name
             *
             * @param name of the model to search for
             *
             * @return pointer to Model or nullptr if not found
             */
            Model* findModelByName(const std::string& name) {
                auto it = models.find(name);
                if (it != models.end()) {
                    return &it->second;
                }

                return nullptr;
            }

            /**
             * @brief Starts model manager using provided config
             * 
             * @param filename
             * @return status
             */
            Status start(const std::string& jsonFilename);

            /**
             * @brief Gracefully finish the thread
             * 
             * @return status
             */
            Status join();
    };
}  // namespace ovms

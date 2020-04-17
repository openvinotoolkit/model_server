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

#include <algorithm>
#include <string>

namespace ovms {

using shapesMap = std::map<std::string, std::vector<size_t>>;
using layoutsMap = std::map<std::string, std::string>;
using model_version_t = int64_t;

    /**
     * @brief This class represents model configuration
     */
    class ModelConfig {
    public:
        /**
         * @brief Model name
         */
        std::string name;

        /**
         * @brief Model uri path
         */
        std::string basePath;

        /**
         * @brief Device backend
         */
        std::string backend;

        /**
         * @brief Batch size
         */
        size_t batchSize;

        /**
         * @brief Model version policy
         */
        std::string modelVersionPolicy;

        /**
         * @brief Nireq
         */
        uint64_t nireq;

        /**
         * @brief Plugin config
         */
        std::string pluginConfig;

        /**
         * @brief Shape for single input
         */
        std::vector<size_t> shape;

        /**
         * @brief Map of shapes
         */
        shapesMap shapes;

        /**
         * @brief Map of layouts
         */
        layoutsMap layouts;

        /**
         * @brief Model version
         * 
         */
        model_version_t version;

        /**
         * @brief Construct a new ModelConfig with default values
         */
        ModelConfig() {
            backend = "CPU";
            nireq = 1;
            version = 0;
        }

        /**
         * @brief Parse shapes given as string for backward compatibility with OVMS python version
         */
        void addShapes(std::string) {
        }

        /**
         * @brief Parse layout if given by string
         */
        void addLayouts(std::string) {
        }
    };
}

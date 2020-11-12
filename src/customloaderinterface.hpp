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

#include <iostream>
#include <map>
#include <string>

namespace ovms {

enum class CustomLoaderStatus {
    OK,                /*!< Success */
    MODEL_TYPE_IR,     /*!< When model buffers are returned, they belong to IR model */
    MODEL_TYPE_ONNX,   /*!< When model buffers are returned, they belong to ONXX model */
    MODEL_TYPE_BLOB,   /*!< When model buffers are returned, they belong to Blob */
    MODEL_LOAD_ERROR,  /*!< Error while loading the model */
    MODEL_BLACKLISTED, /*!< Model is blacklisted. Do not load */
    INTERNAL_ERROR     /*!< generic error */
};

/**
     * @brief This class is the custom loader interface base class.
     * Custom Loaders need to implement this interface and define the virtual functions to enable
     * OVMS load a model using a custom loader
     */
class CustomLoaderInterface {
public:
    /**
         * @brief Constructor
         */
    CustomLoaderInterface() {
    }
    /**
         * @brief Destructor
         */
    virtual ~CustomLoaderInterface() {
    }

    /**
         * @brief Initialize the custom loader
         *
         * @param loader config file defined under custom loader config in the config file
         *
         * @return status
         */
    virtual CustomLoaderStatus loaderInit(const std::string& loaderConfigFile) = 0;

    /**
         * @brief Load the model by the custom loader
         *
         * @param model name required to be loaded - defined under model config in the config file
         * @param base path where the required IR files are present
         * @param version of the model
         * @param loader config parameters json as string
         * @param char pointer to the model xml buffer
         * @param length of the model xml buffer
         * @param char pointer to the weights buffer
         * @param length of the weights buffer
         * @return status
         */
    virtual CustomLoaderStatus loadModel(const std::string& modelName,
        const std::string& basePath,
        const int version,
        const std::string& loaderOptions,
        char** xmlBuffer, int* xmlLen,
        char** binBuffer, int* binLen) = 0;

    /**
         * @brief Get the model black list status
         *
         * @param model name for which black list status is required
         * @param version for which the black list status is required
         * @return blacklist status
         */
    virtual CustomLoaderStatus getModelBlacklistStatus(const std::string& modelName, int version) {
        return CustomLoaderStatus::OK;
    }

    /**
         * @brief Unload model resources by custom loader once model is unloaded by OVMS
         *
         * @param model name which is been unloaded
         * @param version which is been unloaded
         * @return status
         */
    virtual CustomLoaderStatus unloadModel(const std::string& modelName, int version) = 0;

    /**
         * @brief Retire the model from customloader when OVMS retires the model
         *
         * @param model name which is being retired
         * @return status
         */
    virtual CustomLoaderStatus retireModel(const std::string& modelName) = 0;

    /**
         * @brief Deinitialize the custom loader
         *
         */
    virtual CustomLoaderStatus loaderDeInit() = 0;
};

// the types of the class factories
typedef CustomLoaderInterface* createCustomLoader_t();

}  // namespace ovms

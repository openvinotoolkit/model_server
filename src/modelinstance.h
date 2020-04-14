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

#include <functional>
#include <string>
#include <vector>

#include <inference_engine.hpp>

#include "tensorinfo.h"
#include "status.h"

namespace ovms {

    using tensorMap = std::map<std::string, std::shared_ptr<TensorInfo>>;
    using model_version_t = int64_t;
    using shapesMap = std::map<std::string, std::vector<size_t>>;
    using layoutsMap = std::map<std::string, std::string>;

    /**
     * @brief This class contains all the information about inference engine model
     */
    class ModelInstance {
        protected:
            /**
             * @brief Inference Engine core object
             */
            InferenceEngine::Core engine;

            /**
             * @brief Inference Engine CNNNetwork object
             */
            InferenceEngine::CNNNetwork network;

            /**
             * @brief Inference Engine device network
             */
            InferenceEngine::ExecutableNetwork execNetwork;

            /**
             * @brief A path for the model
             */
            std::string path;

            /**
             * @brief A model version
             */
            model_version_t version;

            /**
             * @brief A backend to run model
             */
            std::string backend;

            /**
             * @brief Model batch size
             */
            size_t batchSize;

        private:
            /**
             * @brief Holds the information about inputs and it's parameters
             */
            tensorMap inputsInfo;

            /**
             * @brief Holds the information about outputs and it's parameters
             */
            tensorMap outputsInfo;

            /**
             * @brief Inference request object created during network load
             */
            InferenceEngine::InferRequest request;

            /**
             * @brief Internal method for loading inputs/outputs
             * 
             * @param map 
             * @param tensors 
             * @param shapes 
             * @param layouts 
             */
            template<typename T>
            void loadTensors(tensorMap& map,
                            const T& tensors,
                            const shapesMap& shapes,
                            const layoutsMap& layouts);
        public:
            /**
             * @brief A default constructor
             */
            ModelInstance() = default;

            /**
             * @brief Gets Inference Engine reference
             * 
             * @return InferenceEngine::Core
             */
            const InferenceEngine::Core& getInferenceEngine() {
                return engine;
            }

            /**
             * @brief Gets Inference Engine ICNNNetwork reference
             * 
             * @return InferenceEngine::CNNNetwork
             */
            const InferenceEngine::CNNNetwork& getCNNNetwork() {
                return network;
            }

            /**
             * @brief Gets Inference Engine Executable Network reference
             * 
             * @return InferenceEngine::ExecutableNetwork
             */
            const InferenceEngine::ExecutableNetwork& getExecutableNetwork() {
                return execNetwork;
            }

            /**
             * @brief Gets path for the model
             * 
             * @return path
             */
            const std::string& getPath() {
                return path;
            }

            /**
             * @brief Gets version
             * 
             * @return version
             */
            const model_version_t& getVersion() {
                return version;
            }

            /**
             * @brief Gets executing backend enma
             * 
             * @return backend name
             */
            const std::string& getBackend() {
                return backend;
            }

            /**
             * @brief Gets batch size
             * 
             * @return batch size
             */
            size_t getBatchSize() {
                return batchSize;
            }

            /**
             * @brief Get the Inputs Info object
             * 
             * @return const tensorMap& 
             */
            const tensorMap& getInputsInfo() {
                return inputsInfo;
            }

            /**
             * @brief Get the Outputs Info object
             * 
             * @return const tensorMap& 
             */
            const tensorMap& getOutputsInfo() {
                return outputsInfo;
            }

            /**
             * @brief Loads model version, reads CNN network model from files (*.xml and *.bin files) and creates inference engine
             * 
             * shapes and layouts are optional, in case we want to override those read from network
             * 
             * @param path 
             * @param backend 
             * @param version 
             * @param batchSize 
             * @param shapes 
             * @param layouts
             * @return Status 
             */
            Status loadModel(const std::string& path,
                             const std::string& backend,
                             const model_version_t& version,
                             const size_t batchSize,
                             const shapesMap& shapes = {},
                             const layoutsMap& layouts = {});

            /**
             * @brief Execute inference on provided data and input name
             * 
             * @param inputName 
             * @param data 
             * @return InferenceEngine::InferRequest& 
             */
            InferenceEngine::InferRequest& infer(const std::string& inputName, const InferenceEngine::Blob::Ptr data);

            /**
             * @brief Execute inference async on provided data and inputName
             * 
             * @param inputName 
             * @param data 
             * @param callback 
             * @return InferenceEngine::InferRequest& 
             */
            InferenceEngine::InferRequest& inferAsync(const std::string& inputName,
                                                      const InferenceEngine::Blob::Ptr data,
                                                      const std::function<void()>& callback);
    };
}  // namespace ovms

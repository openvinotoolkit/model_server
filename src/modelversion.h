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

#include "status.h"

namespace ovms {

    /**
     * @brief This class contains all the information about inference engine model
     */
    class ModelVersion {
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
            int64_t version;

            /**
             * @brief A backend to run model
             */
            std::string backend;

            /**
             * @brief Model batch size
             */
            size_t batchSize;

            /**
             * @brief Model input
             */
            std::vector<size_t> shape;
        private:
            /**
             * @brief Model input name read from network
             */
            std::string inputName;

             /**
             * @brief Model output name read from network
             */
            std::string outputName;

            /**
             * @brief Inference request object created during network load
             */
            InferenceEngine::InferRequest request;
        public:
            /**
             * @brief A default constructor
             */
            ModelVersion() = default;

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
            const int64_t getVersion() {
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
             * @brief Gets model shape
             *
             * @return model shape
             */
            const std::vector<size_t>& getShape() {
                return shape;
            }

            /**
             * @brief Loads model version, reads CNN network model from files (*.xml and *.bin files) and creates inference engine
             * 
             * @param name of the model
             * @param path to model *.xml and *.bin files
             * @param version model
             * @return Status 
             */
            Status loadModel(const std::string& path,
                             const std::string& backend,
                             const int64_t version,
                             const size_t batchSize,
                             const std::vector<size_t>& shape);

            /**
             * @brief Execute inference on provided data
             * 
             * @param data input Blob pointer
             * @return output Blob pointer
             */
            const InferenceEngine::Blob::Ptr infer(const InferenceEngine::Blob::Ptr data);

            /**
             * @brief Execute inference async on provided data
             * 
             * @param data input Blob pointer
             * @param callback function on inference completion
             * @return InferRequest
             */
            const InferenceEngine::InferRequest& inferAsync(const InferenceEngine::Blob::Ptr data, std::function<void()> callback);
    };
}  // namespace ovms

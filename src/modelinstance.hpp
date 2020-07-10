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

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <inference_engine.hpp>
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "modelconfig.hpp"
#include "ovinferrequestsqueue.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"
#include "modelversionstatus.hpp"

namespace ovms {

    using tensor_map_t = std::map<std::string, std::shared_ptr<TensorInfo>>;

    class ModelInstancePredictRequestsHandlesCountGuard;
    /**
     * @brief This class contains all the information about inference engine model
     */
    class ModelInstance {
    protected:
        /**
         * @brief Inference Engine core object
         */
        std::unique_ptr<InferenceEngine::Core> engine;

        /**
         * @brief Inference Engine CNNNetwork object
         */
        std::unique_ptr<InferenceEngine::CNNNetwork> network;

        /**
         * @brief Inference Engine device network
         */
        std::shared_ptr<InferenceEngine::ExecutableNetwork> execNetwork;

        /**
         * @brief Model name
         */
        std::string name;

        /**
         * @brief A path for the model
         */
        std::string path;

        /**
         * @brief A model version
         */
        model_version_t version = -1;

        /**
         * @brief A model status
         */
        ModelVersionStatus status;

        /**
         * @brief A backend to run model
         */
        std::string backend;

        /**
         * @brief Model batch size
         */
        size_t batchSize = 0;

        /**
         * @brief Load OV CNNNetwork ptr
         *
         * @return CNNNetwork ptr
         */
        virtual std::unique_ptr<InferenceEngine::CNNNetwork> loadOVCNNNetworkPtr(const std::string& modelFile);

        /**
         * @brief Load OV Engine
         */
        void loadOVEngine();

        /**
         * @brief Loads OV CNNNetwork
         *
         * @return Status
         */
        Status loadOVCNNNetwork();

        /**
         * @brief Sets OV ExecutableNetworkPtr
         */
        virtual void loadExecutableNetworkPtr(const plugin_config_t& pluginConfig);

        /**
         * @brief Loads OV ExecutableNetwork
         *
         * @return Status
         */
        Status loadOVExecutableNetwork(plugin_config_t pluginConfig);

        /**
         * @brief Prepares inferenceRequestsQueue
         */
        void prepareInferenceRequestsQueue();

        /**
         * @brief Fetch model file paths
         *
         * @return Status
         */
        Status fetchModelFilepaths();

        /**
         * @brief Find file path with extension in model path
         *
         * @return Returns filename with desired extension if exists otherwise empty string
         */
        std::string findModelFilePathWithExtension(const std::string& extension) const;

        /**
         * @brief Stores required model files extensions to be able to load model
         */
        static constexpr std::array<const char*, 2> REQUIRED_MODEL_FILES_EXTENSIONS {".bin", ".xml"};

        /**
         * @brief Notifies model instance users who wait for loading
         */
        std::condition_variable modelLoadedNotify;

    private:
        /**
         * @brief Holds the information about inputs and it's parameters
         */
        tensor_map_t inputsInfo;

        /**
         * @brief Holds the information about outputs and it's parameters
         */
        tensor_map_t outputsInfo;

        /**
         * @brief Holds currently loaded model configuration
         */
        ModelConfig config;

        /**
         * @brief Holds model required file names
         */
        std::map<std::string, std::string> modelFiles;

        /**
         * @brief OpenVINO inference execution stream pool
         */
        std::unique_ptr<OVInferRequestsQueue> inferRequestsQueue;

        /**
         * @brief Holds current usage count in predict requests
         * 
         * Needed for gating model unloading.
         */
        std::atomic<uint64_t> predictRequestsHandlesCount = 0;

        /**
         * @brief Lock to disable concurrent modelinstance load/unload/reload
         */
        std::recursive_mutex loadingMutex;

        /**
         * @brief Internal method for loading inputs
         *
         * @param config
         */
        Status loadInputTensors(const ModelConfig& config);

        /**
         * @brief Internal method for loading outputs
         *
         * @param config
         */
        void loadOutputTensors(const ModelConfig& config);

        /**
         * @brief Performs model loading
         *
         * @return status
         */
        Status loadModelImpl(const ModelConfig& config, const size_t predictRequestedBatchSize = 0);

        /**
         * @brief Configures batchsize
         */
        void configureBatchSize(const ModelConfig& config, const size_t predictRequestedBatchSize = 0);

    public:
        /**
         * @brief A default constructor
         */
        ModelInstance() = default;

        /**
         * @brief Destroy the Model Instance object
         */
        virtual ~ModelInstance() = default;

        /**
         * @brief Increases predict requests usage count
         */
        void increasePredictRequestsHandlesCount() {
            ++predictRequestsHandlesCount;
        }

        /**
         * @brief Decreases predict requests usage count
         */
        void decreasePredictRequestsHandlesCount() {
            --predictRequestsHandlesCount;
        }

        /**
         * @brief Gets the model name
         * 
         * @return model name
         */
        virtual const std::string& getName() const {
            return name;
        }

        /**
         * @brief Gets path for the model
         *
         * @return path
         */
        const std::string& getPath() const {
            return path;
        }

        /**
         * @brief Gets version
         *
         * @return version
         */
        virtual model_version_t getVersion() const {
            return version;
        }

        /**
         * @brief Gets model status
         *
         * @return status
         */
        const ModelVersionStatus& getStatus() const {
            return status;
        }

        /**
         * @brief Gets executing backend name
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
        virtual size_t getBatchSize() const {
            return batchSize;
        }

        /**
         * @brief Get the Inputs Info object
         *
         * @return const tensor_map_t& 
         */
        virtual const tensor_map_t& getInputsInfo() const {
            return inputsInfo;
        }

        /**
         * @brief Get the Outputs Info object
         *
         * @return const tensor_map_t& 
         */
        virtual const tensor_map_t& getOutputsInfo() const {
            return outputsInfo;
        }

        /**
         * @brief Check if can unload infer requests
         *
         * @return bool 
         */
        virtual bool canUnloadInstance() const {
            return 0 == predictRequestsHandlesCount;
        }

        /**
         * @brief Get OV streams pool
         * 
         * @return OVStreamsQueue
         */
        OVInferRequestsQueue& getInferRequestsQueue() {
            return *inferRequestsQueue;
        }

        /**
         * @brief Loads model version, reads CNN network model from files (*.xml and *.bin files) and creates inference engine
         *
         * @param config model configuration
         *
         * @return Status
         */
        virtual Status loadModel(const ModelConfig& config);

        /**
         * @brief Reloads model version, reads CNN network model from files (*.xml and *.bin files) and creates inference engine
         *
         * @param config model configuration
         *
         * @return Status
         */
        virtual Status reloadModel(const ModelConfig& config);

        /**
         * @brief Reloads model version with different batch size, reads CNN network model from files (*.xml and *.bin files) and recreates inference engine
         *
         * @param batchSize batch size
         * @param predictHandlesCounterGuard predictHandlesCounterGuardPtr
         *
         * @return Status
         */
        virtual Status reloadModel(size_t batchSize, std::unique_ptr<ModelInstancePredictRequestsHandlesCountGuard>& predictHandlesCounterGuardPtr);

        /**
         * @brief Unloads model version
         *
         */
        virtual void unloadModel();

        /**
         * @brief Wait for model to change to AVAILABLE state
         *
         * @param waitForModelLoadedTimeoutMilliseconds
         * @param predictHandlesCounterGuard
         *
         * @return Status
         */ 
        Status waitForLoaded(const uint waitForModelLoadedTimeoutMilliseconds,
                           std::unique_ptr<ModelInstancePredictRequestsHandlesCountGuard>& predictHandlesCounterGuard);

        const Status validate(const tensorflow::serving::PredictRequest* request);

        static const int WAIT_FOR_MODEL_LOADED_TIMEOUT_MILLISECONDS = 100;
    };

    class ModelInstancePredictRequestsHandlesCountGuard {
    public:
        ModelInstancePredictRequestsHandlesCountGuard() = delete;
        ModelInstancePredictRequestsHandlesCountGuard(ModelInstance& modelInstance) : modelInstance(modelInstance) {
            modelInstance.increasePredictRequestsHandlesCount();
        }
        ~ModelInstancePredictRequestsHandlesCountGuard() {
            modelInstance.decreasePredictRequestsHandlesCount();
        }
    private:
        ModelInstance& modelInstance;
    };
}  // namespace ovms

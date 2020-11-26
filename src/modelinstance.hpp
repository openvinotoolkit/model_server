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
#include <sstream>
#include <string>
#include <vector>

#include <inference_engine.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "customloaderconfig.hpp"
#include "customloaderinterface.hpp"
#include "modelchangesubscription.hpp"
#include "modelconfig.hpp"
#include "modelinstanceunloadguard.hpp"
#include "modelversionstatus.hpp"
#include "ovinferrequestsqueue.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace ovms {

using tensor_map_t = std::map<std::string, std::shared_ptr<TensorInfo>>;

class DynamicModelParameter {
public:
    DynamicModelParameter() :
        batchSize(0),
        shapes({}) {}
    DynamicModelParameter(int batchSize) :
        batchSize(batchSize),
        shapes({}) {}
    DynamicModelParameter(const std::map<std::string, shape_t>& shapes) :
        batchSize(0),
        shapes(shapes) {}

    bool isBatchSizeRequested() const { return batchSize > 0; }
    bool isShapeRequested(const std::string& name) const { return shapes.count(name) && shapes.at(name).size() > 0; }

    int getBatchSize() const { return batchSize; }
    const shape_t& getShape(const std::string& name) const { return shapes.at(name); }

private:
    int batchSize;
    std::map<std::string, shape_t> shapes;
};

class PipelineDefinition;

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
    const std::string name;

    /**
         * @brief A path for the model
         */
    std::string path;

    /**
         * @brief A model version
         */
    const model_version_t version = -1;

    /**
         * @brief A model status
         */
    ModelVersionStatus status;

    /**
         * @brief Target device to run model
         */
    std::string targetDevice;

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
    Status loadOVExecutableNetwork(const ModelConfig& config);

    /**
         * @brief Prepares inferenceRequestsQueue
         */
    Status prepareInferenceRequestsQueue(const ModelConfig& config);

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
      * @brief Stores required openVINO model files extensions to be able to load model
      *        Order is important, first file on the list is passed to LoadNetwork
      */
    static constexpr std::array<const char*, 2> OV_MODEL_FILES_EXTENSIONS{".xml", ".bin"};

    /**
      * @brief Stores required onnx model files extensions to be able to load model
      */
    static constexpr std::array<const char*, 1> ONNX_MODEL_FILES_EXTENSIONS{".onnx"};

    /**
         * @brief Notifies model instance users who wait for loading
         */
    std::condition_variable modelLoadedNotify;

    /**
         * @brief Holds currently loaded model configuration
         */
    ModelConfig config;

    /**
         * @brief Loads OV CNNNetwork Using the Custom Loader
         *
         * @return Status
         */
    Status loadOVCNNNetworkUsingCustomLoader();

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
      * @brief Holds model required file names. First is loaded
      */
    std::vector<std::string> modelFiles;

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
    Status loadInputTensors(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter());

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
    Status loadModelImpl(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter());

    /**
         * @brief Configures batchsize
         */
    void configureBatchSize(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter());

    const Status validatePrecision(const ovms::TensorInfo& networkInput,
        const tensorflow::TensorProto& requestInput);

    const Status validateNumberOfShapeDimensions(const ovms::TensorInfo& networkInput,
        const tensorflow::TensorProto& requestInput);

    const bool checkBatchSizeMismatch(const ovms::TensorInfo& networkInput,
        const tensorflow::TensorProto& requestInput);

    const bool checkShapeMismatch(const ovms::TensorInfo& networkInput,
        const tensorflow::TensorProto& requestInput,
        const Mode& batchingMode);

    const Status validateTensorContentSize(const ovms::TensorInfo& networkInput,
        const tensorflow::TensorProto& requestInput);

    uint32_t getNumOfParallelInferRequests(const ModelConfig& config);
    uint32_t getNumOfParallelInferRequestsUnbounded(const ModelConfig& config);

    /**
         * @brief Reloads model input/output metadata from current state of CNNNetwork
         *
         * @return Status
         */
    Status recoverFromReshapeError();

    /**
         * @brief Recover from any state model is put into when reload is requested
         * 
         * @param status returned from reload operation
         *
         * @return Status
         */
    Status recoverFromReloadingError(const Status& status);

    ModelChangeSubscription subscriptionManager;

public:
    /**
         * @brief A default constructor
         */
    ModelInstance(const std::string& name, model_version_t version) :
        name(name),
        version(version),
        subscriptionManager(std::string("model: ") + name + std::string(" version: ") + std::to_string(version)) {}

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
         * @brief Gets executing target device name
         *
         * @return target device name
         */
    const std::string& getTargetDevice() {
        return targetDevice;
    }

    /**
         * @brief Gets batch size
         *
         * @return batch size
         */
    virtual size_t getBatchSize() const {
        return network->getBatchSize();
    }

    /**
         * @brief Gets model config
         *
         * @return model config
         */
    virtual const ModelConfig& getModelConfig() const {
        return config;
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
         * @brief Combines plugin config from user with default config calculated at runtime
         *
         * @return plugin config
         */
    static plugin_config_t prepareDefaultPluginConfig(const ModelConfig& config);

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
    virtual Status reloadModel(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter());

    /**
         * @brief Reloads model version with different batch size or shape, reads CNN network model from files (*.xml and *.bin files) and recreates inference engine
         *
         * @param batchSize new batch size
         * @param shape new shape
         * @param unloadGuard unloadGuardPtr
         * 
         * @return Status
         */
    virtual Status reloadModel(size_t batchSize, std::map<std::string, shape_t> shape, std::unique_ptr<ModelInstanceUnloadGuard>& unloadGuardPtr);

    /**
         * @brief Unloads model version
         *
         */
    virtual void unloadModel();

    /**
         * @brief Wait for model to change to AVAILABLE state
         *
         * @param waitForModelLoadedTimeoutMilliseconds
         * @param modelInstanceUnloadGuard
         *
         * @return Status
         */
    Status waitForLoaded(const uint waitForModelLoadedTimeoutMilliseconds,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuard);

    void subscribe(PipelineDefinition& pd);

    void unsubscribe(PipelineDefinition& pd);

    const ModelChangeSubscription& getSubscribtionManager() const { return subscriptionManager; }

    const Status validate(const tensorflow::serving::PredictRequest* request);
};
}  // namespace ovms

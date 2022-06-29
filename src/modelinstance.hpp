//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

#include <openvino/openvino.hpp>

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
#include "sequence_processing_spec.hpp"
#include "status.hpp"
#include "tensorinfo.hpp"

namespace inference {
class ModelInferRequest;
class ModelInferResponse;
}  // namespace inference

namespace ovms {

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
     * @brief This class contains all the information about model
     */
class ModelInstance {
protected:
    /**
         * @brief Performs model loading
         *
         * @return status
         */
    virtual Status loadModelImpl(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter());

    /**
         * @brief OpenVINO Runtime Core object reference
         */
    ov::Core& ieCore;

    /**
         * @brief OpenVINO Runtime Model object
         */
    std::shared_ptr<ov::Model> model;

    /**
         * @brief OpenVINO Runtime CompiledModel object
         */
    std::shared_ptr<ov::CompiledModel> compiledModel;

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
         * @brief A model subscription manager
         */
    ModelChangeSubscription subscriptionManager;

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
      * @brief Stores required OpenVINO model files extensions to be able to load model
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
         * @brief Load OV CNNNetwork ptr
         *
         * @return CNNNetwork ptr
         */
    virtual std::shared_ptr<ov::Model> loadOVModelPtr(const std::string& modelFile);

    /**
         * @brief Lock to disable concurrent modelinstance load/unload/reload
         */
    std::recursive_mutex loadingMutex;

    /**
         * @brief Load OV Engine
         */
    void loadOVEngine();

    /**
         * @brief Loads OV CNNNetwork
         *
         * @return Status
         */
    Status loadOVModel();

    /**
         * @brief Sets OV CompiledModelPtr
         */
    virtual void loadCompiledModelPtr(const plugin_config_t& pluginConfig);

    /**
         * @brief Loads OV CompiledModel
         *
         * @return Status
         */
    virtual Status loadOVCompiledModel(const ModelConfig& config);

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

    const Layout getReportedTensorLayout(const ModelConfig& config, const std::string& name, bool isInput);

    /**
         * @brief Find file path with extension in model path
         *
         * @return Returns filename with desired extension if exists otherwise empty string
         */
    std::string findModelFilePathWithExtension(const std::string& extension) const;

    /**
         * @brief Loads OV CNNNetwork Using the Custom Loader
         *
         * @return Status
         */
    Status loadOVModelUsingCustomLoader();

    template <typename RequestType>
    const Status validate(const RequestType* request);

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
         * @brief Internal method for loading tensors
         *
         * @param config
         */
    Status loadTensors(const ModelConfig& config, bool needsToApplyLayoutConfiguration, const DynamicModelParameter& parameter = DynamicModelParameter());

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
    Status loadOutputTensors(const ModelConfig& config);

    /**
      * @brief Flag determining if cache is disabled
      */
    bool cacheDisabled = false;

    /**
         * @brief Configures batchsize
         */
    void configureBatchSize(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter());

    uint32_t getNumOfParallelInferRequests(const ModelConfig& config);
    uint32_t getNumOfParallelInferRequestsUnbounded(const ModelConfig& config);

    /**
         * @brief Recover from any state model is put into when reload is requested
         * 
         * @param status returned from reload operation
         *
         * @return Status
         */
    Status recoverFromReloadingError(const Status& status);

    /**
         * @brief Perform full model reload with dynamic parameter
         * 
         * @param status returned from reload operation
         * @param parameter requested dynamic parameter
         * 
         * @return Status
         */
    Status reshapeWithFullReload(const Status& status, const DynamicModelParameter& parameter);

    /**
      * Variable to tell reload is due to customloader config change
      */
    bool isCustomLoaderConfigChanged;

public:
    /**
         * @brief A default constructor
         */
    ModelInstance(const std::string& name, model_version_t version, ov::Core& ieCore) :
        ieCore(ieCore),
        name(name),
        version(version),
        subscriptionManager(std::string("model: ") + name + std::string(" version: ") + std::to_string(version)),
        status(name, version) { isCustomLoaderConfigChanged = false; }

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
         * @brief sets the flag in model instance indicating change in custom loader configuration.
	 */
    void setCustomLoaderConfigChangeFlag() {
        isCustomLoaderConfigChanged = true;
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
      * @brief Internal method for setting cache options
      */
    Status setCacheOptions(const ModelConfig& config);

    /**
         * @brief Check if cache is disabled
         *
         * @return cache disabled
         */
    const bool isCacheDisabled() {
        return cacheDisabled;
    }

    /**
         * @brief Gets batch size
         *
         * @return batch size
         */
    virtual Dimension getBatchSize() const {
        return Dimension(ov::get_batch(model));
    }

    const size_t getBatchSizeIndex() const;

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
         * @brief Loads model version
         *
         * @param config model configuration
         *
         * @return Status
         */
    virtual Status loadModel(const ModelConfig& config);

    /**
         * @brief Reloads model version
         *
         * @param config model configuration
         *
         * @return Status
         */
    virtual Status reloadModel(const ModelConfig& config, const DynamicModelParameter& parameter = DynamicModelParameter());

    /**
         * @brief Reloads model version with different batch size or shape
         *
         * @param batchSize new batch size
         * @param shape new shape
         * @param unloadGuard unloadGuardPtr
         *
         * @return Status
         */
    virtual Status reloadModel(std::optional<Dimension> batchSize, std::map<std::string, shape_t> shape, std::unique_ptr<ModelInstanceUnloadGuard>& unloadGuardPtr);

    /**
         * @brief Reloads model version if status of request validation indicates there's a need for reshape or batch size change
         *
         * @param validationStatus status of request validation 
         * @param requestProto pointer to the request proto
         * @param modelUnloadGuardPtr unloadGuardPtr
         * 
         * @return Status
         */
    Status reloadModelIfRequired(
        Status validationStatus,
        const std::optional<Dimension>& requestedBatchSize,
        const std::map<std::string, shape_t>& requestedShapes,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr);

    /**
         * @brief Unloads model version
         * @param isPermanent defines if the unload operation should be permanent and should change instance state to End after it is completed
         * otherwise model might be unloaded temporarily so the instance state should be preserved as Loading
         */
    virtual void retireModel(bool isPermanent = true);

    /**
         * @brief Cleans model version that failed to load
         */
    virtual void cleanupFailedLoad();

    void unloadModelComponents();

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

    Status performInference(ov::InferRequest& inferRequest);

    virtual Status infer(const tensorflow::serving::PredictRequest* requestProto,
        tensorflow::serving::PredictResponse* responseProto,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr);
    virtual Status infer(const ::inference::ModelInferRequest* requestProto,
        ::inference::ModelInferResponse* responseProto,
        std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr);
};
}  // namespace ovms

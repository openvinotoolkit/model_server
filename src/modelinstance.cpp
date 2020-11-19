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
#include "modelinstance.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <dirent.h>
#include <spdlog/spdlog.h>
#include <sys/types.h>

#include "config.hpp"
#include "customloaders.hpp"
#include "stringutils.hpp"

using namespace InferenceEngine;

namespace ovms {

const char* CPU_THROUGHPUT_STREAMS = "CPU_THROUGHPUT_STREAMS";
const char* NIREQ = "NIREQ";

const uint MAX_NIREQ_COUNT = 100000;

const int DEFAULT_OV_STREAMS = std::thread::hardware_concurrency() / 4;

const uint UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS = 10;

void ModelInstance::subscribe(PipelineDefinition& pd) {
    subscriptionManager.subscribe(pd);
}

void ModelInstance::unsubscribe(PipelineDefinition& pd) {
    subscriptionManager.unsubscribe(pd);
}
Status ModelInstance::loadInputTensors(const ModelConfig& config, const DynamicModelParameter& parameter) {
    if (config.isShapeAnonymousFixed() && network->getInputsInfo().size() > 1) {
        Status status = StatusCode::ANONYMOUS_FIXED_SHAPE_NOT_ALLOWED;
        SPDLOG_WARN(status.string());
        return status;
    }

    auto networkShapes = network->getInputShapes();
    const auto& networkInputs = network->getInputsInfo();
    bool reshapeRequired = false;
    auto& configShapes = config.getShapes();
    for (const auto& shape : configShapes) {
        if (shape.first == ANONYMOUS_INPUT_NAME) {
            continue;
        }
        if (networkInputs.count(shape.first) == 0) {
            SPDLOG_WARN("Config shape - {} not found in network", shape.first);
            return StatusCode::CONFIG_SHAPE_IS_NOT_IN_NETWORK;
        }
    }
    for (const auto& pair : networkInputs) {
        const auto& name = pair.first;
        auto input = pair.second;

        // Data from network
        auto precision = input->getPrecision();
        auto layout = input->getLayout();
        auto shape = input->getTensorDesc().getDims();

        // Data from config
        if (config.getLayout().size()) {
            // Single layout for all inputs
            layout = TensorInfo::getLayoutFromString(config.getLayout());
        } else if (config.getLayouts().count(name)) {
            // Layout defined for specific input
            layout = TensorInfo::getLayoutFromString(config.getLayouts().at(name));
        }
        input->setLayout(layout);

        if (config.getBatchSize() > 0 || parameter.isBatchSizeRequested()) {
            // leave shape untouched
        } else if (config.isShapeAuto(name) && parameter.isShapeRequested(name)) {
            shape = parameter.getShape(name);
        } else if (config.getShapes().count(name) && config.getShapes().at(name).shape.size()) {
            shape = config.getShapes().at(name).shape;
        } else if (config.getShapes().count(ANONYMOUS_INPUT_NAME) && config.getShapes().at(ANONYMOUS_INPUT_NAME).shape.size()) {
            shape = config.getShapes().at(ANONYMOUS_INPUT_NAME).shape;
        }

        SPDLOG_DEBUG("Network shape - {}; Final shape - {}", TensorInfo::shapeToString(networkShapes[name]), TensorInfo::shapeToString(shape));

        if (networkShapes[name] != shape) {
            reshapeRequired = true;
            networkShapes[name] = shape;
        }

        auto mappingName = config.getMappingInputByKey(name);
        auto tensor = std::make_shared<TensorInfo>(name, mappingName, precision, shape, layout);
        std::string precision_str = tensor->getPrecisionAsString();
        this->inputsInfo[tensor->getMappedName()] = std::move(tensor);
        std::stringstream shape_stream;
        std::copy(shape.begin(), shape.end(), std::ostream_iterator<size_t>(shape_stream, " "));
        SPDLOG_INFO("Input name: {}; mapping_name: {}; shape: {}; precision: {}, layout:{}",
            name, mappingName, shape_stream.str(), precision_str, TensorInfo::getStringFromLayout(input->getLayout()));
    }

    // Update OV model shapes
    if (reshapeRequired) {
        SPDLOG_DEBUG("model: {}, version: {}; reshaping inputs", getName(), getVersion());
        try {
            network->reshape(networkShapes);
        } catch (const InferenceEngine::details::InferenceEngineException& e) {
            SPDLOG_WARN("OV does not support reshaping model: {} with provided shape", getName());
            SPDLOG_DEBUG("Description: {}", e.what());
            return StatusCode::RESHAPE_ERROR;
        }
    } else {
        SPDLOG_DEBUG("model: {}, version: {}; reshaping inputs is not required", getName(), getVersion());
    }

    return StatusCode::OK;
}

void ModelInstance::loadOutputTensors(const ModelConfig& config) {
    for (const auto& pair : network->getOutputsInfo()) {
        const auto& name = pair.first;
        auto output = pair.second;

        // Data from network
        auto precision = output->getPrecision();
        auto layout = output->getLayout();
        auto shape = output->getDims();
        auto mappingName = config.getMappingOutputByKey(name);
        auto tensor = std::make_shared<TensorInfo>(name, mappingName, precision, shape, layout);
        std::string precision_str = tensor->getPrecisionAsString();
        this->outputsInfo[tensor->getMappedName()] = std::move(tensor);
        std::stringstream shape_stream;
        std::copy(shape.begin(), shape.end(), std::ostream_iterator<size_t>(shape_stream, " "));
        SPDLOG_INFO("Output name: {} ; mapping name: {}; shape: {} ; precision: {}, layout:{}",
            name, mappingName, shape_stream.str(), precision_str, TensorInfo::getStringFromLayout(output->getLayout()));
    }
}

// Temporary methods. To be replaces with proper storage class.
bool dirExists(const std::string& path) {
    DIR* dir = opendir(path.c_str());
    if (dir) {
        closedir(dir);
        return true;
    }

    return false;
}

std::string findFilePathWithExtension(const std::string& path, const std::string& extension) {
    struct dirent* entry;
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        SPDLOG_WARN("Failed to opendir: {}", path);
        return std::string();
    }

    while ((entry = readdir(dir)) != nullptr) {
        auto name = std::string(entry->d_name);
        if (endsWith(name, extension)) {
            closedir(dir);
            if (endsWith(name, "/")) {
                return path + name;
            } else {
                return path + '/' + name;
            }
        }
    }
    closedir(dir);

    return std::string();
}

std::string ModelInstance::findModelFilePathWithExtension(const std::string& extension) const {
    return findFilePathWithExtension(path, extension);
}

uint ModelInstance::getNumOfParallelInferRequestsUnbounded(const ModelConfig& modelConfig) {
    uint numberOfParallelInferRequests = 0;
    if (modelConfig.getNireq() > 0) {
        return modelConfig.getNireq();
    }
    auto& ovmsConfig = ovms::Config::instance();
    if (ovmsConfig.nireq() > 0) {
        // nireq is set globally for all models in ovms startup parameters
        return ovmsConfig.nireq();
    }
    std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
    try {
        numberOfParallelInferRequests = execNetwork->GetMetric(key).as<unsigned int>();
    } catch (const details::InferenceEngineException& ex) {
        SPDLOG_WARN("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS with error {}. Using 1 nireq.", ex.what());
        numberOfParallelInferRequests = 1u;
    }
    return numberOfParallelInferRequests;
}

uint ModelInstance::getNumOfParallelInferRequests(const ModelConfig& modelConfig) {
    uint nireq = getNumOfParallelInferRequestsUnbounded(modelConfig);
    if (nireq > MAX_NIREQ_COUNT) {
        SPDLOG_WARN("Invalid nireq because its value was too high:{}. Maximum value:{}", nireq, MAX_NIREQ_COUNT);
        return 0;
    } else if (nireq < 1u) {
        SPDLOG_WARN("Ignored configured nireq because it has to be above 0 and was:{}. Set to 1", nireq);
        return 1u;
    }
    return nireq;
}

void ModelInstance::loadOVEngine() {
    engine = std::make_unique<InferenceEngine::Core>();
}

std::unique_ptr<InferenceEngine::CNNNetwork> ModelInstance::loadOVCNNNetworkPtr(const std::string& modelFile) {
    return std::make_unique<InferenceEngine::CNNNetwork>(engine->ReadNetwork(modelFile));
}

Status ModelInstance::loadOVCNNNetwork() {
    auto& modelFile = modelFiles[0];
    SPDLOG_DEBUG("Try reading model file:{}", modelFile);
    try {
        network = loadOVCNNNetworkPtr(modelFile);
    } catch (std::exception& e) {
        SPDLOG_ERROR("Error:{}; occurred during loading CNNNetwork for model:{} version:{}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

Status ModelInstance::loadOVCNNNetworkUsingCustomLoader() {
    SPDLOG_DEBUG("Try reading model using a custom loader");
    try {
        std::vector<uint8_t> model;
        std::vector<uint8_t> weights;

        SPDLOG_INFO("loading CNNNetwork for model:{} basepath:{} <> {} version:{}", getName(), getPath(), this->config.getBasePath().c_str(), getVersion());

        custom_loader_options_config_t customLoaderOptionsConfig = this->config.getCustomLoaderOptionsConfigMap();
        const std::string loaderName = customLoaderOptionsConfig["loader_name"];

        auto& customloaders = ovms::CustomLoaders::instance();
        auto customLoaderInterfacePtr = customloaders.find(loaderName);
        if (customLoaderInterfacePtr == nullptr) {
            SPDLOG_INFO("Loader {} is not in loaded customloaders list", loaderName);
            throw std::invalid_argument("customloader not exisiting");
        }

        CustomLoaderStatus res = customLoaderInterfacePtr->loadModel(this->config.getName(),
            this->config.getBasePath(),
            getVersion(),
            this->config.getCustomLoaderOptionsConfigStr(), model, weights);

        if ((res == CustomLoaderStatus::MODEL_LOAD_ERROR) || (res == CustomLoaderStatus::INTERNAL_ERROR)) {
            return StatusCode::INTERNAL_ERROR;
        }

        std::string strModel(model.begin(), model.end());

        if (res == CustomLoaderStatus::MODEL_TYPE_IR) {
            network = std::make_unique<InferenceEngine::CNNNetwork>(engine->ReadNetwork(strModel,
                make_shared_blob<uint8_t>({Precision::U8, {weights.size()}, C}, weights.data())));
        } else if (res == CustomLoaderStatus::MODEL_TYPE_ONNX) {
            network = std::make_unique<InferenceEngine::CNNNetwork>(engine->ReadNetwork(strModel, InferenceEngine::Blob::CPtr()));
        } else if (res == CustomLoaderStatus::MODEL_TYPE_BLOB) {
            return StatusCode::INTERNAL_ERROR;
        }
    } catch (std::exception& e) {
        SPDLOG_ERROR("Error:{}; occurred during loading CNNNetwork for model:{} version:{}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

void ModelInstance::loadExecutableNetworkPtr(const plugin_config_t& pluginConfig) {
    execNetwork = std::make_shared<InferenceEngine::ExecutableNetwork>(engine->LoadNetwork(*network, targetDevice, pluginConfig));
}

plugin_config_t ModelInstance::prepareDefaultPluginConfig(const ModelConfig& config) {
    plugin_config_t pluginConfig = config.getPluginConfig();
    // For CPU and GPU, if user did not specify, calculate CPU_THROUGHPUT_STREAMS automatically
    if (config.isDeviceUsed("CPU")) {
        if (pluginConfig.count("CPU_THROUGHPUT_STREAMS") == 0) {
            pluginConfig["CPU_THROUGHPUT_STREAMS"] = "CPU_THROUGHPUT_AUTO";
        }
    }
    if (config.isDeviceUsed("GPU")) {
        if (pluginConfig.count("GPU_THROUGHPUT_STREAMS") == 0) {
            pluginConfig["GPU_THROUGHPUT_STREAMS"] = "GPU_THROUGHPUT_AUTO";
        }
    }
    return pluginConfig;
}

Status ModelInstance::loadOVExecutableNetwork(const ModelConfig& config) {
    plugin_config_t pluginConfig = prepareDefaultPluginConfig(config);
    try {
        loadExecutableNetworkPtr(pluginConfig);
    } catch (std::exception& e) {
        Status status = StatusCode::CANNOT_LOAD_NETWORK_INTO_TARGET_DEVICE;
        SPDLOG_ERROR("{}; error: {}; model:{}; version:{}; device:{}",
            status.string(),
            e.what(),
            getName(),
            getVersion(),
            config.getTargetDevice());
        return StatusCode::CANNOT_LOAD_NETWORK_INTO_TARGET_DEVICE;
    }
    SPDLOG_INFO("Plugin config for device {}:", targetDevice);
    for (const auto pair : pluginConfig) {
        const auto key = pair.first;
        const auto value = pair.second;
        SPDLOG_INFO("{}: {}", key, value);
    }
    return StatusCode::OK;
}

Status ModelInstance::fetchModelFilepaths() {
    if (this->config.isCustomLoaderRequiredToLoadModel()) {
        // not required if the model is loaded using a custom loader and can be returned from here
        return StatusCode::OK;
    }

    SPDLOG_DEBUG("Getting model files from path:{}", path);
    if (!dirExists(path)) {
        SPDLOG_ERROR("Missing model directory {}", path);
        return StatusCode::PATH_INVALID;
    }

    bool found = true;
    for (auto extension : OV_MODEL_FILES_EXTENSIONS) {
        auto file = findModelFilePathWithExtension(extension);
        if (file.empty()) {
            found = false;
        }
        modelFiles.push_back(file);
    }

    if (!found) {
        found = true;
        modelFiles.clear();
        for (auto extension : ONNX_MODEL_FILES_EXTENSIONS) {
            auto file = findModelFilePathWithExtension(extension);
            if (file.empty()) {
                found = false;
            }
            modelFiles.push_back(file);
        }
    }

    if (!found) {
        SPDLOG_ERROR("Could not find file for model:{} version:{} in path:{}", getName(), getVersion(), path);
        return StatusCode::FILE_INVALID;
    }

    return StatusCode::OK;
}

Status ModelInstance::prepareInferenceRequestsQueue(const ModelConfig& config) {
    uint numberOfParallelInferRequests = getNumOfParallelInferRequests(config);
    if (numberOfParallelInferRequests == 0) {
        return Status(StatusCode::INVALID_NIREQ, "Exceeded allowed nireq value");
    }
    inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(*execNetwork, numberOfParallelInferRequests);
    SPDLOG_INFO("Loaded model {}; version: {}; batch size: {}; No of InferRequests: {}",
        getName(),
        getVersion(),
        getBatchSize(),
        numberOfParallelInferRequests);
    return StatusCode::OK;
}

void ModelInstance::configureBatchSize(const ModelConfig& config, const DynamicModelParameter& parameter) {
    if (parameter.isBatchSizeRequested()) {
        network->setBatchSize(parameter.getBatchSize());
    } else if (config.getBatchSize() > 0) {
        network->setBatchSize(config.getBatchSize());
    }
}

Status ModelInstance::loadModelImpl(const ModelConfig& config, const DynamicModelParameter& parameter) {
    subscriptionManager.notifySubscribers();
    this->path = config.getPath();
    this->targetDevice = config.getTargetDevice();
    this->config = config;
    auto status = fetchModelFilepaths();
    if (!status.ok()) {
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return status;
    }
    try {
        if (!this->engine)
            loadOVEngine();
        status = StatusCode::OK;
        if (!this->network) {
            if (this->config.isCustomLoaderRequiredToLoadModel()) {
                // loading the model using the custom loader
                status = loadOVCNNNetworkUsingCustomLoader();
            } else {
                status = loadOVCNNNetwork();
            }
        }

        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }

        configureBatchSize(this->config, parameter);
        status = loadInputTensors(this->config, parameter);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }
        loadOutputTensors(this->config);
        status = loadOVExecutableNetwork(this->config);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }
        status = prepareInferenceRequestsQueue(this->config);
        if (!status.ok()) {
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return status;
        }
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        SPDLOG_ERROR("exception occurred while loading network: {}", e.what());
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return StatusCode::NETWORK_NOT_LOADED;
    }
    this->status.setAvailable();
    modelLoadedNotify.notify_all();
    return status;
}

Status ModelInstance::loadModel(const ModelConfig& config) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    SPDLOG_INFO("Loading model: {}, version: {}, from path: {}, with target device: {} ...",
        config.getName(), config.getVersion(), config.getPath(), config.getTargetDevice());
    if (config.getBatchingMode() == AUTO) {
        SPDLOG_INFO("Batch size mode for model {} is set to auto", config.getName());
    } else if (config.anyShapeSetToAuto()) {
        SPDLOG_INFO("Some inputs shapes for model {} are set to auto", config.getName());
    }
    this->status = ModelVersionStatus(config.getName(), config.getVersion());
    this->status.setLoading();
    return loadModelImpl(config);
}

Status ModelInstance::recoverFromReshapeError() {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setLoading();
    if (!canUnloadInstance()) {
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        SPDLOG_ERROR("Cannot recover model (name:{}; version:{}) from reshape error, inferences are still in progress", getName(), getVersion());
        return Status(StatusCode::INTERNAL_ERROR, "cannot recover model");
    }
    auto status = this->loadInputTensors(this->config);
    if (!status.ok()) {
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return status;
    }
    this->loadOutputTensors(this->config);
    this->status.setAvailable();
    this->modelLoadedNotify.notify_all();
    return StatusCode::OK;
}

Status ModelInstance::reloadModel(const ModelConfig& config, const DynamicModelParameter& parameter) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setLoading();
    while (!canUnloadInstance()) {
        SPDLOG_INFO("Waiting to reload model: {} version: {}. Blocked by: {} inferences in progress.",
            getName(), getVersion(), predictRequestsHandlesCount);
        std::this_thread::sleep_for(std::chrono::milliseconds(UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS));
    }
    return loadModelImpl(config, parameter);
}

Status ModelInstance::recoverFromReloadingError(const Status& status) {
    if (status == StatusCode::RESHAPE_ERROR) {
        auto recoveryStatus = this->recoverFromReshapeError();
        if (!recoveryStatus.ok()) {
            return recoveryStatus;
        }
        return status;
    }
    SPDLOG_WARN("Failed to reload model:{} version:{} with error:{}. Reloading to previous configuration",
        getName(), getVersion(), status.string());
    auto recoveryStatus = reloadModel(config);
    if (!recoveryStatus.ok()) {
        SPDLOG_WARN("Failed to reload model:{} version:{} to previous configuration with error:{}",
            getName(), getVersion(), recoveryStatus.string());
    }
    return status;
}

Status ModelInstance::reloadModel(size_t batchSize, std::map<std::string, shape_t> requestShapes, std::unique_ptr<ModelInstanceUnloadGuard>& unloadGuard) {
    // temporarily release current predictRequest lock on model loading
    unloadGuard.reset();
    // block concurrent requests for reloading/unloading - assure that after reload predict request
    // will block further requests for reloading/unloading until inference is performed
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    SPDLOG_INFO("Will reload model:{} version:{}", getName(), getVersion());

    DynamicModelParameter parameter;
    if (batchSize > 0) {
        parameter = DynamicModelParameter(batchSize);
    } else if (requestShapes.size() > 0) {
        parameter = DynamicModelParameter(requestShapes);
    } else {
        SPDLOG_DEBUG("Error: requested model:{} version:{} reload with no batchsize and shapes set.", getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }

    auto status = reloadModel(config, parameter);
    if (!status.ok()) {
        return this->recoverFromReloadingError(status);
    } else {
        unloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
    }
    return status;
}

Status ModelInstance::waitForLoaded(const uint waitForModelLoadedTimeoutMilliseconds,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuard) {
    // order is important here for performance reasons
    // assumption: model is already loaded for most of the calls
    modelInstanceUnloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
    if (getStatus().getState() == ModelVersionState::AVAILABLE) {
        SPDLOG_DEBUG("Model:{}, version:{} already loaded", getName(), getVersion());
        return StatusCode::OK;
    }
    SPDLOG_INFO("Model:{} version:{} is still loading", getName(), getVersion());
    modelInstanceUnloadGuard.reset();

    // wait several time since no guarantee that cv wakeup will be triggered before calling wait_for
    const uint waitLoadedTimestepMilliseconds = 100;
    const uint waitCheckpoints = waitForModelLoadedTimeoutMilliseconds / waitLoadedTimestepMilliseconds;
    uint waitCheckpointsCounter = waitCheckpoints;
    SPDLOG_INFO("Waiting for loaded state for model:{} version:{} with timestep:{} timeout:{} check count:{}", getName(), getVersion(),
        waitLoadedTimestepMilliseconds, waitForModelLoadedTimeoutMilliseconds, waitCheckpointsCounter);
    std::mutex cv_mtx;
    std::unique_lock<std::mutex> cv_lock(cv_mtx);
    while (waitCheckpointsCounter-- > 0) {
        if (modelLoadedNotify.wait_for(cv_lock,
                std::chrono::milliseconds(waitLoadedTimestepMilliseconds),
                [this]() {
                    return this->getStatus().getState() > ModelVersionState::LOADING;
                })) {
            SPDLOG_INFO("Waiting for model:{} version:{} loaded state for:{} time",
                getName(), getVersion(), waitCheckpoints - waitCheckpointsCounter);
        }
        modelInstanceUnloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
        if (getStatus().getState() == ModelVersionState::AVAILABLE) {
            SPDLOG_INFO("Succesfully waited for model:{}, version:{}", getName(), getVersion());
            return StatusCode::OK;
        }
        modelInstanceUnloadGuard.reset();
        if (ModelVersionState::AVAILABLE < getStatus().getState()) {
            SPDLOG_INFO("Stopped waiting for model:{} version:{} since it is unloading.", getName(), getVersion());
            return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
        }
    }
    SPDLOG_INFO("Waiting for loaded state reached timeout for model:{} version:{}",
        getName(), getVersion());
    if (getStatus().getState() > ModelVersionState::AVAILABLE) {
        SPDLOG_DEBUG("Waiting for model:{}, version:{} ended since it started unloading.", getName(), getVersion());
        return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
    } else {
        SPDLOG_DEBUG("Waiting for model:{}, version:{} ended due to timeout.", getName(), getVersion());
        return StatusCode::MODEL_VERSION_NOT_LOADED_YET;
    }
}

void ModelInstance::unloadModel() {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setUnloading();
    while (!canUnloadInstance()) {
        SPDLOG_DEBUG("Waiting to unload model:{} version:{}. Blocked by:{} inferences in progres.",
            getName(), getVersion(), predictRequestsHandlesCount);
        std::this_thread::sleep_for(std::chrono::milliseconds(UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS));
    }
    inferRequestsQueue.reset();
    execNetwork.reset();
    network.reset();
    engine.reset();
    outputsInfo.clear();
    inputsInfo.clear();
    modelFiles.clear();
    status.setEnd();

    if (this->config.isCustomLoaderRequiredToLoadModel()) {
        custom_loader_options_config_t customLoaderOptionsConfig = this->config.getCustomLoaderOptionsConfigMap();
        const std::string loaderName = customLoaderOptionsConfig["loader_name"];
        auto& customloaders = ovms::CustomLoaders::instance();
        auto customLoaderInterfacePtr = customloaders.find(loaderName);
        if (customLoaderInterfacePtr == nullptr) {
            SPDLOG_INFO("The loader {} is no longer available", loaderName);
        } else {
            // once model is unloaded, notify custom loader object about the unload
            customLoaderInterfacePtr->unloadModel(getName(), getVersion());
        }
    }
}

const Status ModelInstance::validatePrecision(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    // Network and request must have the same precision
    if (requestInput.dtype() != networkInput.getPrecisionAsDataType()) {
        std::stringstream ss;
        ss << "Expected: " << networkInput.getPrecisionAsString()
           << "; Actual: " << TensorInfo::getDataTypeAsString(requestInput.dtype());
        const std::string details = ss.str();
        SPDLOG_DEBUG("[Model:{} version:{}] Invalid precision - {}", getName(), getVersion(), details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }
    return StatusCode::OK;
}

const Status ModelInstance::validateNumberOfShapeDimensions(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    // Network and request must have the same number of shape dimensions, higher than 0
    auto& shape = networkInput.getShape();
    if (requestInput.tensor_shape().dim_size() <= 0 ||
        shape.size() != static_cast<size_t>(requestInput.tensor_shape().dim_size())) {
        std::stringstream ss;
        ss << "Expected: " << TensorInfo::shapeToString(shape)
           << "; Actual: " << TensorInfo::tensorShapeToString(requestInput.tensor_shape());
        const std::string details = ss.str();
        SPDLOG_DEBUG("[Model:{} version:{}] Invalid number of shape dimensions - {}", getName(), getVersion(), details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

const bool ModelInstance::checkBatchSizeMismatch(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    if (static_cast<size_t>(requestInput.tensor_shape().dim(0).size()) != getBatchSize())
        return true;
    return false;
}

const bool ModelInstance::checkShapeMismatch(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput,
    const Mode& batchingMode) {
    // Network and request must have the same shape
    auto& shape = networkInput.getShape();
    int i = (batchingMode == AUTO) ? 1 : 0;  // If batch size is automatic, omit first dimension
    for (; i < requestInput.tensor_shape().dim_size(); i++) {
        if (requestInput.tensor_shape().dim(i).size() < 0 ||
            shape[i] != static_cast<size_t>(requestInput.tensor_shape().dim(i).size())) {
            return true;
        }
    }
    return false;
}

const Status ModelInstance::validateTensorContentSize(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    /*
    int8        data in request.tensor_content
    uint8       data in request.tensor_content
    int16       data in request.tensor_content
    uint16      request.tensor_content is empty, data located in request.int_val
    int32       data in request.tensor_content
    uint32      data in request.tensor_content
    int64       data in request.tensor_content
    uint64      data in request.tensor_content
    float16     request.tensor_content is empty, data located in request.half_val
    float32     data in request.tensor_content
    double      data in request.tensor_content

    _TENSOR_CONTENT_TYPES
    https://github.com/tensorflow/tensorflow/blob/903a6399aab19b549fefd0ead836af644f3d00f8/tensorflow/python/framework/tensor_util.py#L237
*/

    size_t expectedValueCount = 1;
    for (int i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
        expectedValueCount *= requestInput.tensor_shape().dim(i).size();
    }

    // Network expects tensor content size or value count
    if (requestInput.dtype() == tensorflow::DataType::DT_UINT16) {
        if (requestInput.int_val_size() < 0 ||
            expectedValueCount != static_cast<size_t>(requestInput.int_val_size())) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << "; Actual: " << requestInput.int_val_size();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[Model:{} version:{}] Invalid number of values in tensor proto container - {}", getName(), getVersion(), details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else if (requestInput.dtype() == tensorflow::DataType::DT_HALF) {
        if (requestInput.half_val_size() < 0 ||
            expectedValueCount != static_cast<size_t>(requestInput.half_val_size())) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << "; Actual: " << requestInput.half_val_size();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[Model:{} version:{}] Invalid number of values in tensor proto container - {}", getName(), getVersion(), details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else {
        size_t expectedContentSize = expectedValueCount * networkInput.getPrecision().size();
        if (expectedContentSize != requestInput.tensor_content().size()) {
            std::stringstream ss;
            ss << "Expected: " << expectedContentSize << " bytes; Actual: " << requestInput.tensor_content().size() << " bytes";
            const std::string details = ss.str();
            SPDLOG_DEBUG("[Model:{} version:{}] Invalid content size of tensor proto - {}", getName(), getVersion(), details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
    }
    return StatusCode::OK;
}

const Status ModelInstance::validate(const tensorflow::serving::PredictRequest* request) {
    Status finalStatus = StatusCode::OK;

    // Network and request must have the same amount of inputs
    if (request->inputs_size() < 0 || getInputsInfo().size() != static_cast<size_t>(request->inputs_size())) {
        std::stringstream ss;
        ss << "Expected: " << getInputsInfo().size() << "; Actual: " << request->inputs_size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[Model:{} version:{}] Invalid number of inputs - {}", getName(), getVersion(), details);
        return Status(StatusCode::INVALID_NO_OF_INPUTS, details);
    }

    for (const auto& pair : getInputsInfo()) {
        const auto& name = pair.first;
        auto networkInput = pair.second;
        auto it = request->inputs().find(name);

        // Network and request must have the same names of inputs
        if (it == request->inputs().end()) {
            std::stringstream ss;
            ss << "Required input: " << name;
            const std::string details = ss.str();
            SPDLOG_DEBUG("[Model:{} version:{}] Missing input with specific name - {}", getName(), getVersion(), details);
            return Status(StatusCode::INVALID_MISSING_INPUT, details);
        }

        auto& requestInput = it->second;
        Mode batchingMode = getModelConfig().getBatchingMode();
        Mode shapeMode = getModelConfig().isShapeAuto(name) ? AUTO : FIXED;

        auto status = validatePrecision(*networkInput, requestInput);
        if (!status.ok())
            return status;

        status = validateNumberOfShapeDimensions(*networkInput, requestInput);
        if (!status.ok())
            return status;

        if (checkBatchSizeMismatch(*networkInput, requestInput)) {
            if (batchingMode == AUTO) {
                finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
            } else if (shapeMode != AUTO) {
                std::stringstream ss;
                ss << "Expected: " << getBatchSize() << "; Actual: " << requestInput.tensor_shape().dim(0).size();
                const std::string details = ss.str();
                SPDLOG_DEBUG("[Model:{} version:{}] Invalid batch size - {}", getName(), getVersion(), details);
                return Status(StatusCode::INVALID_BATCH_SIZE, details);
            }
        }

        if (checkShapeMismatch(*networkInput, requestInput, batchingMode)) {
            if (shapeMode == AUTO) {
                finalStatus = StatusCode::RESHAPE_REQUIRED;
            } else {
                std::stringstream ss;
                ss << "Expected: " << TensorInfo::shapeToString(networkInput->getShape())
                   << "; Actual: " << TensorInfo::tensorShapeToString(requestInput.tensor_shape());
                const std::string details = ss.str();
                SPDLOG_DEBUG("[Model:{} version:{}] Invalid shape - {}", getName(), getVersion(), details);
                return Status(StatusCode::INVALID_SHAPE, details);
            }
        }

        status = validateTensorContentSize(*networkInput, requestInput);
        if (!status.ok())
            return status;
    }
    return finalStatus;
}
}  // namespace ovms

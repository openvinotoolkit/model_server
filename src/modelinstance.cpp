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
#include <dirent.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <spdlog/spdlog.h>

#include "config.hpp"
#include "modelinstance.hpp"

using namespace InferenceEngine;

namespace ovms {

const char* CPU_THROUGHPUT_STREAMS = "CPU_THROUGHPUT_STREAMS";
const char* NIREQ = "NIREQ";

const int DEFAULT_OV_STREAMS = std::thread::hardware_concurrency() / 4;

const uint UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS = 10;

Status ModelInstance::loadInputTensors(const ModelConfig& config) {
    auto networkShapes = network->getInputShapes();

    for (const auto& pair : network->getInputsInfo()) {
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

        // One shape for all inputs
        if (config.getShape().size()) {
            shape = config.getShape();
        } else if (config.getShapes().count(name)) {
            // Shape for specific input
            shape = config.getShapes().at(name);
        }

        if (config.getBatchSize() > 0) {
            shape[0] = config.getBatchSize();
        }

        networkShapes[name] = shape;
        auto mappingName = config.getMappingInputByKey(name);
        auto tensor = std::make_shared<TensorInfo>(name, mappingName, precision, shape, layout);
        std::string precision_str = tensor->getPrecisionAsString();
        this->inputsInfo[tensor->getMappedName()] = std::move(tensor);
        std::stringstream shape_stream;
        std::copy(shape.begin(), shape.end(), std::ostream_iterator<size_t>(shape_stream, " "));
        spdlog::info("Input name: {}; mapping_name: {}; shape: {}; precision: {}, layout:{}",
            name, mappingName, shape_stream.str(), precision_str, TensorInfo::getStringFromLayout(input->getLayout()));
    }

    // Update OV model shapes
    if (config.isReshapeRequested()) {
        spdlog::debug("model: {}, version: {}; reshaping inputs", getName(), getVersion());
        try {
            network->reshape(networkShapes);
        } catch (const InferenceEngine::details::InferenceEngineException& e) {
            spdlog::error("could not perform reshape on model {}: {}", getName(), e.what());
            return StatusCode::RESHAPE_ERROR;
        }
    } else {
        spdlog::debug("model: {}, version: {}; reshaping inputs is not required", getName(), getVersion());
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
        spdlog::info("Output name: {} ; mapping name: {}; shape: {} ; precision: {}, layout:{}",
            name, mappingName, shape_stream.str(), precision_str, TensorInfo::getStringFromLayout(output->getLayout()));
    }
}

// Temporary methods. To be replaces with proper storage class.
bool dirExists(const std::string& path) {
    DIR *dir = opendir(path.c_str());
    if (dir) {
        closedir(dir);
        return true;
    }

    return false;
}

std::string findFilePathWithExtension(const std::string& path, const std::string& extension) {
    struct dirent *entry;
    DIR *dir = opendir(path.c_str());

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

uint getOVCPUThroughputStreams() {
    const char* environmentVariableBuffer = std::getenv(CPU_THROUGHPUT_STREAMS);
    if (environmentVariableBuffer) {
        auto result = stou32(environmentVariableBuffer);
        if (result && result.value() > 0) {
            return result.value();
        }
    }

    return std::max(std::thread::hardware_concurrency() / 8, 1u);
}

uint getNumberOfParallelInferRequests() {
    const char* environmentVariableBuffer = std::getenv(NIREQ);
    if (environmentVariableBuffer) {
        auto result = stou32(environmentVariableBuffer);
        if (result && result.value() > 0) {
            return result.value();
        }
    }
    auto& config = ovms::Config::instance();
    return std::max(config.nireq(), 1u);
}

void ModelInstance::loadOVEngine() {
    engine = std::make_unique<InferenceEngine::Core>();
}

std::unique_ptr<InferenceEngine::CNNNetwork> ModelInstance::loadOVCNNNetworkPtr(const std::string& modelFile) {
    return std::make_unique<InferenceEngine::CNNNetwork>(engine->ReadNetwork(modelFile));
}

Status ModelInstance::loadOVCNNNetwork() {
    auto& modelFile = modelFiles[".xml"];
    spdlog::debug("Try reading model file:{}", modelFile);
    try {
        network = loadOVCNNNetworkPtr(modelFile);
    } catch (std::exception& e) {
        spdlog::error("Error:{}; occured during loading CNNNetwork for model:{} version:{}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

void ModelInstance::loadExecutableNetworkPtr(const plugin_config_t& pluginConfig) {
    execNetwork = std::make_shared<InferenceEngine::ExecutableNetwork>(engine->LoadNetwork(*network, backend, pluginConfig));
}

Status ModelInstance::loadOVExecutableNetwork(plugin_config_t pluginConfig) {
    if (pluginConfig.count("CPU_THROUGHPUT_STREAMS") == 0) {
        uint ovBackendStreamsCount = getOVCPUThroughputStreams();
        pluginConfig["CPU_THROUGHPUT_STREAMS"] = std::to_string(ovBackendStreamsCount);
    }
    try {
        loadExecutableNetworkPtr(pluginConfig);
    } catch (std::exception& e) {
        spdlog::error("Error:{}; occured during loading ExecutableNetwork for model:{} version:{}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    spdlog::info("Plugin config for device {}:", backend);
    for (const auto pair : pluginConfig) {
        const auto key = pair.first;
        const auto value = pair.second;
        spdlog::info("{}: {}", key, value);
    }
    return StatusCode::OK;
}

Status ModelInstance::fetchModelFilepaths() {
    spdlog::debug("Getting model files from path:{}", path);
    if (!dirExists(path)) {
        spdlog::error("Missing model directory {}", path);
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return StatusCode::PATH_INVALID;
    }
    for (auto extension : REQUIRED_MODEL_FILES_EXTENSIONS) {
        auto file = findModelFilePathWithExtension(extension);
        if (file.empty()) {
            spdlog::error("Could not find *{} file for model:{} version:{} in path:{}", extension, getName(), getVersion(), path);
            return StatusCode::FILE_INVALID;
        }
        modelFiles[extension] = file;
    }
    return StatusCode::OK;
}

void ModelInstance::prepareInferenceRequestsQueue() {
    uint numberOfParallelInferRequests = getNumberOfParallelInferRequests();
    inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(*execNetwork, numberOfParallelInferRequests);
    spdlog::info("Loaded model {}; version: {}; batch size: {}; No of InferRequests: {}",
        getName(),
        getVersion(),
        getBatchSize(),
        numberOfParallelInferRequests);
}

void ModelInstance::configureBatchSize(const ModelConfig& config, const size_t predictRequestedBatchSize) {
    if (0 == predictRequestedBatchSize) {
        batchSize = config.getBatchSize() > 0 ? config.getBatchSize() : network->getBatchSize();
    } else {
        batchSize = predictRequestedBatchSize;
    }
    network->setBatchSize(batchSize);
}

Status ModelInstance::loadModelImpl(const ModelConfig& config, const size_t predictRequestedBatchSize) {
    this->path = config.getPath();
    this->backend = config.getBackend();
    auto status = fetchModelFilepaths();
    if (!status.ok()) {
        return status;
    }
    try {
        loadOVEngine();
        status = loadOVCNNNetwork();
        if (!status.ok()) {
            return status;
        }
        configureBatchSize(config, predictRequestedBatchSize);
        status = loadInputTensors(config);
        if (!status.ok()) {
            return status;
        }
        loadOutputTensors(config);
        status = loadOVExecutableNetwork(config.getPluginConfig());
        if (!status.ok()) {
            return status;
        }
        prepareInferenceRequestsQueue();
    }
    catch (const InferenceEngine::details::InferenceEngineException& e) {
        spdlog::error("exception occurred while loading network: {}", e.what());
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return StatusCode::NETWORK_NOT_LOADED;
    }
    this->status.setAvailable();
    this->config = config;
    modelLoadedNotify.notify_all();
    return status;
}

Status ModelInstance::loadModel(const ModelConfig& config) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    spdlog::info("Loading model:{}, version:{}, from path:{}, with backend:{} ...",
        config.getName(), config.getVersion(), config.getPath(), config.getBackend());
    this->status = ModelVersionStatus(config.getName(), config.getVersion());
    this->status.setLoading();
    this->name = config.getName();
    this->version = config.getVersion();
    return loadModelImpl(config);
}

Status ModelInstance::reloadModel(const ModelConfig& config) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setLoading();
    while (!canUnloadInstance()) {
        SPDLOG_INFO("Waiting to reload model: {} version: {}. Blocked by: {} inferences in progress.",
            getName(), getVersion(), predictRequestsHandlesCount);
        std::this_thread::sleep_for(std::chrono::milliseconds(UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS));
    }
    return loadModelImpl(config);
}

Status ModelInstance::reloadModel(size_t batchSize, std::unique_ptr<ModelInstancePredictRequestsHandlesCountGuard>& predictHandlesCounterGuard) {
    // temporarily release current predictRequest lock on model loading
    predictHandlesCounterGuard.reset();
    // block concurrent requests for reloading/unloading - assure that after reload predict request
    // will block further requests for reloading/unloading until inference is performed
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    SPDLOG_INFO("Will reload model:{} version:{} from batch size:{} to batch size:{}",
        getName(), getVersion(), getBatchSize(), batchSize);
    ModelConfig configWithNewBatchSize = config;
    configWithNewBatchSize.setBatchSize(batchSize);
    ModelConfig oldConfig = config;
    auto status = reloadModel(configWithNewBatchSize);
    if (!status.ok()) {
        SPDLOG_WARN("Failed to reload model:{} version:{} with new batch size:{} with error:{} Reloading to previous batch size:{} ...",
            getName(), getVersion(), batchSize, status.string(), getBatchSize());
        status = reloadModel(oldConfig);
        if (!status.ok()) {
            SPDLOG_ERROR("Failed to reload model:{} version:{} back to previous batch size:{} with error:{}",
                getName(), getVersion(), getBatchSize(), status.string());
        }
    } else {
        predictHandlesCounterGuard = std::make_unique<ModelInstancePredictRequestsHandlesCountGuard>(*this);
    }
    return status;
}

Status ModelInstance::waitForLoaded(const uint waitForModelLoadedTimeoutMilliseconds,
                                  std::unique_ptr<ModelInstancePredictRequestsHandlesCountGuard>& predictHandlesCounterGuard) {
    // order is important here for performance reasons
    // assumption: model is already loaded for most of the calls
    predictHandlesCounterGuard = std::make_unique<ModelInstancePredictRequestsHandlesCountGuard>(*this);
    if (getStatus().getState() == ModelVersionState::AVAILABLE) {
        SPDLOG_INFO("Model:{}, version:{} already loaded", getName(), getVersion());
        return StatusCode::OK;
    }
    SPDLOG_INFO("Model:{} version:{} is still loading", getName(), getVersion());
    predictHandlesCounterGuard.reset();

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
                                       [this](){
                                           return this->getStatus().getState() > ModelVersionState::LOADING;
                                       })) {
            SPDLOG_INFO("Waiting for model:{} version:{} loaded state for:{} time",
                getName(), getVersion(), waitCheckpoints - waitCheckpointsCounter);
        }
        predictHandlesCounterGuard = std::make_unique<ModelInstancePredictRequestsHandlesCountGuard>(*this);
        if (getStatus().getState() == ModelVersionState::AVAILABLE) {
            SPDLOG_INFO("Succesfully waited for model:{}, version:{}", getName(), getVersion());
            return StatusCode::OK;
        }
        predictHandlesCounterGuard.reset();
        if (ModelVersionState::AVAILABLE < getStatus().getState()) {
            SPDLOG_INFO("Stopped waiting for model:{} version:{} since it is unloading.", getName(), getVersion());
            return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
        }
    }
    SPDLOG_INFO("Waiting for loaded state reached timeout for model:{} version:{}",
        getName(), getVersion());
    if (getStatus().getState() > ModelVersionState::AVAILABLE) {
        SPDLOG_ERROR("Waiting for model:{}, version:{} ended since it started unloading.", getName(), getVersion());
        return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
    } else {
        SPDLOG_ERROR("Waiting for model:{}, version:{} ended due to timeout.", getName(), getVersion());
        return StatusCode::MODEL_VERSION_NOT_LOADED_YET;
    }
}

void ModelInstance::unloadModel() {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setUnloading();
    while (!canUnloadInstance()) {
        SPDLOG_INFO("Waiting to unload model:{} version:{}. Blocked by:{} inferences in progres.",
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
}

const Status ModelInstance::validate(const tensorflow::serving::PredictRequest* request) {
    // Network and request must have the same amount of inputs
    if (request->inputs_size() < 0 || getInputsInfo().size() != static_cast<size_t>(request->inputs_size())) {
        SPDLOG_DEBUG("invalid number of inputs: expected {}; actual {}", getInputsInfo().size(), request->inputs_size());
        return StatusCode::INVALID_NO_OF_INPUTS;
    }

    for (const auto& pair : getInputsInfo()) {
        const auto& name = pair.first;
        auto networkInput = pair.second;
        auto it = request->inputs().find(name);

        // Network and request must have the same names of inputs
        if (it == request->inputs().end()) {
            SPDLOG_DEBUG("missing input with specific name: {}", name);
            return StatusCode::INVALID_MISSING_INPUT;
        }

        auto& requestInput = it->second;
        auto& shape = networkInput->getShape();

        // Network and request must have the same number of shape dimensions, higher than 0
        if (requestInput.tensor_shape().dim_size() <= 0 ||
            shape.size() != static_cast<size_t>(requestInput.tensor_shape().dim_size())) {
            std::stringstream stream;
            std::copy(shape.begin(), shape.end(), std::ostream_iterator<size_t>(stream, " "));

            SPDLOG_DEBUG("invalid number of shape dimensions: expected {}; actual {}", stream.str(), requestInput.tensor_shape().DebugString());
            return StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS;
        }

        // First shape must be equal to batch size
        if (static_cast<size_t>(requestInput.tensor_shape().dim(0).size()) != getBatchSize()) {
            SPDLOG_DEBUG("invalid batch size: expected {}; actual {}", getBatchSize(), requestInput.tensor_shape().dim(0).size());
            return StatusCode::INVALID_BATCH_SIZE;
        }

        // Network and request must have the same shape
        for (int i = 1; i < requestInput.tensor_shape().dim_size(); i++) {
            if (requestInput.tensor_shape().dim(i).size() < 0 ||
                shape[i] != static_cast<size_t>(requestInput.tensor_shape().dim(i).size())) {
                std::stringstream stream;
                std::copy(shape.begin(), shape.end(), std::ostream_iterator<size_t>(stream, " "));

                SPDLOG_DEBUG("invalid shape: expected {}; actual {}", stream.str(), requestInput.tensor_shape().DebugString());
                return StatusCode::INVALID_SHAPE;
            }
        }

        // Network and request must have the same precision
        if (requestInput.dtype() != networkInput->getPrecisionAsDataType()) {
            SPDLOG_DEBUG("invalid precision: expected {}; actual {}", networkInput->getPrecisionAsDataType(), requestInput.dtype());
            return StatusCode::INVALID_PRECISION;
        }

        size_t expectedValueCount = std::accumulate(
            networkInput->getShape().begin(),
            networkInput->getShape().end(),
            1,
            std::multiplies<size_t>());

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

        // Network expects tensor content size or value count
        if (requestInput.dtype() == tensorflow::DataType::DT_UINT16) {
            if (requestInput.int_val_size() < 0 ||
                expectedValueCount != static_cast<size_t>(requestInput.int_val_size())) {
                SPDLOG_DEBUG("invalid number of values in tensor proto container: expected {}; actual {}", expectedValueCount, requestInput.int_val_size());
                return StatusCode::INVALID_VALUE_COUNT;
            }
        } else if (requestInput.dtype() == tensorflow::DataType::DT_HALF) {
            if (requestInput.half_val_size() < 0 ||
                expectedValueCount != static_cast<size_t>(requestInput.half_val_size())) {
                SPDLOG_DEBUG("invalid number of values in tensor proto container: expected {}; actual {}", expectedValueCount, requestInput.int_val_size());
                return StatusCode::INVALID_VALUE_COUNT;
            }
        } else {
            size_t expectedContentSize = expectedValueCount * networkInput->getPrecision().size();
            if (expectedContentSize != requestInput.tensor_content().size()) {
                SPDLOG_DEBUG("invalid content size of tensor proto: expected {}B; actual {}B", expectedContentSize, requestInput.tensor_content().size());
                return StatusCode::INVALID_CONTENT_SIZE;
            }
        }
    }
    return StatusCode::OK;
}

}  // namespace ovms

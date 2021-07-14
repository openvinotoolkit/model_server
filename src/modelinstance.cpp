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
#include "deserialization.hpp"
#include "executingstreamidguard.hpp"
#include "filesystem.hpp"
#include "logging.hpp"
#include "ov_utils.hpp"
#include "prediction_service_utils.hpp"
#include "serialization.hpp"
#include "stringutils.hpp"
#include "tensorinfo.hpp"
#include "timer.hpp"

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
    if (!config.getLayout().empty() && network->getInputsInfo().size() > 1) {
        Status status = StatusCode::ANONYMOUS_FIXED_LAYOUT_NOT_ALLOWED;
        SPDLOG_WARN(status.string());
        return status;
    }

    auto networkShapes = network->getInputShapes();
    const auto& networkInputs = network->getInputsInfo();
    bool reshapeRequired = false;
    for (const auto& [name, _] : config.getShapes()) {
        if (name == ANONYMOUS_INPUT_NAME) {
            continue;
        }
        if (networkInputs.count(name) == 0) {
            SPDLOG_WARN("Config shape - {} not found in network", name);
            return StatusCode::CONFIG_SHAPE_IS_NOT_IN_NETWORK;
        }
    }
    for (const auto& [name, _] : config.getLayouts()) {
        if (networkInputs.count(name) == 0 && network->getOutputsInfo().count(name) == 0) {
            SPDLOG_WARN("Config layout - {} not found in network", name);
            return StatusCode::CONFIG_LAYOUT_IS_NOT_IN_NETWORK;
        }
    }

    this->inputsInfo.clear();

    for (const auto& pair : networkInputs) {
        const auto& name = pair.first;
        auto input = pair.second;
        auto shape = input->getTensorDesc().getDims();
        if (config.getBatchSize() > 0 || parameter.isBatchSizeRequested()) {
            // leave shape untouched
        } else if (config.isShapeAuto(name) && parameter.isShapeRequested(name)) {
            shape = parameter.getShape(name);
        } else if (config.getShapes().count(name) && config.getShapes().at(name).shape.size()) {
            shape = config.getShapes().at(name).shape;
        } else if (config.getShapes().count(ANONYMOUS_INPUT_NAME) && config.getShapes().at(ANONYMOUS_INPUT_NAME).shape.size()) {
            shape = config.getShapes().at(ANONYMOUS_INPUT_NAME).shape;
        }

        SPDLOG_DEBUG("Network shape for input: {} - {}; final shape {}", name,
            TensorInfo::shapeToString(networkShapes[name]),
            TensorInfo::shapeToString(shape));

        if (networkShapes[name] != shape) {
            reshapeRequired = true;
            networkShapes[name] = shape;
        }
    }

    // Update OV model shapes
    if (reshapeRequired) {
        SPDLOG_DEBUG("model: {}, version: {}; reshaping inputs", getName(), getVersion());
        try {
            SPDLOG_INFO("Initial network inputs: {}", getNetworkInputsInfoString(networkInputs, config));
            network->reshape(networkShapes);
        } catch (const InferenceEngine::Exception& e) {
            SPDLOG_WARN("OV does not support reshaping model: {} with provided shape", getName());
            SPDLOG_DEBUG("Description: {}", e.what());
            return StatusCode::RESHAPE_ERROR;
        }
    } else {
        SPDLOG_DEBUG("model: {}, version: {}; reshaping inputs is not required", getName(), getVersion());
    }

    for (const auto& pair : networkInputs) {
        const auto& name = pair.first;
        auto input = pair.second;

        // Data from network
        auto precision = input->getPrecision();
        auto layout = input->getLayout();
        auto shape = input->getTensorDesc().getDims();

        if (!config.getLayout().empty()) {
            layout = TensorInfo::getLayoutFromString(config.getLayout());
        } else if (config.getLayouts().size() > 0) {
            auto it = config.getLayouts().find(name);
            if (it != config.getLayouts().end()) {
                layout = TensorInfo::getLayoutFromString(it->second);
            }
        }

        input->setLayout(layout);

        auto mappingName = config.getMappingInputByKey(name);
        auto tensor = std::make_shared<TensorInfo>(name, mappingName, precision, shape, layout);
        this->inputsInfo[tensor->getMappedName()] = std::move(tensor);
    }
    SPDLOG_INFO("Final network inputs: {}", getNetworkInputsInfoString(networkInputs, config));
    return StatusCode::OK;
}

void ModelInstance::loadOutputTensors(const ModelConfig& config) {
    this->outputsInfo.clear();
    for (const auto& pair : network->getOutputsInfo()) {
        const auto& name = pair.first;
        auto output = pair.second;

        // Data from network
        auto precision = output->getPrecision();
        auto layout = output->getLayout();

        if (config.getLayouts().size() > 0) {
            auto it = config.getLayouts().find(name);
            if (it != config.getLayouts().end()) {
                layout = TensorInfo::getLayoutFromString(it->second);
            }
        }

        output->setLayout(layout);

        auto shape = output->getDims();
        auto effectiveShape = output->getTensorDesc().getBlockingDesc().getBlockDims();
        auto mappingName = config.getMappingOutputByKey(name);
        auto tensor = std::make_shared<TensorInfo>(name, mappingName, precision, shape, layout);
        std::string precision_str = tensor->getPrecisionAsString();
        this->outputsInfo[tensor->getMappedName()] = std::move(tensor);
        std::stringstream shape_stream;
        std::copy(shape.begin(), shape.end(), std::ostream_iterator<size_t>(shape_stream, " "));
        std::stringstream effective_shape_stream;
        std::copy(effectiveShape.begin(), effectiveShape.end(), std::ostream_iterator<size_t>(effective_shape_stream, " "));
        SPDLOG_INFO("Output name: {}; mapping name: {}; shape: {}; effective shape {}; precision: {}; layout: {}",
            name, mappingName, shape_stream.str(), effective_shape_stream.str(), precision_str,
            TensorInfo::getStringFromLayout(output->getLayout()));
    }
}

// Temporary methods. To be replaces with proper storage class.
bool dirExists(const std::string& path) {
    if (FileSystem::isPathEscaped(path)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", path);
        return false;
    }
    DIR* dir = opendir(path.c_str());
    if (dir) {
        closedir(dir);
        return true;
    }

    return false;
}

std::string findFilePathWithExtension(const std::string& path, const std::string& extension) {
    struct dirent* entry;
    if (FileSystem::isPathEscaped(path)) {
        SPDLOG_ERROR("Path {} escape with .. is forbidden.", path);
        return std::string();
    }
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
    } catch (const Exception& ex) {
        SPDLOG_WARN("Failed to query OPTIMAL_NUMBER_OF_INFER_REQUESTS with error {}. Using 1 nireq.", ex.what());
        numberOfParallelInferRequests = 1u;
    }
    return numberOfParallelInferRequests;
}

uint ModelInstance::getNumOfParallelInferRequests(const ModelConfig& modelConfig) {
    uint nireq = getNumOfParallelInferRequestsUnbounded(modelConfig);
    if (nireq > MAX_NIREQ_COUNT) {
        SPDLOG_WARN("Invalid nireq because its value was too high: {}. Maximum value: {}", nireq, MAX_NIREQ_COUNT);
        return 0;
    } else if (nireq < 1u) {
        SPDLOG_WARN("Ignored configured nireq because it has to be above 0 and was: {}. Set to 1", nireq);
        return 1u;
    }
    return nireq;
}

void ModelInstance::loadOVEngine() {
    engine = std::make_unique<InferenceEngine::Core>();
    if (ovms::Config::instance().cpuExtensionLibraryPath() != "") {
        SPDLOG_INFO("Loading custom CPU extension from {}", ovms::Config::instance().cpuExtensionLibraryPath());
        try {
            auto extension_ptr = std::make_shared<InferenceEngine::Extension>(ovms::Config::instance().cpuExtensionLibraryPath());
            SPDLOG_INFO("Custom CPU extention loaded. Adding it.");
            engine->AddExtension(extension_ptr, "CPU");
            SPDLOG_INFO("Extention added.");
        } catch (std::exception& ex) {
            SPDLOG_CRITICAL("Custom CPU extention loading has failed! Reason: {}", ex.what());
            throw;
        } catch (...) {
            SPDLOG_CRITICAL("Custom CPU extention loading has failed with an unknown error!");
            throw;
        }
    }
}

std::unique_ptr<InferenceEngine::CNNNetwork> ModelInstance::loadOVCNNNetworkPtr(const std::string& modelFile) {
    return std::make_unique<InferenceEngine::CNNNetwork>(engine->ReadNetwork(modelFile));
}

Status ModelInstance::loadOVCNNNetwork() {
    auto& modelFile = modelFiles[0];
    SPDLOG_DEBUG("Try reading model file: {}", modelFile);
    try {
        network = loadOVCNNNetworkPtr(modelFile);
    } catch (std::exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading CNNNetwork for model: {} version: {}", e.what(), getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }
    return StatusCode::OK;
}

Status ModelInstance::loadOVCNNNetworkUsingCustomLoader() {
    SPDLOG_DEBUG("Try reading model using a custom loader");
    try {
        std::vector<uint8_t> model;
        std::vector<uint8_t> weights;

        SPDLOG_INFO("loading CNNNetwork for model: {} basepath: {} <> {} version: {}", getName(), getPath(), this->config.getBasePath().c_str(), getVersion());

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

        if (res == CustomLoaderStatus::MODEL_LOAD_ERROR) {
            return StatusCode::FILE_INVALID;
        }

        if ((res == CustomLoaderStatus::INTERNAL_ERROR) || (res == CustomLoaderStatus::MODEL_BLACKLISTED)) {
            return StatusCode::INTERNAL_ERROR;
        }

        std::string strModel(model.begin(), model.end());

        if (res == CustomLoaderStatus::MODEL_TYPE_IR) {
            Blob::Ptr blobWts = make_shared_blob<uint8_t>({Precision::U8, {weights.size()}, C});
            blobWts->allocate();
            std::memcpy(blobWts->buffer(), weights.data(), weights.size());
            network = std::make_unique<InferenceEngine::CNNNetwork>(engine->ReadNetwork(strModel, blobWts));
        } else if (res == CustomLoaderStatus::MODEL_TYPE_ONNX) {
            network = std::make_unique<InferenceEngine::CNNNetwork>(engine->ReadNetwork(strModel, InferenceEngine::Blob::CPtr()));
        } else if (res == CustomLoaderStatus::MODEL_TYPE_BLOB) {
            return StatusCode::INTERNAL_ERROR;
        }
    } catch (std::exception& e) {
        SPDLOG_ERROR("Error: {}; occurred during loading CNNNetwork for model: {} version: {}", e.what(), getName(), getVersion());
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
        SPDLOG_ERROR("{}; error: {}; model: {}; version: {}; device: {}",
            status.string(),
            e.what(),
            getName(),
            getVersion(),
            config.getTargetDevice());
        return status;
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

    SPDLOG_DEBUG("Getting model files from path: {}", path);
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
        SPDLOG_ERROR("Could not find file for model: {} version: {} in path: {}", getName(), getVersion(), path);
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
    } catch (const InferenceEngine::Exception& e) {
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

Status ModelInstance::reloadModel(const ModelConfig& config, const DynamicModelParameter& parameter) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setLoading();
    while (!canUnloadInstance()) {
        SPDLOG_INFO("Waiting to reload model: {} version: {}. Blocked by: {} inferences in progress.",
            getName(), getVersion(), predictRequestsHandlesCount);
        std::this_thread::sleep_for(std::chrono::milliseconds(UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS));
    }
    if ((this->config.isCustomLoaderRequiredToLoadModel()) && (isCustomLoaderConfigChanged)) {
        // unloading and the loading back the model
        isCustomLoaderConfigChanged = false;
        retireModel(isCustomLoaderConfigChanged);
    }
    return loadModelImpl(config, parameter);
}

Status ModelInstance::recoverFromReloadingError(const Status& status) {
    SPDLOG_WARN("Failed to perform complete reload with requested dynamic parameter. Model: {} version: {} with error: {}. Reloading to previous configuration",
        getName(), getVersion(), status.string());
    bool changeStatus{false};
    retireModel(changeStatus);

    auto recoveryStatus = reloadModel(config);
    if (!recoveryStatus.ok()) {
        SPDLOG_WARN("Failed to recover model: {} version: {} to previous configuration with error: {}",
            getName(), getVersion(), recoveryStatus.string());
    }
    return status;
}

Status ModelInstance::reshapeWithFullReload(const Status& status, const DynamicModelParameter& parameter) {
    SPDLOG_WARN("Failed to reload model: {} version: {} with error: {}. Trying to perform complete reload with requested dynamic parameter",
        getName(), getVersion(), status.string());
    bool changeStatus{false};
    retireModel(changeStatus);

    auto recoveryStatus = reloadModel(config, parameter);
    if (!recoveryStatus.ok()) {
        SPDLOG_WARN("Failed to reload model: {} version: {} to previous configuration with error: {}",
            getName(), getVersion(), recoveryStatus.string());
    }
    return recoveryStatus;
}

Status ModelInstance::reloadModel(size_t batchSize, std::map<std::string, shape_t> requestShapes, std::unique_ptr<ModelInstanceUnloadGuard>& unloadGuard) {
    // temporarily release current predictRequest lock on model loading
    unloadGuard.reset();
    // block concurrent requests for reloading/unloading - assure that after reload predict request
    // will block further requests for reloading/unloading until inference is performed
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    SPDLOG_INFO("Will reload model: {} version: {}", getName(), getVersion());

    DynamicModelParameter parameter;
    if (batchSize > 0) {
        parameter = DynamicModelParameter(batchSize);
    } else if (requestShapes.size() > 0) {
        parameter = DynamicModelParameter(requestShapes);
    } else {
        SPDLOG_DEBUG("Error: requested model: {} version: {} reload with no batchsize and shapes set.", getName(), getVersion());
        return StatusCode::INTERNAL_ERROR;
    }

    auto status = reloadModel(config, parameter);
    if (!status.ok()) {
        status = this->reshapeWithFullReload(status, parameter);
        if (!status.ok()) {
            return this->recoverFromReloadingError(status);
        }
    }
    unloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
    return status;
}

Status ModelInstance::reloadModelIfRequired(
    Status validationStatus,
    const tensorflow::serving::PredictRequest* requestProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    Status status = validationStatus;
    if (status.batchSizeChangeRequired()) {
        status = reloadModel(getRequestBatchSize(requestProto), {}, modelUnloadGuardPtr);
        if (!status.ok()) {
            SPDLOG_ERROR("Model: {}, version: {} reload (batch size change) failed. Status Code: {}, Error {}",
                getName(), getVersion(), status.getCode(), status.string());
        }
    } else if (status.reshapeRequired()) {
        status = reloadModel(0, getRequestShapes(requestProto), modelUnloadGuardPtr);
        if (!status.ok() && status != StatusCode::RESHAPE_ERROR) {
            SPDLOG_ERROR("Model: {}, version: {} reload (reshape) failed. Status Code: {}, Error: {}",
                getName(), getVersion(), status.getCode(), status.string());
        }
    } else if (!status.ok()) {
        SPDLOG_WARN("Model: {}, version: {} validation of inferRequest failed. Status Code: {}, Error: {}",
            getName(), getVersion(), status.getCode(), status.string());
    }
    return status;
}

Status ModelInstance::waitForLoaded(const uint waitForModelLoadedTimeoutMilliseconds,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuard) {
    // order is important here for performance reasons
    // assumption: model is already loaded for most of the calls
    modelInstanceUnloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
    if (getStatus().getState() == ModelVersionState::AVAILABLE) {
        SPDLOG_DEBUG("Model: {}, version: {} already loaded", getName(), getVersion());
        return StatusCode::OK;
    }
    modelInstanceUnloadGuard.reset();

    // wait several time since no guarantee that cv wakeup will be triggered before calling wait_for
    const uint waitLoadedTimestepMilliseconds = 100;
    const uint waitCheckpoints = waitForModelLoadedTimeoutMilliseconds / waitLoadedTimestepMilliseconds;
    uint waitCheckpointsCounter = waitCheckpoints;
    SPDLOG_DEBUG("Waiting for loaded state for model: {} version: {} with timestep: {} timeout: {} check count: {}", getName(), getVersion(),
        waitLoadedTimestepMilliseconds, waitForModelLoadedTimeoutMilliseconds, waitCheckpointsCounter);
    std::mutex cv_mtx;
    std::unique_lock<std::mutex> cv_lock(cv_mtx);
    while (waitCheckpointsCounter-- > 0) {
        if (modelLoadedNotify.wait_for(cv_lock,
                std::chrono::milliseconds(waitLoadedTimestepMilliseconds),
                [this]() {
                    return this->getStatus().getState() > ModelVersionState::LOADING;
                })) {
            SPDLOG_INFO("Waiting for model: {} version: {} loaded state for: {} time",
                getName(), getVersion(), waitCheckpoints - waitCheckpointsCounter);
        }
        modelInstanceUnloadGuard = std::make_unique<ModelInstanceUnloadGuard>(*this);
        if (getStatus().getState() == ModelVersionState::AVAILABLE) {
            SPDLOG_INFO("Succesfully waited for model: {}, version: {}", getName(), getVersion());
            return StatusCode::OK;
        }
        modelInstanceUnloadGuard.reset();
        if (ModelVersionState::AVAILABLE < getStatus().getState()) {
            SPDLOG_INFO("Stopped waiting for model: {} version: {} since it is unloading.", getName(), getVersion());
            return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
        }
    }
    SPDLOG_INFO("Waiting for loaded state reached timeout for model: {} version: {}",
        getName(), getVersion());
    if (getStatus().getState() > ModelVersionState::AVAILABLE) {
        SPDLOG_DEBUG("Waiting for model: {}, version: {} ended since it started unloading.", getName(), getVersion());
        return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
    } else {
        SPDLOG_DEBUG("Waiting for model: {}, version: {} ended due to timeout.", getName(), getVersion());
        return StatusCode::MODEL_VERSION_NOT_LOADED_YET;
    }
}

void ModelInstance::retireModel(bool isPermanent) {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    if (isPermanent) {
        this->status.setUnloading();
    } else {
        this->status.setLoading();
    }
    unloadModelComponents();
    if (isPermanent) {
        status.setEnd();
    }
}

void ModelInstance::cleanupFailedLoad() {
    std::lock_guard<std::recursive_mutex> loadingLock(loadingMutex);
    this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
    unloadModelComponents();
}

void ModelInstance::unloadModelComponents() {
    subscriptionManager.notifySubscribers();
    while (!canUnloadInstance()) {
        SPDLOG_DEBUG("Waiting to unload model: {} version: {}. Blocked by: {} inferences in progres.",
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

    if (this->config.isCustomLoaderRequiredToLoadModel()) {
        custom_loader_options_config_t customLoaderOptionsConfig = this->config.getCustomLoaderOptionsConfigMap();
        const std::string loaderName = customLoaderOptionsConfig["loader_name"];
        auto& customloaders = ovms::CustomLoaders::instance();
        auto customLoaderInterfacePtr = customloaders.find(loaderName);
        if (customLoaderInterfacePtr == nullptr) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "The loader {} is no longer available for model: {} version : {}",
                loaderName, getName(), getVersion());
        } else {
            // once model is unloaded, notify custom loader object about the unload
            customLoaderInterfacePtr->unloadModel(getName(), getVersion());
        }
    }
}

const Status ModelInstance::checkIfShapeValuesNegative(const tensorflow::TensorProto& requestInput) {
    for (int i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
        if (requestInput.tensor_shape().dim(i).size() < 0) {
            const std::string details = "Negative dimension size is not acceptable: " + TensorInfo::tensorShapeToString(requestInput.tensor_shape());
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "[Model: {} version: {}] Invalid shape - {}", getName(), getVersion(), details);
            return Status(StatusCode::INVALID_SHAPE, details);
        }
    }
    return StatusCode::OK;
}
const Status ModelInstance::validateNumberOfInputs(const tensorflow::serving::PredictRequest* request, const size_t expectedNumberOfInputs) {
    if (request->inputs_size() < 0 || expectedNumberOfInputs != static_cast<size_t>(request->inputs_size())) {
        std::stringstream ss;
        ss << "Expected: " << expectedNumberOfInputs << "; Actual: " << request->inputs_size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[Model:{} version:{}] Invalid number of inputs - {}", getName(), getVersion(), details);
        return Status(StatusCode::INVALID_NO_OF_INPUTS, details);
    }
    return StatusCode::OK;
}

const Status ModelInstance::validatePrecision(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    // Network and request must have the same precision
    if (requestInput.dtype() != networkInput.getPrecisionAsDataType()) {
        std::stringstream ss;
        ss << "Expected: " << networkInput.getPrecisionAsString()
           << "; Actual: " << TensorInfo::getDataTypeAsString(requestInput.dtype());
        const std::string details = ss.str();
        SPDLOG_DEBUG("[Model: {} version: {}] Invalid precision - {}", getName(), getVersion(), details);
        return Status(StatusCode::INVALID_PRECISION, details);
    }
    return StatusCode::OK;
}

const Status ModelInstance::validateNumberOfShapeDimensions(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    // Network and request must have the same number of shape dimensions, higher than 0
    auto& shape = networkInput.getEffectiveShape();
    if (requestInput.tensor_shape().dim_size() <= 0 ||
        shape.size() != static_cast<size_t>(requestInput.tensor_shape().dim_size())) {
        std::stringstream ss;
        ss << "Expected: " << TensorInfo::shapeToString(shape)
           << "; Actual: " << TensorInfo::tensorShapeToString(requestInput.tensor_shape());
        const std::string details = ss.str();
        SPDLOG_DEBUG("[Model: {} version: {}] Invalid number of shape dimensions - {}", getName(), getVersion(), details);
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
    auto& shape = networkInput.getEffectiveShape();
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
            SPDLOG_DEBUG("[Model: {} version: {}] Invalid number of values in tensor proto container - {}", getName(), getVersion(), details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else if (requestInput.dtype() == tensorflow::DataType::DT_HALF) {
        if (requestInput.half_val_size() < 0 ||
            expectedValueCount != static_cast<size_t>(requestInput.half_val_size())) {
            std::stringstream ss;
            ss << "Expected: " << expectedValueCount << "; Actual: " << requestInput.half_val_size();
            const std::string details = ss.str();
            SPDLOG_DEBUG("[Model: {} version: {}] Invalid number of values in tensor proto container - {}", getName(), getVersion(), details);
            return Status(StatusCode::INVALID_VALUE_COUNT, details);
        }
    } else {
        size_t expectedContentSize = expectedValueCount * networkInput.getPrecision().size();
        if (expectedContentSize != requestInput.tensor_content().size()) {
            std::stringstream ss;
            ss << "Expected: " << expectedContentSize << " bytes; Actual: " << requestInput.tensor_content().size() << " bytes";
            const std::string details = ss.str();
            SPDLOG_DEBUG("[Model: {} version: {}] Invalid content size of tensor proto - {}", getName(), getVersion(), details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
    }
    return StatusCode::OK;
}

const bool ModelInstance::checkBinaryInputBatchSizeMismatch(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    if (requestInput.string_val_size() < 0) {
        return true;
    }
    if (getBatchSize() != static_cast<size_t>(requestInput.string_val_size())) {
        return true;
    }
    return false;
}

const Status ModelInstance::validateNumberOfBinaryInputShapeDimensions(const ovms::TensorInfo& networkInput,
    const tensorflow::TensorProto& requestInput) {
    if (requestInput.tensor_shape().dim_size() != 1) {
        std::stringstream ss;
        ss << "Expected number of binary input shape dimensions: 1; Actual: " << requestInput.tensor_shape().dim_size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[Model: {} version: {}] Invalid number of shape dimensions - {}", getName(), getVersion(), details);
        return Status(StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, details);
    }
    return StatusCode::OK;
}

const Status ModelInstance::validate(const tensorflow::serving::PredictRequest* request) {
    Status finalStatus = StatusCode::OK;

    // Network and request must have the same amount of inputs
    auto expectedNumberOfInputs = getInputsInfo().size();
    finalStatus = validateNumberOfInputs(request, expectedNumberOfInputs);
    if (!finalStatus.ok())
        return finalStatus;

    for (const auto& pair : getInputsInfo()) {
        const auto& name = pair.first;
        auto networkInput = pair.second;
        auto it = request->inputs().find(name);

        // Network and request must have the same names of inputs
        if (it == request->inputs().end()) {
            std::stringstream ss;
            ss << "Required input: " << name;
            const std::string details = ss.str();
            SPDLOG_DEBUG("[Model: {} version: {}] Missing input with specific name - {}", getName(), getVersion(), details);
            return Status(StatusCode::INVALID_MISSING_INPUT, details);
        }

        auto& requestInput = it->second;
        Mode batchingMode = getModelConfig().getBatchingMode();
        Mode shapeMode = getModelConfig().isShapeAuto(name) ? AUTO : FIXED;

        auto status = checkIfShapeValuesNegative(requestInput);
        if (!status.ok())
            return status;

        if (requestInput.dtype() == tensorflow::DataType::DT_STRING) {
            // binary inputs will be validated during conversion to blob
            SPDLOG_DEBUG("Received request containing binary inputs");
            status = validateNumberOfBinaryInputShapeDimensions(*networkInput, requestInput);
            if (!status.ok()) {
                return status;
            }

            if (checkBinaryInputBatchSizeMismatch(*networkInput, requestInput)) {
                if (batchingMode == AUTO) {
                    finalStatus = StatusCode::BATCHSIZE_CHANGE_REQUIRED;
                } else {
                    std::stringstream ss;
                    ss << "Expected: " << getBatchSize() << "; Actual: " << requestInput.string_val_size();
                    const std::string details = ss.str();
                    SPDLOG_DEBUG("[Model: {} version: {}] Invalid batch size - {}", getName(), getVersion(), details);
                    return Status(StatusCode::INVALID_BATCH_SIZE, details);
                }
            }
            continue;
        }

        status = validatePrecision(*networkInput, requestInput);
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
                SPDLOG_DEBUG("[Model: {} version: {}] Invalid batch size - {}", getName(), getVersion(), details);
                return Status(StatusCode::INVALID_BATCH_SIZE, details);
            }
        }

        if (checkShapeMismatch(*networkInput, requestInput, batchingMode)) {
            if (shapeMode == AUTO) {
                finalStatus = StatusCode::RESHAPE_REQUIRED;
            } else {
                std::stringstream ss;
                ss << "Expected: " << TensorInfo::shapeToString(networkInput->getEffectiveShape())
                   << "; Actual: " << TensorInfo::tensorShapeToString(requestInput.tensor_shape());
                const std::string details = ss.str();
                SPDLOG_DEBUG("[Model: {} version: {}] Invalid shape - {}", getName(), getVersion(), details);
                return Status(StatusCode::INVALID_SHAPE, details);
            }
        }

        status = validateTensorContentSize(*networkInput, requestInput);
        if (!status.ok())
            return status;
    }
    return finalStatus;
}

Status ModelInstance::performInference(InferenceEngine::InferRequest& inferRequest) {
    try {
        inferRequest.StartAsync();
        InferenceEngine::StatusCode sts = inferRequest.Wait(InferenceEngine::IInferRequest::RESULT_READY);
        if (sts != InferenceEngine::StatusCode::OK) {
            Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
            SPDLOG_ERROR("Async infer failed {}: {}", status.string(), sts);
            return status;
        }
    } catch (const InferenceEngine::Exception& e) {
        Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
        SPDLOG_ERROR("Async caught an exception {}: {}", status.string(), e.what());
        return status;
    }
    return StatusCode::OK;
}

Status ModelInstance::infer(const tensorflow::serving::PredictRequest* requestProto,
    tensorflow::serving::PredictResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    Timer timer;
    using std::chrono::microseconds;

    auto status = validate(requestProto);
    status = reloadModelIfRequired(status, requestProto, modelUnloadGuardPtr);
    if (!status.ok())
        return status;
    timer.start("get infer request");
    ExecutingStreamIdGuard executingStreamIdGuard(getInferRequestsQueue());
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = executingStreamIdGuard.getInferRequest();
    timer.stop("get infer request");
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("get infer request") / 1000);

    timer.start("deserialize");
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*requestProto, getInputsInfo(), inferRequest);
    timer.stop("deserialize");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("deserialize") / 1000);

    timer.start("prediction");
    status = performInference(inferRequest);
    timer.stop("prediction");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("prediction") / 1000);

    timer.start("serialize");
    status = serializePredictResponse(inferRequest, getOutputsInfo(), responseProto);
    timer.stop("serialize");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), getVersion(), executingInferId, timer.elapsed<microseconds>("serialize") / 1000);

    return StatusCode::OK;
}
}  // namespace ovms

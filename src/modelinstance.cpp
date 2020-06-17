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

const int DEFAULT_OV_STREAMS = std::thread::hardware_concurrency() / 4;
const uint UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS = 10;


void ModelInstance::loadInputTensors(const ModelConfig& config) {
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
        spdlog::info("Input name: {}; mapping_name: {}; shape: {} ; precision: {}", name, mappingName, shape_stream.str(), precision_str);
    }

    // Update OV model shapes
    network->reshape(networkShapes);
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
        spdlog::info("Output name: {} ; mapping name: {}; shape: {} ; precision: {}", name, mappingName, shape_stream.str(), precision_str);
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

std::string getModelFile(const std::string path) {
    struct dirent *entry;
    DIR *dir = opendir(path.c_str());

    while ((entry = readdir(dir)) != nullptr) {
        auto name = std::string(entry->d_name);
        if (endsWith(name, ".xml")) {
            closedir(dir);
            if (endsWith(name, "/")) {
                return path + name;
            } else {
                return path + "/" + name;
            }
        }
    }
    closedir(dir);

    return path;
}

uint getOVCPUThroughputStreams() {
    const char* environmentVariableBuffer = std::getenv("CPU_THROUGHPUT_STREAMS");
    if (environmentVariableBuffer) {
        auto result = stou32(environmentVariableBuffer);
        if (result && result.value() > 0) {
            return result.value();
        }
    }

    return std::max(std::thread::hardware_concurrency() / 8, 1u);
}

uint getNumberOfParallelInferRequests() {
    const char* environmentVariableBuffer = std::getenv("NIREQ");
    if (environmentVariableBuffer) {
        auto result = stou32(environmentVariableBuffer);
        if (result && result.value() > 0) {
            return result.value();
        }
    }
    auto& config = ovms::Config::instance();
    return std::max(config.nireq(), 1u);
}

Status ModelInstance::loadModel(const ModelConfig& config) {
    this->name = config.getName();
    this->path = config.getPath();
    this->version = config.getVersion();
    this->backend = config.getBackend();
    this->status = ModelVersionStatus(this->name, this->version);
    spdlog::info("Loading model:{}, version:{}, from path:{}, with backend:{} ...",
        config.getName(), config.getVersion(), config.getPath(), config.getBackend());
    // load network
    try {
        this->status.setLoading();
        if (!dirExists(path)) {
            spdlog::error("Missing model directory {}", path);
            this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
            return StatusCode::PATH_INVALID;
        }
        engine = std::make_unique<InferenceEngine::Core>();
        network = std::make_unique<InferenceEngine::CNNNetwork>(engine->ReadNetwork(getModelFile(path)));
        this->batchSize = config.getBatchSize() > 0 ? config.getBatchSize() : network->getBatchSize();

        network->setBatchSize(this->batchSize);
        loadInputTensors(config);
        loadOutputTensors(config);
        auto pluginConfig = config.getPluginConfig();
        if (pluginConfig.count("CPU_THROUGHPUT_STREAMS") == 0) {
            uint ovBackendStreamsCount = getOVCPUThroughputStreams();
            pluginConfig["CPU_THROUGHPUT_STREAMS"] = std::to_string(ovBackendStreamsCount);
        }
        execNetwork = std::make_shared<InferenceEngine::ExecutableNetwork>(engine->LoadNetwork(*network, backend, pluginConfig));
        spdlog::info("Plugin config for device {}:", backend);
        for (const auto pair : pluginConfig) {
            const auto key = pair.first;
            const auto value = pair.second;
            spdlog::info("{}: {}", key, value);
        }
        uint numberOfParallelInferRequests = getNumberOfParallelInferRequests();
        inferRequestsQueue = std::make_shared<OVInferRequestsQueue>(*execNetwork, numberOfParallelInferRequests);
        spdlog::info("Loaded model {}; version: {}; No of InferRequests: {}",
            config.getName(),
            config.getVersion(),
            numberOfParallelInferRequests);

        this->status.setAvailable();
    }
    catch (const InferenceEngine::details::InferenceEngineException& e) {
        spdlog::error("exception occurred while loading network: {}", e.what());
        this->status.setLoading(ModelVersionStatusErrorCode::UNKNOWN);
        return StatusCode::NETWORK_NOT_LOADED;
    }

    return StatusCode::OK;
}

void ModelInstance::unloadModel() {
    this->status.setUnloading();
    while (!canUnloadInferRequests()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(UNLOAD_AVAILABILITY_CHECKING_INTERVAL_MILLISECONDS));
    }
    inferRequestsQueue.reset();
    execNetwork.reset();
    network.reset();
    engine.reset();
    this->outputsInfo.clear();
    this->inputsInfo.clear();
    this->status.setEnd();
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

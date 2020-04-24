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
#include <iostream>
#include <string>
#include <sys/types.h>

#include <cstdlib>

#include "config.h"
#include "modelinstance.h"

using namespace InferenceEngine;

namespace ovms {

const int DEFAULT_OV_STREAMS = std::thread::hardware_concurrency() / 4;
const int DEFAULT_OV_BACKEND_STREAMS = DEFAULT_OV_STREAMS;

void ModelInstance::loadInputTensors(const ModelConfig& config) {
    auto networkShapes = network.getInputShapes();

    for (const auto& pair : network.getInputsInfo()) {
        const auto& name = pair.first;
        auto input = pair.second;

        // Data from network
        auto precision = input->getPrecision();
        auto layout = input->getLayout();
        auto shape = input->getTensorDesc().getDims();
        auto desc = input->getTensorDesc();

        // Data from config
        if (config.getLayout().size()) {
            // Single layout for all inputs
            layout = TensorInfo::getLayoutFromString(config.getLayout());
        } 
        else if (config.getLayouts().count(name)) {
            // Layout defined for specific input
            layout = TensorInfo::getLayoutFromString(config.getLayouts().at(name));
        }
        input->setLayout(layout);

        // One shape for all inputs
        if (config.getShape().size()) {
            shape = config.getShape();
        }
        // Shape for specific input
        else if (config.getShapes().count(name)) {
            shape = config.getShapes().at(name);
        }

        if (config.getBatchSize() > 0) {
            shape[0] = config.getBatchSize();
        }

        networkShapes[name] = shape;
        this->inputsInfo[name] = std::make_shared<TensorInfo>(
            name, precision, shape, layout, desc);
    }

    // Update OV model shapes
    network.reshape(networkShapes);
}

void ModelInstance::loadOutputTensors(const ModelConfig& config) {
    for (const auto& pair : network.getOutputsInfo()) {
        const auto& name = pair.first;
        auto output = pair.second;

        // Data from network
        auto precision = output->getPrecision();
        auto layout = output->getLayout();
        auto shape = output->getDims();
        auto desc = output->getTensorDesc();

        this->outputsInfo[name] = std::make_shared<TensorInfo>(
            name, precision, shape, layout, desc);
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
        return std::atoi(environmentVariableBuffer);
    }

    auto& config = ovms::Config::instance();
    uint configGRPCServersCount = config.cpuThroughputStreams();
    return configGRPCServersCount;
}

uint getNumberOfParallelInferRequests() {
    const char* environmentVariableBuffer = std::getenv("NIREQ");
    if (environmentVariableBuffer) {
        return std::atoi(environmentVariableBuffer);
    }

    auto& config = ovms::Config::instance();
    uint configGRPCServersCount = config.nireq();
    return configGRPCServersCount;
}

Status ModelInstance::loadModel(const ModelConfig& config) {
    this->path = config.getBasePath();
    this->version = config.getVersion();
    this->backend = config.getBackend();

    // load network
    try {
        if (!dirExists(path)) {
            return Status::PATH_INVALID;
        }
        network = engine.ReadNetwork(getModelFile(path));
        this->batchSize = config.getBatchSize() > 0 ? config.getBatchSize() : network.getBatchSize();

        network.setBatchSize(this->batchSize);

        loadInputTensors(config);
        loadOutputTensors(config);

        int ovBackendStreamsCount = getOVCPUThroughputStreams();
        execNetwork = engine.LoadNetwork(network, backend, {{ "CPU_THROUGHPUT_STREAMS", std::to_string(ovBackendStreamsCount)}});
        std::cout << "Starting OpenVINO CPU streams:" << ovBackendStreamsCount << std::endl;

        int numberOfParallelInferRequests = getNumberOfParallelInferRequests();
        std::cout << "Starting OpenVINO InferRequestsQueue:" << numberOfParallelInferRequests << std::endl;
        inferRequestsQueue = std::make_unique<OVInferRequestsQueue>(execNetwork, numberOfParallelInferRequests);
    }
    catch (const InferenceEngine::details::InferenceEngineException& e) {
        // Logger(Log::Error, e.what());
        std::cout << e.what() << std::endl;
        return Status::NETWORK_NOT_LOADED;
    }

    return Status::OK;
}

const ValidationStatusCode ModelInstance::validate(const tensorflow::serving::PredictRequest* request) {

    // Network and request must have the same amount of inputs
    if (request->inputs_size() >= 0 && getInputsInfo().size() != (size_t) request->inputs_size()) {
        return ValidationStatusCode::INVALID_INPUT_ALIAS;
    }

    for (const auto& pair : getInputsInfo()) {
        const auto& name = pair.first;
        auto networkInput = pair.second;
        auto it = request->inputs().find(name);

        // Network and request must have the same names of inputs
        if (it == request->inputs().end()) {
            return ValidationStatusCode::INVALID_INPUT_ALIAS;
        }

        auto& requestInput = it->second;
        auto& shape = networkInput->getShape();

        // Network and request must have the same number of shape dimensions
        if (requestInput.tensor_shape().dim_size() >= 0 &&
            shape.size() != (size_t) requestInput.tensor_shape().dim_size()) {
            return ValidationStatusCode::INVALID_SHAPE;
        }

        // First shape must be equal to batch size
        if (requestInput.tensor_shape().dim_size() > 0 && requestInput.tensor_shape().dim(0).size() != getBatchSize()) {
            return ValidationStatusCode::INCORRECT_BATCH_SIZE;
        }

        // Network and request must have the same shape
        for (int i = 1; i < requestInput.tensor_shape().dim_size(); i++) {
            if (requestInput.tensor_shape().dim(i).size() >= 0 && shape[i] != (size_t) requestInput.tensor_shape().dim(i).size()) {
                return ValidationStatusCode::INVALID_SHAPE;
            }
        }

        // Network expects tensor content size
        size_t expectedContentSize = std::accumulate(
            networkInput->getShape().begin(),
            networkInput->getShape().end(),
            1,
            std::multiplies<size_t>());

        expectedContentSize *= networkInput->getPrecision().size();

        if (expectedContentSize != requestInput.tensor_content().size()) {
            return ValidationStatusCode::INVALID_CONTENT_SIZE;
        }

        // Network and request must have the same precision
        if (requestInput.dtype() != networkInput->getPrecisionAsDataType()) {
            return ValidationStatusCode::INVALID_PRECISION;
        }
    }

    return ValidationStatusCode::OK;
}

} // namespace ovms

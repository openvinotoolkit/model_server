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

#include "modelinstance.h"

using namespace InferenceEngine;

namespace ovms {

template<typename T>
void ModelInstance::loadTensors(tensorMap& map,
                                const T& tensors,
                                const shapesMap& shapes,
                                const layoutsMap& layouts) {
    for (const auto& t : tensors) {
        auto precision = t.second->getPrecision();
        auto ieLayout = layouts.count(t.first) ? TensorInfo::getLayoutFromString(layouts.at(t.first)) :
                                                 t.second->getLayout();
        InferenceEngine::TensorDesc tensorDesc = t.second->getTensorDesc();
        std::vector<size_t> shape;
        if (shapes.count(t.first)) {
            shape.assign(shapes.at(t.first).begin(), shapes.at(t.first).end());
        } else {
            shape.assign(tensorDesc.getDims().begin(), tensorDesc.getDims().end());
        }

        // If shape or layout provided, create custom TensorDesc
        if (shapes.count(t.first) || layouts.count(t.first)) {
            tensorDesc = InferenceEngine::TensorDesc(precision, shape, ieLayout);
        }
        map[t.first] = std::make_shared<TensorInfo>(
            t.first,
            precision,
            shape,
            ieLayout,
            tensorDesc);
    }
}

// Temporary methods. To be replaces with proper storage class.
bool endsWith(std::string token, std::string match)
{
	auto it = match.begin();
	return token.size() >= match.size() &&
		std::all_of(std::next(token.begin(),token.size() - match.size()), token.end(), [&it](const char & c){
			return ::tolower(c) == ::tolower(*(it++))  ;
	    });
}

std::string getModelFile(const std::string path) {
    struct dirent *entry;
    DIR *dir = opendir(path.c_str());

    while ((entry = readdir(dir)) != NULL) {
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

Status ModelInstance::loadModel(const ModelConfig& config) {
    this->path = config.basePath;
    this->version = config.version;
    this->backend = config.backend;

    // load network
    try {
        network = engine.ReadNetwork(getModelFile(path));
        this->batchSize = config.batchSize > 0 ? config.batchSize : network.getBatchSize();

        loadTensors(inputsInfo,  network.getInputsInfo(), config.shapes, config.layouts);
        loadTensors(outputsInfo, network.getOutputsInfo(), config.shapes, config.layouts);

        execNetwork = engine.LoadNetwork(network, backend, {{ "CPU_THROUGHPUT_STREAMS", std::to_string(OV_STREAMS_COUNT)}});
        request = execNetwork.CreateInferRequest();
        ovstreams = std::make_unique<OVStreamsQueue>(execNetwork, OV_STREAMS_COUNT);
    }
    catch (const InferenceEngine::details::InferenceEngineException& e) {
        // Logger(Log::Error, e.what());
        return Status::NETWORK_NOT_LOADED;
    }

    return Status::OK;
}

InferenceEngine::InferRequest& ModelInstance::infer(const std::string& inputName, const InferenceEngine::Blob::Ptr data) {
    request.SetBlob(inputName, data);
    request.Infer();

    return request;
}

InferenceEngine::InferRequest& ModelInstance::inferAsync(const std::string& inputName,
                                                         const InferenceEngine::Blob::Ptr data,
                                                         const std::function<void()>& callback) {
    request.SetBlob(inputName, data);
    request.SetCompletionCallback(callback);
    request.StartAsync();

    return request;
}

const ValidationStatusCode ModelInstance::validate(const tensorflow::serving::PredictRequest* request) {

    // Network and request must have the same amount of inputs
    if (request->inputs_size() >= 0 && inputsInfo.size() != (size_t) request->inputs_size()) {
        return ValidationStatusCode::INVALID_INPUT_ALIAS;
    }

    for (const auto& pair : inputsInfo) {
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
        if (requestInput.tensor_shape().dim_size() >= 0 && shape.size() != (size_t) requestInput.tensor_shape().dim_size()) {
            return ValidationStatusCode::INVALID_SHAPE;
        }

        // Network and request must have the same shape
        for (int i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
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
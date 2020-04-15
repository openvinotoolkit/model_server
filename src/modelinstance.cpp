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
#include <string>

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

Status ModelInstance::loadModel( const std::string& path,
                                 const std::string& backend,
                                 const model_version_t& version,
                                 const size_t batchSize,
                                 const shapesMap& shapes,
                                 const layoutsMap& layouts) {
    this->path = path;
    this->version = version;
    this->backend = backend;

    // load network
    try {
        network = engine.ReadNetwork(path);
        this->batchSize = batchSize > 0 ? batchSize : network.getBatchSize();

        loadTensors(inputsInfo,  network.getInputsInfo(),  shapes, layouts);
        loadTensors(outputsInfo, network.getOutputsInfo(), shapes, layouts);

        execNetwork = engine.LoadNetwork(network, backend);
        request = execNetwork.CreateInferRequest();
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
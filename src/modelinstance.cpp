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
        execNetwork = engine.LoadNetwork(network, backend);
        this->batchSize = batchSize > 0 ? batchSize : network.getBatchSize();

        loadTensors(inputsInfo, network.getInputsInfo(), shapes, layouts);
        loadTensors(outputsInfo, network.getOutputsInfo(), shapes, layouts);
        
        request = execNetwork.CreateInferRequest();
    }
    catch (const InferenceEngine::details::InferenceEngineException e) {
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

} // namespace ovms
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
#include "modelversion.h"

using namespace InferenceEngine;

namespace ovms {

Status ModelVersion::loadModel( const std::string& path,
                                const std::string& backend,
                                const int64_t version,
                                const size_t batchSize,
                                const std::vector<size_t>& shape) {
    this->path = path;
    this->version = version;
    this->backend = backend;
    this->batchSize = batchSize;
    std::copy(shape.begin(), shape.end(), std::back_inserter(this->shape)); 

    // load network
    network = engine.ReadNetwork(path);
    execNetwork = engine.LoadNetwork(network, backend);

    // setup input
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    this->inputName = network.getInputsInfo().begin()->first;
    // TODO read layout and precision from configuration
    input_info->setPrecision(Precision::FP32);

    // setup output
    DataPtr output_info = network.getOutputsInfo().begin()->second;
    this->outputName = network.getOutputsInfo().begin()->first;
    output_info->setPrecision(Precision::FP32);
    
    request = execNetwork.CreateInferRequest();
    
    return Status::OK;
}

const InferenceEngine::Blob::Ptr ModelVersion::infer(const InferenceEngine::Blob::Ptr input) {
    request.SetBlob(inputName, input);
    request.Infer();

    return request.GetBlob(outputName);
}

const InferenceEngine::InferRequest& ModelVersion::inferAsync(const InferenceEngine::Blob::Ptr input, std::function<void()> callback) {
    request.SetBlob(inputName, input);
    request.SetCompletionCallback(callback);
    request.StartAsync();

    return request;
}

} // namespace ovms
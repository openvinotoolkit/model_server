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
#include <condition_variable>
#include <inference_engine.hpp>

#include "tensorflow/core/framework/tensor.h"

#include "modelmanager.h"
#include "prediction_service.hpp"

#define DEBUG

#include "timer.h"


using grpc::ServerContext;

using namespace InferenceEngine;

using tensorflow::TensorProto;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

using ovms::ModelManager;
using ovms::ValidationStatus;
using ovms::ValidationStatusCode;

namespace ovms {

template<typename T>
Blob::Ptr makeBlob(const TensorProto& requestInput, const std::shared_ptr<TensorInfo>& networkInput) {
    return make_shared_blob<T>(
        networkInput->getTensorDesc(),
        const_cast<T*>(reinterpret_cast<const T*>(requestInput.tensor_content().data())));
}

Blob::Ptr deserialize(const TensorProto& requestInput, const std::shared_ptr<TensorInfo>& networkInput) {
    switch (networkInput->getPrecision()) {
        case Precision::FP32:   return makeBlob<float>  (requestInput, networkInput);
        case Precision::I32:    return makeBlob<int32_t>(requestInput, networkInput);
        case Precision::U8:     return makeBlob<uint8_t>(requestInput, networkInput);
        default:                return nullptr;
    }
}

void serialize(TensorProto& responseOutput, const std::shared_ptr<TensorInfo>& networkOutput, Blob::Ptr blob) {
    responseOutput.Clear();

    switch (networkOutput->getPrecision()) {
        case Precision::FP32: responseOutput.set_dtype(tensorflow::DataTypeToEnum<float>::value); break;
        //case Precision::DOUBLE?: responseOutput.set_dtype(tensorflow::DataTypeToEnum<float>::value); // unsupported by OV?
        case Precision::I32:  responseOutput.set_dtype(tensorflow::DataTypeToEnum<int>::value); break;
    }

    responseOutput.mutable_tensor_shape()->Clear();
    for (auto dim : networkOutput->getShape()) {
        responseOutput.mutable_tensor_shape()->add_dim()->set_size(dim);
    }

    responseOutput.mutable_tensor_content()->assign((char*) blob->buffer(), blob->byteSize());

}

void infer(InferRequest& inferRequest) {
    std::condition_variable cv;
    std::mutex mx;
    std::unique_lock<std::mutex> lock(mx);

    inferRequest.SetCompletionCallback([&]{
        cv.notify_one();
    });

    inferRequest.StartAsync();
    cv.wait(lock);
}

grpc::Status ovms::PredictionServiceImpl::Predict(
            ServerContext*      context,
    const   PredictRequest*     request,
            PredictResponse*    response) {

    ModelManager& manager = ModelManager::getInstance();

    auto& modelName = request->model_spec().name();
    auto modelVersionId = request->model_spec().version().value();

    auto model = manager.findModelByName(modelName);
    if (model == nullptr) {
        return grpc::Status(
            grpc::StatusCode::NOT_FOUND,
            ValidationStatus::getError(
                ValidationStatusCode::MODEL_NAME_MISSING));
    }

    auto modelVersion = model->getModelInstanceByVersion(modelVersionId);
    if (modelVersion == nullptr) {
        return grpc::Status(
            grpc::StatusCode::NOT_FOUND,
            ValidationStatus::getError(
                ValidationStatusCode::MODEL_VERSION_MISSING));
    }

    auto& inferRequest  = modelVersion->getInferRequest();

    auto result = modelVersion->validate(request);
    if (result != ValidationStatusCode::OK) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            ValidationStatus::getError(result));
    }

    // Deserialization
    try {
        for (const auto& pair : modelVersion->getInputsInfo()) {
            const auto& name = pair.first;
            auto networkInput = pair.second;
            auto& requestInput = request->inputs().find(name)->second;

            auto blob = deserialize(requestInput, networkInput);
            if (blob == nullptr) {
                throw;
            }
            inferRequest.SetBlob(name, blob);
        }
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        std::cout << e.what() << std::endl;
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            ValidationStatus::getError(
                ValidationStatusCode::DESERIALIZATION_ERROR));
    }

    // Infer
    try {
        infer(inferRequest);
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        std::cout << e.what() << std::endl;
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            ValidationStatus::getError(
                ValidationStatusCode::INFERENCE_ERROR));
    }

    // Serialization
    for (const auto& pair : modelVersion->getOutputsInfo()) {
        const auto& name = pair.first;
        auto networkOutput = pair.second;
        auto blob = inferRequest.GetBlob(name);

        auto& tensorProto = (*response->mutable_outputs())[name];
        serialize(tensorProto, networkOutput, blob);
    }

    return grpc::Status::OK;
}

} // namespace ovms
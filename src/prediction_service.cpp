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
#include <spdlog/spdlog.h>

#include "ovinferrequestsqueue.hpp"
#include "modelmanager.hpp"
#include "prediction_service.hpp"

#include "get_model_metadata_impl.hpp"

#define DEBUG
#include "timer.hpp"

using grpc::ServerContext;

using namespace InferenceEngine;

using tensorflow::TensorProto;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

namespace ovms {

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

grpc::Status getModelInstance(const PredictRequest* request, std::shared_ptr<ovms::ModelInstance>& modelInstance) {
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


    if (modelVersionId != 0) {
        modelInstance = model->getModelInstanceByVersion(modelVersionId);
        if (modelInstance == nullptr) {
            return grpc::Status(
                grpc::StatusCode::NOT_FOUND,
                ValidationStatus::getError(
                    ValidationStatusCode::MODEL_VERSION_MISSING));
        }
    } else {
        modelInstance = model->getDefaultModelInstance();
    }
    
    return grpc::Status::OK;
}

grpc::Status validateRequest(const PredictRequest* request, ovms::ModelInstance& modelInstance) {
    auto result = modelInstance.validate(request);
    if (result != ValidationStatusCode::OK) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            ValidationStatus::getError(result));
    }
    return grpc::Status::OK;
}

template<typename T>
Blob::Ptr makeBlob(const TensorProto& requestInput, const std::shared_ptr<TensorInfo>& tensorInfo) {
    return make_shared_blob<T>(
        tensorInfo->getTensorDesc(),
        const_cast<T*>(reinterpret_cast<const T*>(requestInput.tensor_content().data())));
}

Blob::Ptr deserialize(const TensorProto& requestInput, const std::shared_ptr<TensorInfo>& tensorInfo) {
    switch (tensorInfo->getPrecision()) {
        case Precision::FP32:   return makeBlob<float>  (requestInput, tensorInfo);
        case Precision::I32:    return makeBlob<int32_t>(requestInput, tensorInfo);
        case Precision::U8:     return makeBlob<uint8_t>(requestInput, tensorInfo);
        default:                return nullptr;
    }
}

grpc::Status deserialize(const PredictRequest* request, const tensor_map_t& inputMap, InferenceEngine::InferRequest& inferRequest) {
    try {
        for (const auto& pair : inputMap) {
            const auto& name = pair.first;
            auto tensorInfo = pair.second;
            auto& requestInput = request->inputs().find(name)->second;

            auto blob = deserialize(requestInput, tensorInfo);
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
    return grpc::Status::OK;
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

void serialize(InferenceEngine::InferRequest& inferRequest, const tensor_map_t& outputMap, PredictResponse* response)
{
    for (const auto& pair : outputMap) {
        const auto& name = pair.first;
        auto networkOutput = pair.second;
        auto blob = inferRequest.GetBlob(name);

        auto& tensorProto = (*response->mutable_outputs())[name];
        serialize(tensorProto, networkOutput, blob);
    }
}

struct ExecutingStreamIdGuard {
    ExecutingStreamIdGuard(ovms::OVInferRequestsQueue& inferRequestsQueue) :
        inferRequestsQueue_(inferRequestsQueue),
        id_(inferRequestsQueue_.getIdleStream()) {}
    ~ExecutingStreamIdGuard(){
        inferRequestsQueue_.returnStream(id_);
    }
    int getId() { return id_; }
private:
    ovms::OVInferRequestsQueue& inferRequestsQueue_;
    const int id_;
};

grpc::Status performInference(ovms::OVInferRequestsQueue& inferRequestsQueue, const int executingInferId, InferenceEngine::InferRequest& inferRequest) {
    try {
        inferRequest.SetCompletionCallback([&inferRequestsQueue, executingInferId]() {
            inferRequestsQueue.signalCompletedInference(executingInferId);
        });
        inferRequest.StartAsync();
        inferRequestsQueue.waitForAsync(executingInferId);
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        std::cout << e.what() << std::endl;
        return grpc::Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        ValidationStatus::getError(
            ValidationStatusCode::INFERENCE_ERROR));
    }
    return grpc::Status::OK;
}

grpc::Status ovms::PredictionServiceImpl::Predict(
            ServerContext*      context,
    const   PredictRequest*     request,
            PredictResponse*    response) {

    Timer timer;
    std::shared_ptr<ovms::ModelInstance> modelVersion;
    grpc::Status status = getModelInstance(request, modelVersion);
    if(!status.ok())
        return status;

    status = validateRequest(request, *modelVersion);
    if(!status.ok())
        return status;

    timer.start("get infer request");
    ovms::OVInferRequestsQueue& inferRequestsQueue = modelVersion->getInferRequestsQueue();
    ExecutingStreamIdGuard executingStreamIdGuard(inferRequestsQueue);
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);
    timer.stop("get infer request");
    spdlog::debug("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
            request->model_spec().name(),modelVersion->getVersion(),executingInferId,timer.elapsed_microseconds("get infer request")/1000);

    timer.start("deserialize");
    status = deserialize(request, modelVersion->getInputsInfo(), inferRequest);
    timer.stop("deserialize");
    spdlog::debug("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
            request->model_spec().name(),modelVersion->getVersion(),executingInferId,timer.elapsed_microseconds("deserialize")/1000);
    if(!status.ok())
        return status;

    timer.start("prediction");
    status = performInference(inferRequestsQueue, executingInferId, inferRequest);
    timer.stop("prediction");
    if(!status.ok())
        return status;
    spdlog::debug("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
            request->model_spec().name(),modelVersion->getVersion(),executingInferId,timer.elapsed_microseconds("prediction")/1000);

    timer.start("serialize");
    serialize(inferRequest, modelVersion->getOutputsInfo(), response);
    timer.stop("serialize");
    spdlog::debug("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
            request->model_spec().name(),modelVersion->getVersion(),executingInferId, timer.elapsed_microseconds("serialize")/1000);
    return grpc::Status::OK;
}

grpc::Status PredictionServiceImpl::GetModelMetadata(
            grpc::ServerContext*                            context,
    const   tensorflow::serving::GetModelMetadataRequest*   request,
            tensorflow::serving::GetModelMetadataResponse*  response) {

    auto status = GetModelMetadataImpl::getModelStatus(request, response);
    if (status == GetModelMetadataStatusCode::OK) {
        return grpc::Status::OK;
    }

    return grpc::Status(
        status == GetModelMetadataStatusCode::MODEL_MISSING ? grpc::StatusCode::NOT_FOUND 
                                                            : grpc::StatusCode::INVALID_ARGUMENT,
        GetModelMetadataStatus::getError(status));
}

} // namespace ovms

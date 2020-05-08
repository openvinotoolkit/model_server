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


#include "deserialization.hpp"
#include "get_model_metadata_impl.hpp"
#include "modelmanager.hpp"
#include "ovinferrequestsqueue.hpp"
#include "prediction_service.hpp"
#include "serialization.hpp"

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

ValidationStatusCode getModelInstance(const PredictRequest* request, std::shared_ptr<ovms::ModelInstance>& modelInstance) {
    ModelManager& manager = ModelManager::getInstance();

    auto& modelName = request->model_spec().name();
    auto modelVersionId = request->model_spec().version().value();

    auto model = manager.findModelByName(modelName);
    if (model == nullptr) {
        return ValidationStatusCode::MODEL_NAME_MISSING;
    }

    if (modelVersionId != 0) {
        modelInstance = model->getModelInstanceByVersion(modelVersionId);
        if (modelInstance == nullptr) {
            return ValidationStatusCode::MODEL_VERSION_MISSING;
        }
    } else {
        modelInstance = model->getDefaultModelInstance();
    }
    return ValidationStatusCode::OK;
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

ValidationStatusCode performInference(ovms::OVInferRequestsQueue& inferRequestsQueue, const int executingInferId, InferenceEngine::InferRequest& inferRequest) {
    try {
        inferRequest.SetCompletionCallback([&inferRequestsQueue, executingInferId]() {
            inferRequestsQueue.signalCompletedInference(executingInferId);
        });
        inferRequest.StartAsync();
        inferRequestsQueue.waitForAsync(executingInferId);
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        std::cout << e.what() << std::endl;
        return ValidationStatusCode::INFERENCE_ERROR;
    }
    return ValidationStatusCode::OK;
}

inline grpc::Status convertStatus(ValidationStatusCode& validationStatusCode) {
    grpc::StatusCode grpcStatusCode;
    switch (validationStatusCode) {
    case ValidationStatusCode::DESERIALIZATION_ERROR:
    case ValidationStatusCode::INCORRECT_BATCH_SIZE:
    case ValidationStatusCode::INFERENCE_ERROR:
    case ValidationStatusCode::INVALID_CONTENT_SIZE:
    case ValidationStatusCode::INVALID_INPUT_ALIAS:
    case ValidationStatusCode::INVALID_PRECISION:
    case ValidationStatusCode::INVALID_SHAPE:
        grpcStatusCode = grpc::StatusCode::INVALID_ARGUMENT;
        break;
    case ValidationStatusCode::MODEL_NAME_MISSING:
    case ValidationStatusCode::MODEL_VERSION_MISSING:
        grpcStatusCode = grpc::StatusCode::NOT_FOUND;
        break;
    case ValidationStatusCode::OK:
        return grpc::Status::OK;
    default:
        grpcStatusCode = grpc::StatusCode::UNKNOWN;
    }
    return grpc::Status(grpcStatusCode, ValidationStatus::getError(validationStatusCode));
}

grpc::Status ovms::PredictionServiceImpl::Predict(
            ServerContext*      context,
    const   PredictRequest*     request,
            PredictResponse*    response) {
    Timer timer;

    std::shared_ptr<ovms::ModelInstance> modelVersion;

    ValidationStatusCode status = getModelInstance(request, modelVersion);
    if(ValidationStatusCode::OK != status)
        return convertStatus(status);

    status = modelVersion->validate(request);
    if(ValidationStatusCode::OK != status)
        return convertStatus(status);

    timer.start("get infer request");
    ovms::OVInferRequestsQueue& inferRequestsQueue = modelVersion->getInferRequestsQueue();
    ExecutingStreamIdGuard executingStreamIdGuard(inferRequestsQueue);
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);
    timer.stop("get infer request");
    spdlog::debug("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
            request->model_spec().name(),modelVersion->getVersion(),executingInferId,timer.elapsed_microseconds("get infer request") / 1000);

    timer.start("deserialize");
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*request, modelVersion->getInputsInfo(), inferRequest);
    timer.stop("deserialize");
    if(ValidationStatusCode::OK != status)
        return convertStatus(status);
    spdlog::debug("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        request->model_spec().name(), modelVersion->getVersion(), executingInferId, timer.elapsed_microseconds("deserialize") / 1000);

    timer.start("prediction");
    status = performInference(inferRequestsQueue, executingInferId, inferRequest);
    timer.stop("prediction");
    // TODO - current return code below is the same as in Python, but INVALID_ARGUMENT does not neccesarily mean
    // that the problem may be input
    if (ValidationStatusCode::OK != status)
        return convertStatus(status);
    spdlog::debug("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
            request->model_spec().name(),modelVersion->getVersion(),executingInferId,timer.elapsed_microseconds("prediction") / 1000);

    timer.start("serialize");
    status = serializePredictResponse(inferRequest, modelVersion->getOutputsInfo(), response);
    timer.stop("serialize");
    if (ValidationStatusCode::OK != status)
        return convertStatus(status);
    spdlog::debug("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
            request->model_spec().name(), modelVersion->getVersion(), executingInferId, timer.elapsed_microseconds("serialize") / 1000);

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

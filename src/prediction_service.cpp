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
#include <memory>
#include <string>
#include <utility>

#include <inference_engine.hpp>
#include "tensorflow/core/framework/tensor.h"
#include <spdlog/spdlog.h>

#include "deserialization.hpp"
#include "get_model_metadata_impl.hpp"
#include "modelmanager.hpp"
#include "ovinferrequestsqueue.hpp"
#include "prediction_service.hpp"
#include "prediction_service_utils.hpp"
#include "serialization.hpp"
#include "status.hpp"

#define DEBUG
#include "timer.hpp"

using grpc::ServerContext;

using namespace InferenceEngine;

using tensorflow::TensorProto;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

namespace ovms {

Status getModelInstance(const PredictRequest* request,
                        std::shared_ptr<ovms::ModelInstance>& modelInstance,
                        std::unique_ptr<ModelInstancePredictRequestsHandlesCountGuard>& modelInstancePredictRequestsHandlesCountGuardPtr) {
    ModelManager& manager = ModelManager::getInstance();
    return getModelInstance(manager, request->model_spec().name(), request->model_spec().version().value(), modelInstance, modelInstancePredictRequestsHandlesCountGuardPtr);
}

grpc::Status validateRequest(const PredictRequest* request, ovms::ModelInstance& modelInstance) {
    auto status = modelInstance.validate(request);
    if (!status.ok()) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            status.string());
    }
    return status;
}

struct ExecutingStreamIdGuard {
    ExecutingStreamIdGuard(ovms::OVInferRequestsQueue& inferRequestsQueue) :
        inferRequestsQueue_(inferRequestsQueue),
        id_(inferRequestsQueue_.getIdleStream()) {}
    ~ExecutingStreamIdGuard() {
        inferRequestsQueue_.returnStream(id_);
    }
    int getId() { return id_; }
private:
    ovms::OVInferRequestsQueue& inferRequestsQueue_;
    const int id_;
};

Status performInference(ovms::OVInferRequestsQueue& inferRequestsQueue, const int executingInferId, InferenceEngine::InferRequest& inferRequest) {
    try {
        inferRequest.SetCompletionCallback([&inferRequestsQueue, executingInferId]() {
            inferRequestsQueue.signalCompletedInference(executingInferId);
        });
        inferRequest.StartAsync();
        inferRequestsQueue.waitForAsync(executingInferId);
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
        SPDLOG_ERROR("{}: {}", status.string(), e.what());
        return status;
    }

    return StatusCode::OK;
}

grpc::Status ovms::PredictionServiceImpl::Predict(
            ServerContext*      context,
    const   PredictRequest*     request,
            PredictResponse*    response) {
    Timer timer;
    spdlog::debug("Got new PredictRequest for model:{}; version:{}",
                  request->model_spec().name(),
                  request->model_spec().version().value());

    std::shared_ptr<ovms::ModelInstance> modelVersion;

    std::unique_ptr<ModelInstancePredictRequestsHandlesCountGuard> modelInstancePredictRequestsHandlesCountGuard;
    auto status = getModelInstance(request, modelVersion, modelInstancePredictRequestsHandlesCountGuard);
    if (!status.ok()) {
        SPDLOG_INFO("Getting modelInstance failed. {}", status.string());
        return status.grpc();
    }

    status = modelVersion->validate(request);
    if (!status.ok()) {
        SPDLOG_INFO("Validation of inferRequest failed. {}", status.string());
        return status.grpc();
    }

    timer.start("get infer request");
    ovms::OVInferRequestsQueue& inferRequestsQueue = modelVersion->getInferRequestsQueue();
    ExecutingStreamIdGuard executingStreamIdGuard(inferRequestsQueue);
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);
    timer.stop("get infer request");
    spdlog::debug("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
            request->model_spec().name(), modelVersion->getVersion(), executingInferId, timer.elapsed_microseconds("get infer request") / 1000);

    timer.start("deserialize");
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*request, modelVersion->getInputsInfo(), inferRequest);
    timer.stop("deserialize");
    if (!status.ok())
        return status.grpc();
    spdlog::debug("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        request->model_spec().name(), modelVersion->getVersion(), executingInferId, timer.elapsed_microseconds("deserialize") / 1000);

    timer.start("prediction");
    status = performInference(inferRequestsQueue, executingInferId, inferRequest);
    timer.stop("prediction");
    // TODO - current return code below is the same as in Python, but INVALID_ARGUMENT does not neccesarily mean
    // that the problem may be input
    if (!status.ok())
        return status.grpc();
    spdlog::debug("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
            request->model_spec().name(), modelVersion->getVersion(), executingInferId, timer.elapsed_microseconds("prediction") / 1000);

    timer.start("serialize");
    status = serializePredictResponse(inferRequest, modelVersion->getOutputsInfo(), response);
    timer.stop("serialize");
    if (!status.ok())
        return status.grpc();
    spdlog::debug("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
            request->model_spec().name(), modelVersion->getVersion(), executingInferId, timer.elapsed_microseconds("serialize") / 1000);

    return grpc::Status::OK;
}

grpc::Status PredictionServiceImpl::GetModelMetadata(
            grpc::ServerContext*                            context,
    const   tensorflow::serving::GetModelMetadataRequest*   request,
            tensorflow::serving::GetModelMetadataResponse*  response) {
    return GetModelMetadataImpl::getModelStatus(request, response).grpc();
}

}  // namespace ovms

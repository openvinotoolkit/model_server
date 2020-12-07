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
#include "prediction_service_utils.hpp"

#include <map>

#include "deserialization.hpp"
#include "executinstreamidguard.hpp"
#include "modelinstance.hpp"
#include "modelinstanceunloadguard.hpp"
#include "modelmanager.hpp"
#include "serialization.hpp"

#define DEBUG
#include "timer.hpp"

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

namespace ovms {

size_t getRequestBatchSize(const tensorflow::serving::PredictRequest* request) {
    auto requestInputItr = request->inputs().begin();
    if (requestInputItr == request->inputs().end()) {
        SPDLOG_WARN("Failed to get batch size of a request. Validation of request failed");
        return 0;
    }
    auto& requestInput = requestInputItr->second;  // assuming same batch size for all inputs
    return static_cast<size_t>(requestInput.tensor_shape().dim(0).size());
}

std::map<std::string, shape_t> getRequestShapes(const tensorflow::serving::PredictRequest* request) {
    std::map<std::string, shape_t> requestShapes;
    for (auto& it : request->inputs()) {
        shape_t requestShape;
        std::string name = it.first;
        auto& requestInput = it.second;
        for (int i = 0; i < requestInput.tensor_shape().dim_size(); i++) {
            requestShape.push_back(requestInput.tensor_shape().dim(i).size());
        }
        requestShapes[name] = std::move(requestShape);
    }
    return requestShapes;
}

Status getModelInstance(ovms::ModelManager& manager,
    const std::string& modelName,
    ovms::model_version_t modelVersionId,
    std::shared_ptr<ovms::ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) {
    SPDLOG_DEBUG("Requesting model: {}; version: {}.", modelName, modelVersionId);

    auto model = manager.findModelByName(modelName);
    if (model == nullptr) {
        return StatusCode::MODEL_NAME_MISSING;
    }
    if (modelVersionId != 0) {
        modelInstance = model->getModelInstanceByVersion(modelVersionId);
        if (modelInstance == nullptr) {
            return StatusCode::MODEL_VERSION_MISSING;
        }
    } else {
        modelInstance = model->getDefaultModelInstance();
        if (modelInstance == nullptr) {
            return StatusCode::MODEL_VERSION_MISSING;
        }
    }

    return modelInstance->waitForLoaded(WAIT_FOR_MODEL_LOADED_TIMEOUT_MS, modelInstanceUnloadGuardPtr);
}

Status getPipeline(ovms::ModelManager& manager,
    std::unique_ptr<ovms::Pipeline>& pipelinePtr,
    const tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response) {

    SPDLOG_DEBUG("Requesting pipeline: {};", request->model_spec().name());
    auto status = manager.createPipeline(pipelinePtr, request->model_spec().name(), request, response);
    return status;
}

Status performInference(ovms::OVInferRequestsQueue& inferRequestsQueue, const int executingInferId, InferenceEngine::InferRequest& inferRequest) {
    try {
        inferRequest.StartAsync();
        InferenceEngine::StatusCode sts = inferRequest.Wait(InferenceEngine::IInferRequest::RESULT_READY);
        if (sts != InferenceEngine::StatusCode::OK) {
            Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
            SPDLOG_ERROR("Async infer failed {}: {}", status.string(), sts);
            return status;
        }
    } catch (const InferenceEngine::details::InferenceEngineException& e) {
        Status status = StatusCode::OV_INTERNAL_INFERENCE_ERROR;
        SPDLOG_ERROR("Async caught an exception {}: {}", status.string(), e.what());
        return status;
    }
    return StatusCode::OK;
}

Status inference(
    ModelInstance& modelVersion,
    const PredictRequest* requestProto,
    PredictResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    Timer timer;
    using std::chrono::microseconds;

    auto status = modelVersion.validate(requestProto);
    status = reloadModelIfRequired(status, modelVersion, requestProto, modelUnloadGuardPtr);
    if (!status.ok())
        return status;

    timer.start("get infer request");
    ovms::OVInferRequestsQueue& inferRequestsQueue = modelVersion.getInferRequestsQueue();
    ExecutingStreamIdGuard executingStreamIdGuard(inferRequestsQueue);
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);
    timer.stop("get infer request");
    SPDLOG_DEBUG("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), modelVersion.getVersion(), executingInferId, timer.elapsed<microseconds>("get infer request") / 1000);

    timer.start("deserialize");
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*requestProto, modelVersion.getInputsInfo(), inferRequest);
    timer.stop("deserialize");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), modelVersion.getVersion(), executingInferId, timer.elapsed<microseconds>("deserialize") / 1000);
    timer.start("prediction");
    status = performInference(inferRequestsQueue, executingInferId, inferRequest);
    timer.stop("prediction");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), modelVersion.getVersion(), executingInferId, timer.elapsed<microseconds>("prediction") / 1000);

    timer.start("serialize");
    status = serializePredictResponse(inferRequest, modelVersion.getOutputsInfo(), responseProto);
    timer.stop("serialize");
    if (!status.ok())
        return status;
    SPDLOG_DEBUG("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        requestProto->model_spec().name(), modelVersion.getVersion(), executingInferId, timer.elapsed<microseconds>("serialize") / 1000);

    return StatusCode::OK;
}

Status reloadModelIfRequired(
    Status validationStatus,
    ModelInstance& modelInstance,
    const PredictRequest* requestProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr) {
    Status status = validationStatus;
    if (status.batchSizeChangeRequired()) {
        status = modelInstance.reloadModel(getRequestBatchSize(requestProto), {}, modelUnloadGuardPtr);
        if (!status.ok()) {
            SPDLOG_ERROR("Model instance reload (batch size change) failed. Status Code: {}, Error {}", status.getCode(), status.string());
        }
    } else if (status.reshapeRequired()) {
        status = modelInstance.reloadModel(0, getRequestShapes(requestProto), modelUnloadGuardPtr);
        if (!status.ok() && status != StatusCode::RESHAPE_ERROR) {
            SPDLOG_ERROR("Model instance reload (reshape) failed. Status Code: {}, Error: {}", status.getCode(), status.string());
        }
    } else if (!status.ok()) {
        SPDLOG_WARN("Validation of inferRequest failed. Status Code: {}, Error: {}", status.getCode(), status.string());
    }
    return status;
}
}  // namespace ovms

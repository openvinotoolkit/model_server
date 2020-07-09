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
#include "modelmanager.hpp"
#include "modelinstance.hpp"
#include "deserialization.hpp"
#include "serialization.hpp"

#define DEBUG
#include "timer.hpp"

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

namespace ovms {

Status checkIfAvailable(const ModelInstance& modelInstance) {
    ModelVersionState modelVersionState = modelInstance.getStatus().getState();
    if (ModelVersionState::AVAILABLE < modelVersionState) {
        return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
    }
    if (ModelVersionState::AVAILABLE > modelVersionState) {
        return StatusCode::MODEL_VERSION_NOT_LOADED_YET;
    }
    return StatusCode::OK;
}

Status checkIfWillEndAsAvailable(const ModelInstance& modelInstance) {
    ModelVersionState modelVersionState = modelInstance.getStatus().getState();
    if (ModelVersionState::AVAILABLE < modelVersionState) {
        return StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE;
    }
    return StatusCode::OK;
}

Status waitIfNotLoadedYet(ModelInstance& modelInstance) {
    if (ModelVersionState::AVAILABLE > modelInstance.getStatus().getState()) {
        SPDLOG_INFO("Waiting for model:{} version:{} since it started loading again.", modelInstance.getName(), modelInstance.getVersion());
        if (!modelInstance.waitForLoaded()) {
            SPDLOG_INFO("Requested model:{} version:{} did not load within acceptable wait timeout.", modelInstance.getName(), modelInstance.getVersion());
            return StatusCode::MODEL_VERSION_NOT_LOADED_YET;
        }
    }
    return StatusCode::OK;
}

Status getModelInstance(ovms::ModelManager& manager,
                        const std::string& modelName,
                        ovms::model_version_t modelVersionId,
                        std::shared_ptr<ovms::ModelInstance>& modelInstance,
                        std::unique_ptr<ModelInstancePredictRequestsHandlesCountGuard>& modelInstancePredictRequestsHandlesCountGuardPtr) {
    spdlog::debug("Requesting model:{}; version:{}.", modelName, modelVersionId);

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
    // don't block modelInstance from unloading if already unloading
    Status status = checkIfWillEndAsAvailable(*modelInstance);
    if (!status.ok()) {
        return status;
    }
    status = waitIfNotLoadedYet(*modelInstance);
    if (!status.ok()) {
        return status;
    }
    modelInstancePredictRequestsHandlesCountGuardPtr = std::make_unique<ModelInstancePredictRequestsHandlesCountGuard>(*modelInstance);
    status = waitIfNotLoadedYet(*modelInstance);
    if (!status.ok()) {
        return status;
    }
    // Check model state to stop blocking model from unloading when state already changed from AVAILABLE. Unloading will be unblocked by
    // ModelInstancePredictRequestsHandlesCountGuard falling out of scope in main Predict()
    return checkIfAvailable(*modelInstance);
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
            ModelInstance   &modelVersion,
    const   PredictRequest  *request_proto,
            PredictResponse *response_proto) {
    Timer timer;
    using std::chrono::microseconds;

    auto status = modelVersion.validate(request_proto);
    if (!status.ok()) {
        SPDLOG_INFO("Validation of inferRequest failed. {}", status.string());
        return status;
    }

    timer.start("get infer request");
    ovms::OVInferRequestsQueue& inferRequestsQueue = modelVersion.getInferRequestsQueue();
    ExecutingStreamIdGuard executingStreamIdGuard(inferRequestsQueue);
    int executingInferId = executingStreamIdGuard.getId();
    InferenceEngine::InferRequest& inferRequest = inferRequestsQueue.getInferRequest(executingInferId);
    timer.stop("get infer request");
    spdlog::debug("Getting infer req duration in model {}, version {}, nireq {}: {:.3f} ms",
            request_proto->model_spec().name(), modelVersion.getVersion(), executingInferId, timer.elapsed<microseconds>("get infer request") / 1000);

    timer.start("deserialize");
    status = deserializePredictRequest<ConcreteTensorProtoDeserializator>(*request_proto, modelVersion.getInputsInfo(), inferRequest);
    timer.stop("deserialize");
    if (!status.ok())
        return status;
    spdlog::debug("Deserialization duration in model {}, version {}, nireq {}: {:.3f} ms",
        request_proto->model_spec().name(), modelVersion.getVersion(), executingInferId, timer.elapsed<microseconds>("deserialize") / 1000);

    timer.start("prediction");
    status = performInference(inferRequestsQueue, executingInferId, inferRequest);
    timer.stop("prediction");
    // TODO - current return code below is the same as in Python, but INVALID_ARGUMENT does not neccesarily mean
    // that the problem may be input
    if (!status.ok())
        return status;
    spdlog::debug("Prediction duration in model {}, version {}, nireq {}: {:.3f} ms",
            request_proto->model_spec().name(), modelVersion.getVersion(), executingInferId, timer.elapsed<microseconds>("prediction") / 1000);

    timer.start("serialize");
    status = serializePredictResponse(inferRequest, modelVersion.getOutputsInfo(), response_proto);
    timer.stop("serialize");
    if (!status.ok())
        return status;
    spdlog::debug("Serialization duration in model {}, version {}, nireq {}: {:.3f} ms",
            request_proto->model_spec().name(), modelVersion.getVersion(), executingInferId, timer.elapsed<microseconds>("serialize") / 1000);

    return StatusCode::OK;
}

}  // namespace ovms

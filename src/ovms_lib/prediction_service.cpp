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
#include "prediction_service.hpp"

#include <condition_variable>
#include <memory>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop

#include "get_model_metadata_impl.hpp"
#include "modelinstanceunloadguard.hpp"
#include "modelmanager.hpp"
#include "ovinferrequestsqueue.hpp"
#include "prediction_service_utils.hpp"
#include "profiler.hpp"
#include "servablemanagermodule.hpp"
#include "server.hpp"
#include "status.hpp"
#include "timer.hpp"

using grpc::ServerContext;

using tensorflow::TensorProto;

using tensorflow::serving::PredictionService;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

namespace ovms {

PredictionServiceImpl::PredictionServiceImpl(ovms::Server& ovmsServer) :
    ovmsServer(ovmsServer),
    getModelMetadataImpl(ovmsServer) {}

Status PredictionServiceImpl::getModelInstance(const PredictRequest* request,
    std::shared_ptr<ovms::ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return StatusCode::MODEL_NOT_LOADED;
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO if not succeed then return error
    auto& manager = servableManagerModule->getServableManager();
    return manager.getModelInstance(request->model_spec().name(), request->model_spec().version().value(), modelInstance, modelInstanceUnloadGuardPtr);
}

Status PredictionServiceImpl::getPipeline(const PredictRequest* request,
    PredictResponse* response,
    std::unique_ptr<ovms::Pipeline>& pipelinePtr) {
    OVMS_PROFILE_FUNCTION();
    auto module = this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME);
    if (nullptr == module) {
        return StatusCode::MODEL_NOT_LOADED;
    }
    auto servableManagerModule = dynamic_cast<const ServableManagerModule*>(module);
    // TODO if not succeed then return error
    auto& manager = servableManagerModule->getServableManager();
    return manager.createPipeline(pipelinePtr, request->model_spec().name(), request, response);
}

grpc::Status ovms::PredictionServiceImpl::Predict(
    ServerContext* context,
    const PredictRequest* request,
    PredictResponse* response) {
    OVMS_PROFILE_FUNCTION();
    Timer timer;
    timer.start("total");
    using std::chrono::microseconds;
    SPDLOG_DEBUG("Processing gRPC request for model: {}; version: {}",
        request->model_spec().name(),
        request->model_spec().version().value());

    std::shared_ptr<ovms::ModelInstance> modelInstance;
    std::unique_ptr<ovms::Pipeline> pipelinePtr;

    std::unique_ptr<ModelInstanceUnloadGuard> modelInstanceUnloadGuard;
    auto status = getModelInstance(request, modelInstance, modelInstanceUnloadGuard);

    if (status == StatusCode::MODEL_NAME_MISSING) {
        SPDLOG_DEBUG("Requested model: {} does not exist. Searching for pipeline with that name...", request->model_spec().name());
        status = getPipeline(request, response, pipelinePtr);
    }
    if (!status.ok()) {
        SPDLOG_INFO("Getting modelInstance or pipeline failed. {}", status.string());
        return status.grpc();
    }

    if (pipelinePtr) {
        status = pipelinePtr->execute();
    } else {
        status = modelInstance->infer(request, response, modelInstanceUnloadGuard);
    }

    if (!status.ok()) {
        return status.grpc();
    }

    timer.stop("total");
    SPDLOG_DEBUG("Total gRPC request processing time: {} ms", timer.elapsed<microseconds>("total") / 1000);
    return grpc::Status::OK;
}

grpc::Status PredictionServiceImpl::GetModelMetadata(
    grpc::ServerContext* context,
    const tensorflow::serving::GetModelMetadataRequest* request,
    tensorflow::serving::GetModelMetadataResponse* response) {
    OVMS_PROFILE_FUNCTION();
    return getModelMetadataImpl.getModelStatus(request, response).grpc();
}

const GetModelMetadataImpl& PredictionServiceImpl::getTFSModelMetadataImpl() const {
    return this->getModelMetadataImpl;
}

}  // namespace ovms

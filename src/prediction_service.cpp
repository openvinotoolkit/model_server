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

#include "dags/pipeline.hpp"
#include "execution_context.hpp"
#include "get_model_metadata_impl.hpp"
#include "grpc_utils.hpp"
#include "modelinstance.hpp"
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

namespace {
enum : unsigned int {
    TOTAL,
    TIMER_END
};
}

namespace ovms {

PredictionServiceImpl::PredictionServiceImpl(ovms::Server& ovmsServer) :
    ovmsServer(ovmsServer),
    getModelMetadataImpl(ovmsServer),
    modelManager(dynamic_cast<const ServableManagerModule*>(this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME))->getServableManager()) {
    if (nullptr == this->ovmsServer.getModule(SERVABLE_MANAGER_MODULE_NAME)) {
        const char* message = "Tried to create prediction service impl without servable manager module";
        SPDLOG_ERROR(message);
        throw std::logic_error(message);
    }
}

Status PredictionServiceImpl::getModelInstance(const PredictRequest* request,
    std::shared_ptr<ovms::ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr) {
    OVMS_PROFILE_FUNCTION();
    return this->modelManager.getModelInstance(request->model_spec().name(), request->model_spec().version().value(), modelInstance, modelInstanceUnloadGuardPtr);
}

Status PredictionServiceImpl::getPipeline(const PredictRequest* request,
    PredictResponse* response,
    std::unique_ptr<ovms::Pipeline>& pipelinePtr) {
    OVMS_PROFILE_FUNCTION();
    return this->modelManager.createPipeline(pipelinePtr, request->model_spec().name(), request, response);
}

grpc::Status ovms::PredictionServiceImpl::Predict(
    ServerContext* context,
    const PredictRequest* request,
    PredictResponse* response) {
    OVMS_PROFILE_FUNCTION();
    Timer<TIMER_END> timer;
    timer.start(TOTAL);
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
        if (modelInstance) {
            INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().requestFailGrpcPredict);
        }
        SPDLOG_DEBUG("Getting modelInstance or pipeline failed. {}", status.string());
        return grpc(status);
    }

    ExecutionContext executionContext{
        ExecutionContext::Interface::GRPC,
        ExecutionContext::Method::Predict};

    if (pipelinePtr) {
        status = pipelinePtr->execute(executionContext);
        INCREMENT_IF_ENABLED(pipelinePtr->getMetricReporter().getInferRequestMetric(executionContext, status.ok()));
    } else {
        status = modelInstance->infer(request, response, modelInstanceUnloadGuard);
        INCREMENT_IF_ENABLED(modelInstance->getMetricReporter().getInferRequestMetric(executionContext, status.ok()));
    }

    if (!status.ok()) {
        return grpc(status);
    }

    timer.stop(TOTAL);
    double requestTotal = timer.elapsed<microseconds>(TOTAL);
    if (pipelinePtr) {
        OBSERVE_IF_ENABLED(pipelinePtr->getMetricReporter().requestTimeGrpc, requestTotal);
    } else {
        OBSERVE_IF_ENABLED(modelInstance->getMetricReporter().requestTimeGrpc, requestTotal);
    }
    SPDLOG_DEBUG("Total gRPC request processing time: {} ms", requestTotal / 1000);
    return grpc::Status::OK;
}

grpc::Status PredictionServiceImpl::GetModelMetadata(
    grpc::ServerContext* context,
    const tensorflow::serving::GetModelMetadataRequest* request,
    tensorflow::serving::GetModelMetadataResponse* response) {
    OVMS_PROFILE_FUNCTION();
    return grpc(getModelMetadataImpl.getModelStatus(request, response, ExecutionContext(ExecutionContext::Interface::GRPC, ExecutionContext::Method::GetModelMetadata)));
}

const GetModelMetadataImpl& PredictionServiceImpl::getTFSModelMetadataImpl() const {
    return this->getModelMetadataImpl;
}

}  // namespace ovms

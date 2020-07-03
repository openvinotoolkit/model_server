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

#include "get_model_metadata_impl.hpp"
#include "modelmanager.hpp"
#include "ovinferrequestsqueue.hpp"
#include "prediction_service.hpp"
#include "prediction_service_utils.hpp"
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

grpc::Status ovms::PredictionServiceImpl::Predict(
            ServerContext*      context,
    const   PredictRequest*     request,
            PredictResponse*    response) {
    Timer timer;
    timer.start("total");
    using std::chrono::microseconds;
    spdlog::debug("Processing gRPC request for model: {}; version: {}",
                  request->model_spec().name(),
                  request->model_spec().version().value());

    std::shared_ptr<ovms::ModelInstance> modelInstance;

    std::unique_ptr<ModelInstancePredictRequestsHandlesCountGuard> modelInstancePredictRequestsHandlesCountGuard;
    auto status = getModelInstance(request, modelInstance, modelInstancePredictRequestsHandlesCountGuard);
    if (!status.ok()) {
        SPDLOG_INFO("Getting modelInstance failed. {}", status.string());
        return status.grpc();
    }

    // status = assureModelInstanceLoadedWithProperBatchSize(*modelVersion, /*requestBatchSize*/, modelInstancePredictRequestsHandlesCountGuard1);
    // if (!status.ok()) {
    //     SPDLOG_INFO("Assuring modelInstance is loaded with proper meta parameters failed. {}", status.string());
    //     return status.grpc();
    // }

    status = inference(*modelInstance, request, response);
    if (!status.ok()) {
        return status.grpc();
    }

    timer.stop("total");
    spdlog::debug("Total time: {} ms", timer.elapsed<microseconds>("total") / 1000);
    return grpc::Status::OK;
}

grpc::Status PredictionServiceImpl::GetModelMetadata(
            grpc::ServerContext*                            context,
    const   tensorflow::serving::GetModelMetadataRequest*   request,
            tensorflow::serving::GetModelMetadataResponse*  response) {
    return GetModelMetadataImpl::getModelStatus(request, response).grpc();
}

}  // namespace ovms

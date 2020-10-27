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
#pragma once
#include <map>
#include <memory>
#include <string>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "modelinstance.hpp"
#include "modelmanager.hpp"

namespace ovms {

const uint WAIT_FOR_MODEL_LOADED_TIMEOUT_MS = 10000;

size_t getRequestBatchSize(const tensorflow::serving::PredictRequest* request);
std::map<std::string, shape_t> getRequestShapes(const tensorflow::serving::PredictRequest* request);

Status getModelInstance(ModelManager& manager,
    const std::string& modelName,
    model_version_t modelVersionId,
    std::shared_ptr<ModelInstance>& modelInstance,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelInstanceUnloadGuardPtr);

Status getPipeline(ModelManager& manager,
    std::unique_ptr<Pipeline>& pipelinePtr,
    const tensorflow::serving::PredictRequest* request,
    tensorflow::serving::PredictResponse* response);

Status performInference(ovms::OVInferRequestsQueue& inferRequestsQueue, const int executingInferId, InferenceEngine::InferRequest& inferRequest);

Status inference(
    ModelInstance& modelVersion,
    const tensorflow::serving::PredictRequest* requestProto,
    tensorflow::serving::PredictResponse* responseProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr);

Status reloadModelIfRequired(
    Status validationStatus,
    ModelInstance& modelInstance,
    const tensorflow::serving::PredictRequest* requestProto,
    std::unique_ptr<ModelInstanceUnloadGuard>& modelUnloadGuardPtr);
}  // namespace ovms

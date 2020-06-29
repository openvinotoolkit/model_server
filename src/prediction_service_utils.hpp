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
#include <memory>
#include <string>

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "modelmanager.hpp"
#include "modelinstance.hpp"

namespace ovms {

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

class ModelInstancePredictRequestsHandlesCountGuard {
public:
    ModelInstancePredictRequestsHandlesCountGuard(ModelInstance& modelInstance) : modelInstance(modelInstance) {
        modelInstance.increasePredictRequestsHandlesCount();
    }
    ~ModelInstancePredictRequestsHandlesCountGuard() {
        modelInstance.decreasePredictRequestsHandlesCount();
    }
private:
    ModelInstance& modelInstance;
};

Status getModelInstance(ModelManager& manager,
                        const std::string& modelName,
                        model_version_t modelVersionId,
                        std::shared_ptr<ModelInstance>& modelInstance,
                        std::unique_ptr<ModelInstancePredictRequestsHandlesCountGuard>& modelInstancePredictRequestsHandlesCountGuardPtr);

Status performInference(ovms::OVInferRequestsQueue& inferRequestsQueue, const int executingInferId, InferenceEngine::InferRequest& inferRequest);
}  // namespace ovms

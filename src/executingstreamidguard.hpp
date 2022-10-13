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

namespace ov {
class InferRequest;
}

namespace ovms {
class OVInferRequestsQueue;

class ModelMetricReporter;
class OVInferRequestsQueue;

struct ExecutingStreamIdGuard {
    ExecutingStreamIdGuard(ovms::OVInferRequestsQueue& inferRequestsQueue, ModelMetricReporter& reporter);
    ~ExecutingStreamIdGuard();

    int getId();
    ov::InferRequest& getInferRequest();

private:
    class CurrentRequestsMetricGuard {
        ModelMetricReporter& reporter;

    public:
        CurrentRequestsMetricGuard(ModelMetricReporter& reporter);
        ~CurrentRequestsMetricGuard();
    };

    CurrentRequestsMetricGuard currentRequestsMetricGuard;
    OVInferRequestsQueue& inferRequestsQueue_;
    const int id_;
    ov::InferRequest& inferRequest;
    ModelMetricReporter& reporter;
};

}  //  namespace ovms

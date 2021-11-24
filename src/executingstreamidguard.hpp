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

#include "ovinferrequestsqueue.hpp"

namespace ovms {

struct ExecutingStreamIdGuard {
    ExecutingStreamIdGuard(ovms::OVInferRequestsQueue& inferRequestsQueue) :
        inferRequestsQueue_(inferRequestsQueue),
        id_(inferRequestsQueue_.getIdleStream().get()),
        inferRequest(inferRequestsQueue.getInferRequest(id_)) {}
    ~ExecutingStreamIdGuard() {
        inferRequestsQueue_.returnStream(id_);
    }
    int getId() { return id_; }
    InferenceEngine::InferRequest& getInferRequest() { return inferRequest; }

private:
    ovms::OVInferRequestsQueue& inferRequestsQueue_;
    const int id_;
    InferenceEngine::InferRequest& inferRequest;
};

struct ExecutingStreamIdGuard_2 {
    ExecutingStreamIdGuard_2(ovms::OVInferRequestsQueue_2& inferRequestsQueue) :
        inferRequestsQueue_(inferRequestsQueue),
        id_(inferRequestsQueue_.getIdleStream().get()),
        inferRequest(inferRequestsQueue.getInferRequest(id_)) {}
    ~ExecutingStreamIdGuard_2() {
        inferRequestsQueue_.returnStream(id_);
    }
    int getId() { return id_; }
    ov::runtime::InferRequest& getInferRequest() { return inferRequest; }

private:
    ovms::OVInferRequestsQueue_2& inferRequestsQueue_;
    const int id_;
    ov::runtime::InferRequest& inferRequest;
};

}  //  namespace ovms

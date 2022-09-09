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

#include "model_metric_reporter.hpp"
#include "ovinferrequestsqueue.hpp"

namespace ovms {

class CurrentRequestsMetricGuard {
    ModelMetricReporter& reporter;

public:
    CurrentRequestsMetricGuard(ModelMetricReporter& reporter) :
        reporter(reporter) {
        INCREMENT_IF_ENABLED(reporter.currentRequests);
    }
    ~CurrentRequestsMetricGuard() {
        DECREMENT_IF_ENABLED(reporter.currentRequests);
    }
};

struct ExecutingStreamIdGuard {
    ExecutingStreamIdGuard(ovms::OVInferRequestsQueue& inferRequestsQueue, ModelMetricReporter& reporter) :
        metricGuard(reporter),
        inferRequestsQueue_(inferRequestsQueue),
        id_(inferRequestsQueue_.getIdleStream().get()),
        inferRequest(inferRequestsQueue.getInferRequest(id_)),
        reporter(reporter) {
        INCREMENT_IF_ENABLED(reporter.inferReqActive);
    }
    ~ExecutingStreamIdGuard() {
        DECREMENT_IF_ENABLED(reporter.inferReqActive);
        inferRequestsQueue_.returnStream(id_);
    }
    int getId() { return id_; }
    ov::InferRequest& getInferRequest() { return inferRequest; }

private:
    CurrentRequestsMetricGuard metricGuard;
    ovms::OVInferRequestsQueue& inferRequestsQueue_;
    const int id_;
    ov::InferRequest& inferRequest;
    ModelMetricReporter& reporter;
};

}  //  namespace ovms

//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include "executingstreamidguard.hpp"

#include "logging.hpp"
#include "model_metric_reporter.hpp"
#include "ovinferrequestsqueue.hpp"

namespace ovms {

ExecutingStreamIdGuard::CurrentRequestsMetricGuard::CurrentRequestsMetricGuard(ModelMetricReporter& reporter) :
    reporter(reporter) {
    INCREMENT_IF_ENABLED(this->reporter.currentRequests);
}

ExecutingStreamIdGuard::CurrentRequestsMetricGuard::~CurrentRequestsMetricGuard() {
    DECREMENT_IF_ENABLED(this->reporter.currentRequests);
}

ExecutingStreamIdGuard::ExecutingStreamIdGuard(OVInferRequestsQueue& inferRequestsQueue, ModelMetricReporter& reporter) :
    StreamIdGuard(inferRequestsQueue),
    currentRequestsMetricGuard(reporter),
    reporter(reporter) {
    INCREMENT_IF_ENABLED(this->reporter.inferReqActive);
}

ExecutingStreamIdGuard::~ExecutingStreamIdGuard() {
    DECREMENT_IF_ENABLED(this->reporter.inferReqActive);
}

StreamIdGuard::StreamIdGuard(OVInferRequestsQueue& inferRequestsQueue) :
    inferRequestsQueue_(inferRequestsQueue),
    id_(inferRequestsQueue_.getIdleStream().get()),
    inferRequest(inferRequestsQueue.getInferRequest(id_)) {
    SPDLOG_TRACE("Got request id:{}", getId());
}

StreamIdGuard::~StreamIdGuard() {
    this->inferRequestsQueue_.returnStream(this->id_);
}

int StreamIdGuard::getId() { return this->id_; }
ov::InferRequest& StreamIdGuard::getInferRequest() { return this->inferRequest; }

}  //  namespace ovms

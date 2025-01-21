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
#include "nodestreamidguard.hpp"

#include <future>
#include <optional>

#include "../logging.hpp"
#include "../model_metric_reporter.hpp"
#include "../ovinferrequestsqueue.hpp"
#include "../profiler.hpp"

namespace ovms {

NodeStreamIdGuard::NodeStreamIdGuard(OVInferRequestsQueue& inferRequestsQueue, ModelMetricReporter& reporter) :
    inferRequestsQueue_(inferRequestsQueue),
    futureStreamId(inferRequestsQueue_.getIdleStream()),
    reporter(reporter) {
    INCREMENT_IF_ENABLED(this->reporter.currentRequests);
}

NodeStreamIdGuard::~NodeStreamIdGuard() {
    if (!this->disarmed) {
        if (!this->streamId) {
            SPDLOG_DEBUG("Trying to disarm stream Id that is not needed anymore...");
            this->streamId = this->futureStreamId.get();
            INCREMENT_IF_ENABLED(this->reporter.inferReqActive);
        }
        SPDLOG_DEBUG("Returning streamId: {}", this->streamId.value());
        DECREMENT_IF_ENABLED(this->reporter.inferReqActive);
        this->inferRequestsQueue_.returnStream(this->streamId.value());
        DECREMENT_IF_ENABLED(this->reporter.currentRequests);
    }
}

std::optional<int> NodeStreamIdGuard::tryGetId(const uint32_t microseconds) {
    OVMS_PROFILE_FUNCTION();
    if (!this->streamId) {
        if (std::future_status::ready == this->futureStreamId.wait_for(std::chrono::microseconds(microseconds))) {
            this->streamId = this->futureStreamId.get();
            INCREMENT_IF_ENABLED(this->reporter.inferReqActive);
        }
    }
    return this->streamId;
}

bool NodeStreamIdGuard::tryDisarm(const uint32_t microseconds) {
    if (std::future_status::ready == this->futureStreamId.wait_for(std::chrono::microseconds(microseconds))) {
        this->streamId = this->futureStreamId.get();
        SPDLOG_DEBUG("Returning streamId:", this->streamId.value());
        this->inferRequestsQueue_.returnStream(this->streamId.value());
        DECREMENT_IF_ENABLED(this->reporter.currentRequests);
        this->disarmed = true;
    }
    return this->disarmed;
}

}  // namespace ovms

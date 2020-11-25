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

#include <future>
#include <optional>

#include <spdlog/spdlog.h>

#include "ovinferrequestsqueue.hpp"

namespace ovms {
struct NodeStreamIdGuard {
    NodeStreamIdGuard(ovms::OVInferRequestsQueue& inferRequestsQueue) :
        inferRequestsQueue_(inferRequestsQueue),
        futureStreamId(inferRequestsQueue_.getIdleStream()) {}

    ~NodeStreamIdGuard() {
        if (!disarmed) {
            if (!streamId) {
                SPDLOG_DEBUG("Trying to disarm stream Id that is not needed anymore...");
                streamId = futureStreamId.get();
            }
            SPDLOG_DEBUG("Returning streamId: {}", streamId.value());
            inferRequestsQueue_.returnStream(streamId.value());
        }
    }

    std::optional<int> tryGetId(const uint microseconds = 1) {
        if (!streamId) {
            if (std::future_status::ready == futureStreamId.wait_for(std::chrono::microseconds(microseconds))) {
                streamId = futureStreamId.get();
            }
        }
        return streamId;
    }

    bool tryDisarm(const uint microseconds = 1) {
        if (std::future_status::ready == futureStreamId.wait_for(std::chrono::microseconds(microseconds))) {
            streamId = futureStreamId.get();
            SPDLOG_DEBUG("Returning streamId:", streamId.value());
            inferRequestsQueue_.returnStream(streamId.value());
            disarmed = true;
        }
        return disarmed;
    }

private:
    ovms::OVInferRequestsQueue& inferRequestsQueue_;
    std::future<int> futureStreamId;
    std::optional<int> streamId = std::nullopt;
    bool disarmed = false;
};
}  // namespace ovms

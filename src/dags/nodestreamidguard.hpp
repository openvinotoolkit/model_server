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

namespace ovms {
class OVInferRequestsQueue;

class ModelMetricReporter;
class OVInferRequestsQueue;

struct NodeStreamIdGuard {
    NodeStreamIdGuard(OVInferRequestsQueue& inferRequestsQueue, ModelMetricReporter& reporter);
    ~NodeStreamIdGuard();

    std::optional<int> tryGetId(const uint32_t microseconds = 1);
    bool tryDisarm(const uint32_t microseconds = 1);

private:
    OVInferRequestsQueue& inferRequestsQueue_;
    std::future<int> futureStreamId;
    std::optional<int> streamId = std::nullopt;
    bool disarmed = false;
    ModelMetricReporter& reporter;
};
}  // namespace ovms

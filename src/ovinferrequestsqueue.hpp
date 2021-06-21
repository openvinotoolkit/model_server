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

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

namespace InferenceEngine {
class InferRequest;
class ExecutableNetwork;
}  // namespace InferenceEngine

namespace ovms {
/**
* @brief Class representing circular buffer for managing IE ireqs
*/
class OVInferRequestsQueue {
public:
    /**
    * @brief Allocating idle ireqs for execution
    */
    std::future<int> getIdleStream();

    /**
    * @brief Release ireqs after execution
    */
    void returnStream(int ireqId);

    /**
    * @brief Constructor with initialization
    */
    OVInferRequestsQueue(InferenceEngine::ExecutableNetwork& network, int nireq);

    ~OVInferRequestsQueue();

    /**
     * @brief Give InferRequest
     */
    InferenceEngine::InferRequest& getInferRequest(int ireqId) {
        return *inferRequests[ireqId];
    }

protected:
    /**
    * @brief Vector representing circular buffer for infer queue
    */
    std::vector<int> ireqs;

    /**
    * @brief Index of the front of the idle ireqs list
    */
    std::uint32_t front_idx;

    /**
    * @brief Index of the back of the idle ireqs list
    */
    std::atomic<std::uint32_t> back_idx;

    std::mutex front_mut;
    std::mutex queue_mutex;
    /**
    * @brief Vector representing OV ireqs and used for notification about completed inference operations
    */
    std::vector<std::unique_ptr<InferenceEngine::InferRequest>> inferRequests;
    std::queue<std::promise<int>> promises;
    InferenceEngine::ExecutableNetwork& network;
};
}  // namespace ovms

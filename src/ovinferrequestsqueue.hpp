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

#include <inference_engine.hpp>
#include <spdlog/spdlog.h>

namespace ovms {
/**
* @brief Class representing circular buffer for managing IE streams
*/
class OVInferRequestsQueue {
public:
    /**
    * @brief Allocating idle stream for execution
    */
    std::future<int> getIdleStream();

    /**
    * @brief Release stream after execution
    */
    void returnStream(int streamID);

    /**
    * @brief Constructor with initialization
    */
    OVInferRequestsQueue(InferenceEngine::ExecutableNetwork& network, int streamsLength) :
        streams(streamsLength),
        front_idx{0},
        back_idx{0} {
        for (int i = 0; i < streamsLength; ++i) {
            streams[i] = i;
            inferRequests.push_back(network.CreateInferRequest());
        }
    }

    /**
     * @brief Give InferRequest
     */
    InferenceEngine::InferRequest& getInferRequest(int streamID) {
        return inferRequests[streamID];
    }

protected:
    /**
    * @brief Vector representing circular buffer for infer queue
    */
    std::vector<int> streams;

    /**
    * @brief Index of the front of the idle streams list
    */
    std::uint32_t front_idx;

    /**
    * @brief Index of the back of the idle streams list
    */
    std::atomic<std::uint32_t> back_idx;

    /**
    * @brief Vector representing OV streams and used for notification about completed inference operations
    */
    std::mutex front_mut;
    std::mutex queue_mutex;
    /**
     * 
     */
    std::vector<InferenceEngine::InferRequest> inferRequests;
    std::queue<std::promise<int>> promises;
};
}  // namespace ovms

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
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#include "queue.hpp"

namespace ovms {

class OVInferRequestsQueue : public Queue<InferenceEngine::InferRequest> {
public:
    OVInferRequestsQueue(InferenceEngine::ExecutableNetwork& network, int streamsLength) :
        Queue(streamsLength) {
        for (int i = 0; i < streamsLength; ++i) {
            streams[i] = i;
            inferRequests.push_back(network.CreateInferRequest());
        }
    }
};

class OVInferRequestsQueue_2 : public Queue<ov::runtime::InferRequest> {
public:
    OVInferRequestsQueue_2(ov::runtime::ExecutableNetwork& network, int streamsLength) :
        Queue(streamsLength) {
        for (int i = 0; i < streamsLength; ++i) {
            streams[i] = i;
            inferRequests.push_back(network.create_infer_request());
        }
    }
};

}  // namespace ovms

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

#include "ovinferrequestsqueue.hpp"

namespace ovms {

void OVInferRequestsQueue::signalCompletedInference(int streamID) {
    activeStreams[streamID].notify_one();
}

void OVInferRequestsQueue::waitForAsync(int streamID) {
    std::mutex mx;
    std::unique_lock <std::mutex> lock(mx);
    activeStreams[streamID].wait(lock);
}

int OVInferRequestsQueue::getIdleStream() {
    int value;
    std::unique_lock <std::mutex> lk(front_mut);
    if (streams[front_idx] < 0) {
        not_full_cond.wait(lk, [this] { return streams[front_idx] >= 0; });
    }
    value = streams[front_idx];
    streams[front_idx] = -1;  // negative value indicate consumed vector index
    front_idx = (front_idx + 1) % streams.size();
    return value;
}

void OVInferRequestsQueue::returnStream(int streamID) {
    std::uint32_t old_back = back_idx.load();
    while (!back_idx.compare_exchange_weak(
            old_back,
            (old_back + 1) % streams.size(),
            std::memory_order_relaxed))
    {}
    streams[old_back] = streamID;
    not_full_cond.notify_one();
}

}  // namespace ovms

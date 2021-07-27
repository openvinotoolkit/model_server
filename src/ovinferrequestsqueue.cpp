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

#include <utility>

namespace ovms {
std::future<int> OVInferRequestsQueue::getIdleStream() {
    int value;
    std::promise<int> idleStreamPromise;
    std::future<int> idleStreamFuture = idleStreamPromise.get_future();
    std::unique_lock<std::mutex> lk(front_mut);
    if (streams[front_idx] < 0) {  // we need to wait for any idle stream to be returned
        std::unique_lock<std::mutex> queueLock(queue_mutex);
        promises.push(std::move(idleStreamPromise));
    } else {  // we can give idle stream right away
        value = streams[front_idx];
        streams[front_idx] = -1;  // negative value indicate consumed vector index
        front_idx = (front_idx + 1) % streams.size();
        lk.unlock();
        idleStreamPromise.set_value(value);
    }
    return idleStreamFuture;
}

void OVInferRequestsQueue::returnStream(int streamID) {
    std::unique_lock<std::mutex> lk(queue_mutex);
    if (promises.size()) {
        std::promise<int> promise = std::move(promises.front());
        promises.pop();
        lk.unlock();
        promise.set_value(streamID);
        return;
    }
    std::uint32_t old_back = back_idx.load();
    while (!back_idx.compare_exchange_weak(
        old_back,
        (old_back + 1) % streams.size(),
        std::memory_order_relaxed)) {
    }
    streams[old_back] = streamID;
}

}  // namespace ovms

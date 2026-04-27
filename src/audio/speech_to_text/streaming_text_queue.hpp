//*****************************************************************************
// Copyright 2025 Intel Corporation
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

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <utility>

namespace ovms {

// Thread-safe queue for streaming partial results from a background
// generation thread to the MediaPipe LOOPBACK loop.  Used by S2tCalculator
// to bridge ov::genai streamer callbacks and the calculator's Process() cycle.
class StreamingTextQueue {
public:
    void push(std::string text) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(text));
        cv_.notify_one();
    }

    // Signals that generation has finished (successfully or with error).
    void setDone() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        cv_.notify_one();
    }

    // Blocks until a text chunk is available or generation is done.
    // Returns true if a chunk was retrieved, false if done and queue is empty.
    bool waitAndPop(std::string& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (!queue_.empty()) {
            out = std::move(queue_.front());
            queue_.pop();
            return true;
        }
        return false;  // done and empty
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::string> queue_;
    bool done_ = false;
};

}  // namespace ovms

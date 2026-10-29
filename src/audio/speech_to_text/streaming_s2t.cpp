//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "streaming_s2t.hpp"

#include <stdexcept>
#include <utility>

#include "src/logging.hpp"

namespace ovms {
namespace {
constexpr size_t MaxEvents = 1024;
constexpr size_t MaxResultBytes = 1024 * 1024;
}

StreamingS2TBridge::StreamingS2TBridge(Wake wake) :
    wake_(std::move(wake)) {}

void StreamingS2TBridge::begin() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!failure_.ok())
        throw std::runtime_error(failure_.ToString());
    if (!wake_)
        throw std::runtime_error("Streaming STT graph is closed");
    ++active_;
}

void StreamingS2TBridge::complete() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_ > 0)
        --active_;
    changed_.notify_all();
}

void StreamingS2TBridge::publish(StreamingS2TEvent event) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!wake_)
        return;  // Expected after graph cancellation.
    if (!failure_.ok())
        return;
    if (events_.size() >= MaxEvents || event.text.size() > MaxResultBytes - bytes_) {
        SPDLOG_ERROR("Streaming STT result queue overflow");
        events_.clear();
        bytes_ = 0;
        failure_ = absl::ResourceExhaustedError("Streaming STT result queue is full");
        events_.push_back({StreamingS2TEvent::Type::Error, std::move(event.utteranceId),
            "Streaming STT result queue is full"});
        bytes_ = events_.front().text.size();
    } else {
        bytes_ += event.text.size();
        events_.push_back(std::move(event));
    }
    // Queue insertion, wake scheduling and drain acknowledgement share the same lock.
    if (!wakePending_) {
        wakePending_ = true;
        const auto status = wake_();
        if (!status.ok()) {
            failure_ = status;
            SPDLOG_ERROR("Streaming STT graph wakeup failed: {}", status.ToString());
        }
    }
    changed_.notify_all();
}

std::vector<StreamingS2TEvent> StreamingS2TBridge::drain() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<StreamingS2TEvent> result;
    result.swap(events_);
    bytes_ = 0;
    wakePending_ = false;
    return result;
}

absl::Status StreamingS2TBridge::waitFor(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait_for(lock, timeout, [this] { return active_ == 0 || !failure_.ok() || !wake_; });
    return failure_;
}

bool StreamingS2TBridge::busy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_ != 0;
}

void StreamingS2TBridge::invalidate() {
    std::lock_guard<std::mutex> lock(mutex_);
    wake_ = {};
    events_.clear();
    bytes_ = 0;
    active_ = 0;
    changed_.notify_all();
}
}  // namespace ovms

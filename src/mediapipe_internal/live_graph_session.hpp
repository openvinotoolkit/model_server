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
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <utility>

#include "src/status.hpp"

namespace ovms {

/// Immutable, shared payload with checked type access, independent of MediaPipe.
class LiveGraphPacket {
public:
    template <typename T>
    static LiveGraphPacket make(T value) {
        return share<T>(std::make_shared<const T>(std::move(value)));
    }

    template <typename T>
    static LiveGraphPacket share(std::shared_ptr<const T> value) {
        return LiveGraphPacket(std::move(value), typeid(T));
    }

    template <typename T>
    const T* getIf() const {
        return type_ == std::type_index(typeid(T)) ? static_cast<const T*>(value_.get()) : nullptr;
    }

private:
    LiveGraphPacket(std::shared_ptr<const void> value, std::type_index type) :
        value_(std::move(value)), type_(type) {}

    std::shared_ptr<const void> value_;
    std::type_index type_;
};

/// One managed execution of a named graph profile. The session retains its model
/// resources and active-inference lease even if the creating executor is destroyed.
class LiveGraphSession {
public:
    /// Observers run on graph workers; they must not destroy the session or block on it.
    using Observer = std::function<Status(const std::string& streamName, const LiveGraphPacket& packet)>;

    virtual ~LiveGraphSession() = default;
    /// Sends to a declared input with a monotonically increasing session timestamp.
    /// Payload types are checked before enqueueing. Queue overflow returns an error.
    virtual Status send(const std::string& streamName, const LiveGraphPacket& packet) = 0;
    /// Closes one input permanently; domain-specific boundaries are ordinary packets.
    virtual Status closeInput(const std::string& streamName) = 0;
    /// Requests cancellation without joining workers. Destruction waits for shutdown.
    virtual void cancel() = 0;
};

}  // namespace ovms

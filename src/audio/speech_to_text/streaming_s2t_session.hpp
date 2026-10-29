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

#include "absl/status/status.h"
#include "streaming_s2t.hpp"

namespace ovms {
struct SttServable;

/// One live graph execution sharing a configured STT servable.
class StreamingS2TSession {
public:
    using Observer = std::function<void(const StreamingS2TEvent&)>;
    StreamingS2TSession(std::shared_ptr<SttServable> servable, Observer observer);
    ~StreamingS2TSession();
    absl::Status push(const std::string& utteranceId, AudioChunk audio);
    absl::Status finish(const std::string& utteranceId);
    /// Invalidates callback wakeups and requests graph cancellation without joining.
    void cancel();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace ovms

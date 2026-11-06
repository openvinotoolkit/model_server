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

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "src/webrtc/streaming_audio_model.hpp"

namespace ovms {

struct StreamingS2TInput {
    enum class Type { Audio, Finish, Cancel };
    Type type;
    std::string utteranceId;
    std::shared_ptr<AudioChunk> audio;
};

struct StreamingS2TEvent {
    enum class Type { Delta, Final, Error };
    Type type;
    std::string utteranceId;
    std::string text;
};

/// Per-execution callback bridge. No worker holds a CalculatorContext or graph pointer.
class StreamingS2TBridge {
public:
    using Wake = std::function<absl::Status()>;

    explicit StreamingS2TBridge(Wake wake);
    void begin();
    void complete();
    void publish(StreamingS2TEvent event);
    std::vector<StreamingS2TEvent> drain();
    absl::Status waitFor(std::chrono::milliseconds timeout);
    bool busy() const;
    void invalidate();

private:
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    Wake wake_;
    std::vector<StreamingS2TEvent> events_;
    size_t bytes_ = 0;
    size_t active_ = 0;
    bool wakePending_ = false;
    absl::Status failure_;
};

using StreamingS2TBridgeMap = std::unordered_map<std::string, std::shared_ptr<StreamingS2TBridge>>;

}  // namespace ovms

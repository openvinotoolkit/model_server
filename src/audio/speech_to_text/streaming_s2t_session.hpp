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

#include "src/mediapipe_internal/live_graph_session.hpp"
#include "streaming_s2t.hpp"

namespace ovms {
class MediapipeGraphExecutorInterface;

inline constexpr const char* STT_INCREMENTAL_PROFILE = "speech-to-text.incremental";
inline constexpr const char* STT_UTTERANCE_PROFILE = "speech-to-text.utterance";
enum class AudioInputMode { Unsupported, Utterance, Incremental };

/// Typed STT adapter over a generic managed graph session; no model resources are exposed.
class StreamingS2TSession {
public:
    using Observer = std::function<void(const StreamingS2TEvent&)>;
    static AudioInputMode inputMode(const MediapipeGraphExecutorInterface& executor);
    /// Acquires a session lease; the executor need not outlive the resulting adapter.
    static Status create(MediapipeGraphExecutorInterface& executor, Observer observer,
        std::unique_ptr<StreamingS2TSession>& session);
    AudioInputMode inputMode() const { return mode_; }
    Status push(const std::string& utteranceId, AudioChunk audio);
    Status finish(const std::string& utteranceId);
    void cancel();

private:
    StreamingS2TSession(std::unique_ptr<LiveGraphSession> session, AudioInputMode mode);
    std::unique_ptr<LiveGraphSession> session_;
    AudioInputMode mode_;
};
}  // namespace ovms

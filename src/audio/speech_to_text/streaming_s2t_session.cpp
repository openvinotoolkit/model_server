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
#include "streaming_s2t_session.hpp"

#include <utility>

#include "src/mediapipe_internal/mediapipe_graph_executor_interface.hpp"

namespace ovms {

StreamingS2TSession::StreamingS2TSession(std::unique_ptr<LiveGraphSession> session, AudioInputMode mode) :
    session_(std::move(session)), mode_(mode) {}

AudioInputMode StreamingS2TSession::inputMode(const MediapipeGraphExecutorInterface& executor) {
    if (executor.supportsSession(STT_INCREMENTAL_PROFILE))
        return AudioInputMode::Incremental;
    if (executor.supportsSession(STT_UTTERANCE_PROFILE))
        return AudioInputMode::Utterance;
    return AudioInputMode::Unsupported;
}

Status StreamingS2TSession::create(MediapipeGraphExecutorInterface& executor, Observer observer,
    std::unique_ptr<StreamingS2TSession>& session) {
    const AudioInputMode mode = inputMode(executor);
    if (mode == AudioInputMode::Unsupported)
        return Status(StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, "Graph has no live STT profile");
    if (!observer || session)
        return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "STT requires an observer and empty session output");
    std::unique_ptr<LiveGraphSession> liveSession;
    const auto status = executor.startSession(
        mode == AudioInputMode::Incremental ? STT_INCREMENTAL_PROFILE : STT_UTTERANCE_PROFILE,
        [observer = std::move(observer)](const std::string& stream, const LiveGraphPacket& packet) -> Status {
            const auto* event = packet.getIf<StreamingS2TEvent>();
            if (stream != "events" || !event)
                return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Unexpected live STT output");
            observer(*event);
            return StatusCode::OK;
        },
        liveSession);
    if (status.ok())
        session.reset(new StreamingS2TSession(std::move(liveSession), mode));
    return status;
}

Status StreamingS2TSession::push(const std::string& utteranceId, AudioChunk audio) {
    return session_->send("audio", LiveGraphPacket::make(StreamingS2TInput{
        StreamingS2TInput::Type::Audio, utteranceId, std::make_shared<AudioChunk>(std::move(audio))}));
}

Status StreamingS2TSession::finish(const std::string& utteranceId) {
    return session_->send("audio", LiveGraphPacket::make(StreamingS2TInput{
        StreamingS2TInput::Type::Finish, utteranceId, nullptr}));
}

void StreamingS2TSession::cancel() {
    session_->cancel();
}

}  // namespace ovms

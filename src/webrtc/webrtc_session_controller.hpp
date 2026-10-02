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
//*****************************************************************************/
#pragma once

#include <cstddef>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "webrtc_peer_connection.hpp"
#include "audio_utterance_buffer.hpp"
#include "mock_echo_streaming_audio_model.hpp"
#include "omni_audio_adapter.hpp"
#include "opus_audio_codec.hpp"
#include "stt_audio_adapter.hpp"
#include "streaming_audio_processor.hpp"

namespace ovms {

class WebRtcSessionController {
public:
    struct Candidate {
        std::string candidate;
        std::string mid;
    };

    struct OfferResult {
        std::string sessionId;
        std::string type;
        std::string sdp;
        std::vector<Candidate> candidates;
    };

    explicit WebRtcSessionController(size_t maxSessions = 16, std::string omniModelPath = {},
        std::string sttModelPath = {}, std::string sttDevice = "CPU");

    bool createSession(const std::string& offerSdp, const std::string& offerType, OfferResult& result);
    bool addCandidate(const std::string& sessionId, const std::string& candidate, const std::string& mid);
    bool getCandidates(const std::string& sessionId, std::vector<Candidate>& candidates) const;
    bool removeSession(const std::string& sessionId);
    size_t sessionCount() const;

private:
    struct Session {
        explicit Session(rtc::Configuration configuration);
        void sendGeneratedAudio(const AudioChunk& audio, rtc::FrameInfo info, bool complete);
        void sendTranscript(const std::string& message);

        WebRtcPeerConnection peer;
        OpusAudioCodec codec;
        MockEchoStreamingAudioModel model;
        StreamingAudioProcessor processor;
        AudioUtteranceBuffer utteranceBuffer;
        std::deque<float> pendingOutputSamples;
        OmniAudioAdapter::ConversationPtr conversation;
        std::shared_ptr<rtc::DataChannel> transcriptChannel;
        std::mutex transcriptSendMutex;
        std::vector<Candidate> localCandidates;
        std::string localType;
        std::string localSdp;
        std::condition_variable descriptionReady;
        std::mutex mutex;
    };

    mutable std::mutex mutex_;
    size_t maxSessions_;
    std::string omniModelPath_;
    std::string sttModelPath_;
    std::string sttDevice_;
    std::shared_ptr<OmniAudioAdapter> omniAdapter_;
    std::shared_ptr<SttAudioAdapter> sttAdapter_;
    uint64_t nextSessionId_ = 1;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
};

}  // namespace ovms

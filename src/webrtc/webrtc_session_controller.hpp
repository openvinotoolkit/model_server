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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mock_echo_streaming_audio_model.hpp"
#include "opus_audio_codec.hpp"
#include "streaming_audio_processor.hpp"
#include "webrtc_peer_connection.hpp"

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

    explicit WebRtcSessionController(size_t maxSessions = 16);

    bool createSession(const std::string& offerSdp, const std::string& offerType, OfferResult& result);
    bool addCandidate(const std::string& sessionId, const std::string& candidate, const std::string& mid);
    bool removeSession(const std::string& sessionId);
    size_t sessionCount() const;

private:
    struct Session {
        explicit Session(rtc::Configuration configuration);

        WebRtcPeerConnection peer;
        OpusAudioCodec codec;
        MockEchoStreamingAudioModel model;
        StreamingAudioProcessor processor;
        std::vector<Candidate> localCandidates;
        std::string localType;
        std::string localSdp;
    };

    mutable std::mutex mutex_;
    size_t maxSessions_;
    uint64_t nextSessionId_ = 1;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
};

}  // namespace ovms

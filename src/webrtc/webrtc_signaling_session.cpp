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
#include "webrtc_signaling_session.hpp"

namespace ovms {

bool WebRtcSignalingSession::setRemoteOffer(std::string sdp) {
    if (sdp.empty() || state_ != State::New) {
        return false;
    }
    remoteOffer_ = std::move(sdp);
    state_ = State::HaveRemoteOffer;
    return true;
}

bool WebRtcSignalingSession::setLocalAnswer(std::string sdp) {
    if (sdp.empty() || state_ != State::HaveRemoteOffer) {
        return false;
    }
    localAnswer_ = std::move(sdp);
    state_ = State::Stable;
    return true;
}

bool WebRtcSignalingSession::addRemoteCandidate(std::string candidate, std::string mid) {
    if (candidate.empty() || mid.empty() || state_ == State::New) {
        return false;
    }
    remoteCandidates_.push_back({std::move(candidate), std::move(mid)});
    return true;
}

WebRtcSignalingSession::State WebRtcSignalingSession::state() const {
    return state_;
}

const std::string& WebRtcSignalingSession::remoteOffer() const {
    return remoteOffer_;
}

const std::string& WebRtcSignalingSession::localAnswer() const {
    return localAnswer_;
}

const std::vector<WebRtcSignalingSession::Candidate>& WebRtcSignalingSession::remoteCandidates() const {
    return remoteCandidates_;
}

}  // namespace ovms

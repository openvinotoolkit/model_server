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
#include "webrtc_peer_connection.hpp"

#include "src/logging.hpp"

namespace ovms {

WebRtcPeerConnection::WebRtcPeerConnection(rtc::Configuration configuration) :
    peerConnection_(std::make_shared<rtc::PeerConnection>(std::move(configuration))) {
    peerConnection_->onStateChange([](rtc::PeerConnection::State state) {
        SPDLOG_LOGGER_DEBUG(webrtc_logger, "PeerConnection state changed: {}", static_cast<int>(state));
    });
}

void WebRtcPeerConnection::onLocalDescription(DescriptionCallback callback) {
    peerConnection_->onLocalDescription([callback](rtc::Description description) {
        callback(std::string(description), description.typeString());
    });
}

void WebRtcPeerConnection::onLocalCandidate(CandidateCallback callback) {
    peerConnection_->onLocalCandidate([callback](rtc::Candidate candidate) {
        callback(candidate.candidate(), candidate.mid());
    });
}

void WebRtcPeerConnection::onStateChange(StateCallback callback) {
    // libdatachannel keeps a single onStateChange handler, so wrap the caller's
    // callback to preserve the constructor's debug logging.
    peerConnection_->onStateChange([callback](rtc::PeerConnection::State state) {
        SPDLOG_LOGGER_DEBUG(webrtc_logger, "PeerConnection state changed: {}", static_cast<int>(state));
        callback(state);
    });
}

void WebRtcPeerConnection::addAudioTrack() {
    rtc::Description::Audio audio("audio", rtc::Description::Direction::SendRecv);
    audio.addOpusCodec(111);
    audioTrack_ = peerConnection_->addTrack(audio);
}

void WebRtcPeerConnection::createOffer() {
    peerConnection_->setLocalDescription();
}

void WebRtcPeerConnection::setRemoteDescription(const std::string& sdp, const std::string& type) {
    peerConnection_->setRemoteDescription(rtc::Description(sdp, type));
}

void WebRtcPeerConnection::addRemoteCandidate(const std::string& candidate, const std::string& mid) {
    peerConnection_->addRemoteCandidate(rtc::Candidate(candidate, mid));
}

rtc::PeerConnection::State WebRtcPeerConnection::state() const {
    return peerConnection_->state();
}

}  // namespace ovms

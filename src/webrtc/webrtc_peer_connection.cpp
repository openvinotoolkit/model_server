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

#include <cstddef>

#include <rtc/rtpdepacketizer.hpp>
#include <rtc/rtppacketizer.hpp>

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

void WebRtcPeerConnection::onAudioFrame(AudioFrameCallback callback) {
    audioFrameCallback_ = std::move(callback);
    configureAudioTrackCallbacks();
}

void WebRtcPeerConnection::onProcessedAudioFrame(StreamingAudioProcessor& processor, ProcessedAudioFrameCallback callback) {
    audioProcessor_ = &processor;
    processedAudioFrameCallback_ = std::move(callback);
    configureAudioTrackCallbacks();
}

void WebRtcPeerConnection::onAudioTrackOpen(AudioTrackOpenCallback callback) {
    audioTrackOpenCallback_ = std::move(callback);
    configureAudioTrackCallbacks();
}

void WebRtcPeerConnection::onAudioTrack(AudioTrackCallback callback) {
    audioTrackCallback_ = std::move(callback);
    configureAudioTrackCallbacks();
}

void WebRtcPeerConnection::onLocalAudioTrackOpen(AudioTrackOpenCallback callback) {
    localAudioTrackOpenCallback_ = std::move(callback);
    if (audioTrack_) {
        if (audioTrack_->isOpen()) {
            localAudioTrackOpenCallback_();
        } else {
            audioTrack_->onOpen(localAudioTrackOpenCallback_);
        }
    }
}

bool WebRtcPeerConnection::isLocalAudioTrackOpen() const {
    return audioTrack_ && audioTrack_->isOpen();
}

void WebRtcPeerConnection::configureAudioTrackCallbacks() {
    if (audioTrackCallbacksConfigured_) {
        return;
    }
    audioTrackCallbacksConfigured_ = true;
    peerConnection_->onTrack([this](std::shared_ptr<rtc::Track> track) {
        if (track->description().type() != "audio") {
            return;
        }
        remoteAudioTracks_.push_back(track);
        if (audioTrackCallback_) {
            audioTrackCallback_();
        }
        if (audioFrameCallback_ || audioProcessor_) {
            track->setMediaHandler(std::make_shared<rtc::OpusRtpDepacketizer>());
            track->onFrame([this](rtc::binary data, rtc::FrameInfo info) {
                if (audioFrameCallback_) {
                    audioFrameCallback_(data, info);
                }
                if (audioProcessor_) {
                    const uint64_t timestampUs = info.timestampSeconds ?
                        static_cast<uint64_t>(info.timestampSeconds->count() * 1000000.0) :
                        static_cast<uint64_t>(info.timestamp);
                    std::vector<uint8_t> encoded(data.size());
                    for (size_t index = 0; index < data.size(); ++index) {
                        encoded[index] = std::to_integer<uint8_t>(data[index]);
                    }
                    const auto processed = audioProcessor_->process(encoded, timestampUs);
                    if (processedAudioFrameCallback_) {
                        rtc::binary output(processed.size());
                        for (size_t index = 0; index < processed.size(); ++index) {
                            output[index] = static_cast<std::byte>(processed[index]);
                        }
                        processedAudioFrameCallback_(std::move(output), info);
                    }
                }
            });
        }
        if (audioTrackOpenCallback_) {
            if (track->isOpen()) {
                audioTrackOpenCallback_();
            } else {
                track->onOpen(audioTrackOpenCallback_);
            }
        }
    });
}

void WebRtcPeerConnection::addAudioTrack(rtc::Description::Direction direction) {
    rtc::Description::Audio audio("audio", direction);
    audio.addOpusCodec(111);
    audio.addSSRC(1, "ovms-audio", "ovms-audio", "ovms-audio");
    audioTrack_ = peerConnection_->addTrack(audio);
    auto packetizationConfig = std::make_shared<rtc::RtpPacketizationConfig>(1, "ovms", 111, 48000);
    audioTrack_->setMediaHandler(std::make_shared<rtc::OpusRtpPacketizer>(std::move(packetizationConfig)));
    if (localAudioTrackOpenCallback_) {
        if (audioTrack_->isOpen()) {
            localAudioTrackOpenCallback_();
        } else {
            audioTrack_->onOpen(localAudioTrackOpenCallback_);
        }
    }
}

void WebRtcPeerConnection::createOffer() {
    peerConnection_->setLocalDescription();
}

void WebRtcPeerConnection::createAnswer() {
    peerConnection_->setLocalDescription();
}

bool WebRtcPeerConnection::sendAudioFrame(rtc::binary data, rtc::FrameInfo info) {
    if (!audioTrack_) {
        return false;
    }
    audioTrack_->sendFrame(std::move(data), info);
    return true;
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

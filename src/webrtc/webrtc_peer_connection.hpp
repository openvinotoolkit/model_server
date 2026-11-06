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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rtc/rtc.hpp>

namespace ovms {

// Terminates a single WebRTC peer connection (libdatachannel): signaling callbacks,
// a depacketized inbound audio track that can also send audio back, and data channels.
class WebRtcPeerConnection {
public:
    using DescriptionCallback = std::function<void(const std::string& sdp, const std::string& type)>;
    using CandidateCallback = std::function<void(const std::string& candidate, const std::string& mid)>;
    using StateCallback = std::function<void(rtc::PeerConnection::State state)>;
    using AudioFrameCallback = std::function<void(rtc::binary data, rtc::FrameInfo info)>;
    using AudioTrackCallback = std::function<void()>;
    using AudioTrackOpenCallback = std::function<void()>;
    using DataChannelCallback = std::function<void(std::shared_ptr<rtc::DataChannel> channel)>;
    using GatheringCompleteCallback = std::function<void()>;

    explicit WebRtcPeerConnection(rtc::Configuration configuration);

    void onLocalDescription(DescriptionCallback callback);
    void onLocalCandidate(CandidateCallback callback);
    void onStateChange(StateCallback callback);
    void onAudioFrame(AudioFrameCallback callback);
    void onAudioTrack(AudioTrackCallback callback);
    void onAudioTrackOpen(AudioTrackOpenCallback callback);
    void onLocalAudioTrackOpen(AudioTrackOpenCallback callback);
    void onDataChannel(DataChannelCallback callback);
    void onGatheringComplete(GatheringCompleteCallback callback);
    bool isLocalAudioTrackOpen() const;
    // Current local SDP including every ICE candidate gathered so far; empty before a local description exists.
    std::string localDescriptionSdp() const;
    // Closes transports (releasing the UDP port) and drops all registered callbacks.
    void close();

    // Adds a single Opus audio track (PT 111, the standard WebRTC dynamic payload type).
    void addAudioTrack(rtc::Description::Direction direction = rtc::Description::Direction::SendRecv);
    // Triggers local SDP offer generation; result is delivered via the onLocalDescription callback.
    void createOffer();
    // Triggers local SDP answer generation after a remote offer is set.
    void createAnswer();
    bool sendAudioFrame(rtc::binary data, rtc::FrameInfo info);

    void setRemoteDescription(const std::string& sdp, const std::string& type);
    void setRemoteDescription(const rtc::Description& description);
    void addRemoteCandidate(const std::string& candidate, const std::string& mid);

    rtc::PeerConnection::State state() const;
    bool getSelectedCandidatePair(rtc::Candidate& local, rtc::Candidate& remote);

private:
    void configureAudioTrackCallbacks();

    std::shared_ptr<rtc::PeerConnection> peerConnection_;
    std::shared_ptr<rtc::Track> audioTrack_;
    std::vector<std::shared_ptr<rtc::Track>> remoteAudioTracks_;
    bool audioTrackCallbacksConfigured_ = false;
    AudioFrameCallback audioFrameCallback_;
    AudioTrackCallback audioTrackCallback_;
    AudioTrackOpenCallback audioTrackOpenCallback_;
    AudioTrackOpenCallback localAudioTrackOpenCallback_;
    uint64_t outboundAudioPacketNumber_ = 0;
    std::shared_ptr<rtc::RtpPacketizationConfig> outboundPacketizationConfig_;
    std::string outboundCodecName_;
    uint8_t outboundPayloadType_ = 0;
    uint32_t outboundClockRate_ = 0;
    uint8_t outboundChannels_ = 0;
    bool outboundRtpTimestampInitialized_ = false;
    uint32_t outboundRtpTimestamp_ = 0;
};

}  // namespace ovms

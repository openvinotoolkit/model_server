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
#include <vector>

#include <rtc/rtc.hpp>

#include "streaming_audio_processor.hpp"

namespace ovms {

// POC scaffold: terminates a single WebRTC peer connection so the rest of the
// signaling/media pipeline (audio track, echo effect) can be built on top of
// a verified, working libdatachannel integration. No media handling yet.
class WebRtcPeerConnection {
public:
    using DescriptionCallback = std::function<void(const std::string& sdp, const std::string& type)>;
    using CandidateCallback = std::function<void(const std::string& candidate, const std::string& mid)>;
    using StateCallback = std::function<void(rtc::PeerConnection::State state)>;
    using AudioFrameCallback = std::function<void(rtc::binary data, rtc::FrameInfo info)>;
    using AudioTrackCallback = std::function<void()>;
    using AudioTrackOpenCallback = std::function<void()>;
    using ProcessedAudioFrameCallback = std::function<void(rtc::binary data, rtc::FrameInfo info)>;

    explicit WebRtcPeerConnection(rtc::Configuration configuration);

    void onLocalDescription(DescriptionCallback callback);
    void onLocalCandidate(CandidateCallback callback);
    void onStateChange(StateCallback callback);
    void onAudioFrame(AudioFrameCallback callback);
    void onProcessedAudioFrame(StreamingAudioProcessor& processor, ProcessedAudioFrameCallback callback);
    void onAudioTrack(AudioTrackCallback callback);
    void onAudioTrackOpen(AudioTrackOpenCallback callback);
    void onLocalAudioTrackOpen(AudioTrackOpenCallback callback);
    bool isLocalAudioTrackOpen() const;

    // Adds a single Opus audio track (PT 111, the standard WebRTC dynamic payload type).
    void addAudioTrack(rtc::Description::Direction direction = rtc::Description::Direction::SendRecv);
    // Triggers local SDP offer generation; result is delivered via the onLocalDescription callback.
    void createOffer();
    // Triggers local SDP answer generation after a remote offer is set.
    void createAnswer();
    bool sendAudioFrame(rtc::binary data, rtc::FrameInfo info);

    void setRemoteDescription(const std::string& sdp, const std::string& type);
    void addRemoteCandidate(const std::string& candidate, const std::string& mid);

    rtc::PeerConnection::State state() const;

private:
    void configureAudioTrackCallbacks();

    std::shared_ptr<rtc::PeerConnection> peerConnection_;
    std::shared_ptr<rtc::Track> audioTrack_;
    std::vector<std::shared_ptr<rtc::Track>> remoteAudioTracks_;
    bool audioTrackCallbacksConfigured_ = false;
    AudioFrameCallback audioFrameCallback_;
    StreamingAudioProcessor* audioProcessor_ = nullptr;
    ProcessedAudioFrameCallback processedAudioFrameCallback_;
    AudioTrackCallback audioTrackCallback_;
    AudioTrackOpenCallback audioTrackOpenCallback_;
    AudioTrackOpenCallback localAudioTrackOpenCallback_;
};

}  // namespace ovms

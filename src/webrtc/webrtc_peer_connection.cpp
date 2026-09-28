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
#include "rtp_packetization_config_utils.hpp"

#include <chrono>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <optional>

#include <rtc/rtpdepacketizer.hpp>
#include <rtc/rtppacketizer.hpp>

namespace ovms {

namespace {
constexpr uint32_t kOutboundAudioSsrc = 1;

struct NegotiatedAudioCodec {
    std::string name;
    uint8_t payloadType;
    uint32_t clockRate;
    uint8_t channels;
};

std::optional<NegotiatedAudioCodec> findNegotiatedAudioCodec(const rtc::Description::Media& description) {
    std::optional<NegotiatedAudioCodec> fallback;
    for (const int payloadType : description.payloadTypes()) {
        const auto* rtpMap = description.rtpMap(payloadType);
        if (rtpMap == nullptr) {
            continue;
        }
        std::string codecName = rtpMap->format;
        std::transform(codecName.begin(), codecName.end(), codecName.begin(), [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        if (codecName != "opus" && codecName != "pcmu" && codecName != "pcma" && codecName != "g722") {
            continue;
        }
        uint8_t channels = 1;
        if (codecName == "opus" && !rtpMap->encParams.empty()) {
            channels = static_cast<uint8_t>(std::stoi(rtpMap->encParams));
        }
        const NegotiatedAudioCodec codec{codecName, static_cast<uint8_t>(payloadType),
            static_cast<uint32_t>(rtpMap->clockRate), channels};
        if (codecName == "opus") {
            return codec;
        }
        fallback = codec;
    }
    return fallback;
}

std::string utcTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utcTime{};
    gmtime_r(&time, &utcTime);

    std::ostringstream timestamp;
    timestamp << std::put_time(&utcTime, "%Y-%m-%dT%H:%M:%S")
              << '.' << std::setfill('0') << std::setw(3) << milliseconds.count() << 'Z';
    return timestamp.str();
}

const char* toString(rtc::PeerConnection::State state) {
    switch (state) {
    case rtc::PeerConnection::State::New:
        return "New";
    case rtc::PeerConnection::State::Connecting:
        return "Connecting";
    case rtc::PeerConnection::State::Connected:
        return "Connected";
    case rtc::PeerConnection::State::Disconnected:
        return "Disconnected";
    case rtc::PeerConnection::State::Failed:
        return "Failed";
    case rtc::PeerConnection::State::Closed:
        return "Closed";
    default:
        return "Unknown";
    }
}
}  // namespace

WebRtcPeerConnection::WebRtcPeerConnection(rtc::Configuration configuration) :
    peerConnection_(std::make_shared<rtc::PeerConnection>(std::move(configuration))) {
    peerConnection_->onStateChange([](rtc::PeerConnection::State state) {
        SPDLOG_LOGGER_INFO(webrtc_logger, "PeerConnection state changed: {}", toString(state));
    });
}

void WebRtcPeerConnection::onLocalDescription(DescriptionCallback callback) {
    peerConnection_->onLocalDescription([callback](rtc::Description description) {
        SPDLOG_LOGGER_INFO(webrtc_logger, "Local description created, type: {}", description.typeString());
        callback(std::string(description), description.typeString());
    });
}

void WebRtcPeerConnection::onLocalCandidate(CandidateCallback callback) {
    peerConnection_->onLocalCandidate([callback](rtc::Candidate candidate) {
        SPDLOG_LOGGER_DEBUG(webrtc_logger, "Local ICE candidate gathered, mid: {}, candidate: {}", candidate.mid(), candidate.candidate());
        callback(candidate.candidate(), candidate.mid());
    });
}

void WebRtcPeerConnection::onStateChange(StateCallback callback) {
    // libdatachannel keeps a single onStateChange handler, so wrap the caller's
    // callback to preserve the constructor's debug logging.
    peerConnection_->onStateChange([callback](rtc::PeerConnection::State state) {
        SPDLOG_LOGGER_INFO(webrtc_logger, "PeerConnection state changed: {}", toString(state));
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
            SPDLOG_LOGGER_DEBUG(webrtc_logger, "Ignoring non-audio track, mid: {}", track->mid());
            return;
        }
        SPDLOG_LOGGER_INFO(webrtc_logger, "Received remote audio track, mid: {}", track->mid());
        auto trackDescription = track->description();
        SPDLOG_LOGGER_INFO(webrtc_logger, "Negotiated remote audio track: mid={}, type={}, description:\n{}",
            track->mid(), trackDescription.type(), std::string(trackDescription));
        const auto negotiatedCodec = findNegotiatedAudioCodec(trackDescription);
        const auto midExtensionId = findOutboundMidExtensionId(trackDescription);
        if (!negotiatedCodec) {
            SPDLOG_LOGGER_ERROR(webrtc_logger, "No supported negotiated audio codec found for mid={}", track->mid());
            return;
        }
        SPDLOG_LOGGER_INFO(webrtc_logger,
            "negotiated_audio_codec={} payload_type={} clock_rate={} channels={} mid_extension_id={}",
            negotiatedCodec->name, negotiatedCodec->payloadType, negotiatedCodec->clockRate, negotiatedCodec->channels,
            midExtensionId ? std::to_string(*midExtensionId) : "unavailable");
        remoteAudioTracks_.push_back(track);
        if (audioTrackCallback_) {
            audioTrackCallback_();
        }
        if (audioFrameCallback_ || audioProcessor_) {
            // Chain a packetizer after the depacketizer so this same (sendrecv)
            // remote track can also be used to send processed audio back,
            // without adding a second local m-line via addAudioTrack().
            std::shared_ptr<rtc::RtpDepacketizer> depacketizer;
            std::shared_ptr<rtc::RtpPacketizer> packetizer;
            const auto packetizationConfig = createAudioRtpPacketizationConfig(
                trackDescription, track->mid(), kOutboundAudioSsrc,
                negotiatedCodec->payloadType, negotiatedCodec->clockRate);
            if (negotiatedCodec->name == "opus") {
                depacketizer = std::make_shared<rtc::OpusRtpDepacketizer>(negotiatedCodec->clockRate);
                packetizer = std::make_shared<rtc::OpusRtpPacketizer>(packetizationConfig);
            } else if (negotiatedCodec->name == "pcmu") {
                depacketizer = std::make_shared<rtc::PCMURtpDepacketizer>(negotiatedCodec->clockRate);
                packetizer = std::make_shared<rtc::PCMURtpPacketizer>(packetizationConfig);
            } else if (negotiatedCodec->name == "pcma") {
                depacketizer = std::make_shared<rtc::PCMARtpDepacketizer>(negotiatedCodec->clockRate);
                packetizer = std::make_shared<rtc::PCMARtpPacketizer>(packetizationConfig);
            } else {
                depacketizer = std::make_shared<rtc::G722RtpDepacketizer>(negotiatedCodec->clockRate);
                packetizer = std::make_shared<rtc::G722RtpPacketizer>(packetizationConfig);
            }
            outboundPacketizationConfig_ = packetizationConfig;
            outboundCodecName_ = negotiatedCodec->name;
            outboundPayloadType_ = negotiatedCodec->payloadType;
            outboundClockRate_ = negotiatedCodec->clockRate;
            outboundChannels_ = negotiatedCodec->channels;
            outboundRtpTimestampInitialized_ = false;
            depacketizer->addToChain(std::move(packetizer));
            track->setMediaHandler(depacketizer);
            track->onFrame([this](rtc::binary data, rtc::FrameInfo info) {
                SPDLOG_LOGGER_TRACE(webrtc_logger,
                    "Inbound audio packet: codec={}, payload_type={}, clock_rate={}, channels={}, ssrc={}, bytes={}, "
                    "rtp_timestamp={}, timestamp_seconds={}",
                    outboundCodecName_, outboundPayloadType_, outboundClockRate_, outboundChannels_, kOutboundAudioSsrc,
                    data.size(), info.timestamp,
                    info.timestampSeconds ? info.timestampSeconds->count() : -1);
                if (audioFrameCallback_) {
                    audioFrameCallback_(data, info);
                }
                if (audioProcessor_) {
                    const uint64_t timestampUs = info.timestampSeconds ?
                        static_cast<uint64_t>(info.timestampSeconds->count() * 1000000.0) :
                        static_cast<uint64_t>((static_cast<uint64_t>(info.timestamp) * 1000000) /
                            std::max<uint32_t>(outboundClockRate_, 1));
                    std::vector<uint8_t> encoded(data.size());
                    for (size_t index = 0; index < data.size(); ++index) {
                        encoded[index] = std::to_integer<uint8_t>(data[index]);
                    }
                    try {
                        const auto processed = audioProcessor_->process(encoded, timestampUs);
                        SPDLOG_LOGGER_TRACE(webrtc_logger,
                            "Processed audio packet: encoded_bytes_in={}, encoded_bytes_out={}, payload_type={}, "
                            "rtp_timestamp={}, timestamp_us={}",
                            encoded.size(), processed.size(), info.payloadType, info.timestamp, timestampUs);
                        if (processedAudioFrameCallback_) {
                            rtc::binary output(processed.size());
                            for (size_t index = 0; index < processed.size(); ++index) {
                                output[index] = static_cast<std::byte>(processed[index]);
                            }
                            processedAudioFrameCallback_(std::move(output), info);
                        }
                    } catch (const std::exception& e) {
                        SPDLOG_LOGGER_ERROR(webrtc_logger,
                            "Failed to decode/process incoming audio packet: bytes={}, payload_type={}, "
                            "rtp_timestamp={}, error={}",
                            data.size(), info.payloadType, info.timestamp, e.what());
                    }
                }
            });
        }
        if (audioTrackOpenCallback_) {
            if (track->isOpen()) {
                SPDLOG_LOGGER_INFO(webrtc_logger, "Remote audio track already open, mid: {}", track->mid());
                audioTrackOpenCallback_();
            } else {
                track->onOpen([this, track, callback = audioTrackOpenCallback_] {
                    SPDLOG_LOGGER_INFO(webrtc_logger, "Remote audio track opened, mid: {}", track->mid());
                    callback();
                });
            }
        }
    });
}

void WebRtcPeerConnection::addAudioTrack(rtc::Description::Direction direction) {
    SPDLOG_LOGGER_INFO(webrtc_logger, "Adding local audio track, direction: {}", static_cast<int>(direction));
    rtc::Description::Audio audio("audio", direction);
    audio.addOpusCodec(111);
    audio.addSSRC(kOutboundAudioSsrc, "ovms-audio", "ovms-audio", "ovms-audio");
    audioTrack_ = peerConnection_->addTrack(audio);
    auto packetizationConfig = std::make_shared<rtc::RtpPacketizationConfig>(kOutboundAudioSsrc, "ovms", 111, 48000);
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
    SPDLOG_LOGGER_INFO(webrtc_logger, "Creating local SDP offer");
    peerConnection_->setLocalDescription();
}

void WebRtcPeerConnection::createAnswer() {
    SPDLOG_LOGGER_INFO(webrtc_logger, "Creating local SDP answer");
    peerConnection_->setLocalDescription();
}

bool WebRtcPeerConnection::sendAudioFrame(rtc::binary data, rtc::FrameInfo info) {
    // Answering a browser's single sendrecv m-line reuses that remote track for
    // sending back; a distinct local track is only added when we are the offerer.
    const auto& track = audioTrack_ ? audioTrack_ : (remoteAudioTracks_.empty() ? nullptr : remoteAudioTracks_.back());
    if (!track) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Cannot send audio frame, no local or remote audio track available");
        return false;
    }
    const uint64_t packetNumber = ++outboundAudioPacketNumber_;
    if (outboundCodecName_ == "opus" && outboundClockRate_ == OpusAudioCodec::SampleRate) {
        if (!outboundRtpTimestampInitialized_) {
            outboundRtpTimestamp_ = info.timestamp;
            outboundRtpTimestampInitialized_ = true;
        }
        info.payloadType = outboundPayloadType_;
        info.timestamp = outboundRtpTimestamp_;
        info.timestampSeconds.reset();
        if (outboundPacketizationConfig_) {
            outboundPacketizationConfig_->timestamp = outboundRtpTimestamp_;
        }
    }
    const size_t payloadBytes = data.size();
    if (webrtc_logger->should_log(spdlog::level::trace)) {
        std::ostringstream hexPayload;
        hexPayload << std::hex << std::setfill('0');
        for (const auto byte : data) {
            hexPayload << std::setw(2) << static_cast<unsigned int>(std::to_integer<uint8_t>(byte));
        }
        SPDLOG_LOGGER_TRACE(webrtc_logger,
            "Outbound audio packet: utc_timestamp={}, send_packet_number={}, codec={}, payload_type={}, clock_rate={}, channels={}, "
            "ssrc={}, bytes={}, hex={}, rtp_timestamp={}, timestamp_seconds={}",
            utcTimestamp(), packetNumber, outboundCodecName_, info.payloadType, outboundClockRate_, outboundChannels_,
            kOutboundAudioSsrc, data.size(), hexPayload.str(), info.timestamp,
            info.timestampSeconds ? info.timestampSeconds->count() : -1);
    }
    if (outboundPacketizationConfig_ && packetNumber % 50 == 0) {
        SPDLOG_LOGGER_INFO(webrtc_logger,
            "outbound_rtp_pre_srtp: codec={} payload_type={} clock_rate={} channels={} ssrc={} sequence={} timestamp={} payload_bytes={}",
            outboundCodecName_, outboundPayloadType_, outboundClockRate_, outboundChannels_,
            outboundPacketizationConfig_->ssrc, outboundPacketizationConfig_->sequenceNumber,
            outboundPacketizationConfig_->timestamp, payloadBytes);
    }
    track->sendFrame(std::move(data), info);
    if (outboundCodecName_ == "opus" && outboundClockRate_ == OpusAudioCodec::SampleRate) {
        outboundRtpTimestamp_ += static_cast<uint32_t>(OpusAudioCodec::FrameSamples);
    }
    return true;
}

void WebRtcPeerConnection::setRemoteDescription(const std::string& sdp, const std::string& type) {
    SPDLOG_LOGGER_INFO(webrtc_logger, "Setting remote description, type: {}", type);
    peerConnection_->setRemoteDescription(rtc::Description(sdp, type));
}

void WebRtcPeerConnection::addRemoteCandidate(const std::string& candidate, const std::string& mid) {
    SPDLOG_LOGGER_DEBUG(webrtc_logger, "Adding remote ICE candidate, mid: {}, candidate: {}", mid, candidate);
    peerConnection_->addRemoteCandidate(rtc::Candidate(candidate, mid));
}

rtc::PeerConnection::State WebRtcPeerConnection::state() const {
    return peerConnection_->state();
}

bool WebRtcPeerConnection::getSelectedCandidatePair(rtc::Candidate& local, rtc::Candidate& remote) {
    return peerConnection_->getSelectedCandidatePair(&local, &remote);
}

}  // namespace ovms

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
#include "webrtc_session_controller.hpp"

#include <chrono>
#include <exception>
#include <sstream>
#include <utility>
#include <vector>

#include "src/logging.hpp"

namespace ovms {

namespace {

std::string preferOpusAudioCodec(const std::string& offerSdp) {
    std::istringstream input(offerSdp);
    std::vector<std::string> lines;
    std::string line;
    std::string opusPayloadType;
    size_t audioSectionStart = 0;
    bool inAudioSection = false;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.rfind("m=audio ", 0) == 0) {
            audioSectionStart = lines.size();
            inAudioSection = true;
        } else if (line.rfind("m=", 0) == 0) {
            inAudioSection = false;
        } else if (inAudioSection && line.rfind("a=rtpmap:", 0) == 0 && line.find(" opus/") != std::string::npos) {
            const size_t payloadStart = std::string("a=rtpmap:").size();
            const size_t payloadEnd = line.find(' ', payloadStart);
            if (payloadEnd != std::string::npos) {
                opusPayloadType = line.substr(payloadStart, payloadEnd - payloadStart);
            }
        }
        lines.push_back(line);
    }

    if (opusPayloadType.empty() || audioSectionStart >= lines.size()) {
        return offerSdp;
    }
    std::istringstream audioLine(lines[audioSectionStart]);
    std::vector<std::string> fields;
    std::string field;
    while (audioLine >> field) {
        fields.push_back(field);
    }
    if (fields.size() < 4) {
        return offerSdp;
    }
    std::ostringstream rewrittenAudioLine;
    rewrittenAudioLine << fields[0] << ' ' << fields[1] << ' ' << fields[2];
    rewrittenAudioLine << ' ' << opusPayloadType;
    lines[audioSectionStart] = rewrittenAudioLine.str();

    std::ostringstream output;
    bool outputInAudioSection = false;
    for (const auto& rewrittenLine : lines) {
        if (rewrittenLine.rfind("m=audio ", 0) == 0) {
            outputInAudioSection = true;
        } else if (rewrittenLine.rfind("m=", 0) == 0) {
            outputInAudioSection = false;
        }
        if (outputInAudioSection &&
            (rewrittenLine.rfind("a=rtpmap:", 0) == 0 ||
                rewrittenLine.rfind("a=fmtp:", 0) == 0 ||
                rewrittenLine.rfind("a=rtcp-fb:", 0) == 0)) {
            const size_t payloadStart = rewrittenLine.find(':') + 1;
            const size_t payloadEnd = rewrittenLine.find(' ', payloadStart);
            if (payloadEnd != std::string::npos && rewrittenLine.substr(payloadStart, payloadEnd - payloadStart) != opusPayloadType) {
                continue;
            }
        }
        output << rewrittenLine << "\r\n";
    }
    return output.str();
}

}  // namespace

WebRtcSessionController::Session::Session(rtc::Configuration configuration) :
    peer(std::move(configuration)),
    codec(OpusAudioCodec::SampleRate, 2),
    model(OpusAudioCodec::SampleRate),
    processor(codec, model) {
}

WebRtcSessionController::WebRtcSessionController(size_t maxSessions) :
    maxSessions_(maxSessions) {
}

bool WebRtcSessionController::createSession(const std::string& offerSdp, const std::string& offerType, OfferResult& result) {
    if (offerSdp.empty() || offerType != "offer") {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, invalid offer, type: {}", offerType);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sessions_.size() >= maxSessions_) {
            SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, session limit reached: {}", maxSessions_);
            return false;
        }
    }

    SPDLOG_LOGGER_INFO(webrtc_logger, "Creating WebRTC session from browser offer");
    rtc::Configuration configuration;
    // One fixed POC port lets a UDP relay forward browser media to the container.
    configuration.portRangeBegin = 52000;
    configuration.portRangeEnd = 52000;
    auto session = std::make_shared<Session>(std::move(configuration));
    session->peer.onStateChange([session](rtc::PeerConnection::State state) {
        if (state != rtc::PeerConnection::State::Connected) {
            return;
        }
        rtc::Candidate localCandidate;
        rtc::Candidate remoteCandidate;
        if (session->peer.getSelectedCandidatePair(localCandidate, remoteCandidate)) {
            const std::string remoteAddress = remoteCandidate.address().value_or("unknown");
            const uint16_t remotePort = remoteCandidate.port().value_or(0);
            SPDLOG_LOGGER_INFO(webrtc_logger,
                "Selected ICE pair for WebRTC session: local_candidate={}, remote_candidate={}, outbound_destination={}:{}",
                localCandidate.candidate(), remoteCandidate.candidate(), remoteAddress, remotePort);
        } else {
            SPDLOG_LOGGER_WARN(webrtc_logger,
                "WebRTC session connected but selected ICE candidate pair is unavailable");
        }
    });
    session->peer.onLocalDescription([session](const std::string& sdp, const std::string& type) {
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            session->localSdp = sdp;
            session->localType = type;
        }
        session->descriptionReady.notify_all();
    });
    session->peer.onLocalCandidate([session](const std::string& candidate, const std::string& mid) {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->localCandidates.push_back({candidate, mid});
    });
    session->peer.onProcessedAudioFrame(session->processor, [session](rtc::binary data, rtc::FrameInfo info) {
        session->peer.sendAudioFrame(std::move(data), info);
    });
    try {
        // Reuse the browser's single sendrecv m-line (via onTrack) instead of
        // calling addAudioTrack(), which would add a second, unmatched m-line
        // and make libdatachannel emit a renegotiation offer instead of an answer.
        const auto opusPreferredOffer = preferOpusAudioCodec(offerSdp);
        session->peer.setRemoteDescription(opusPreferredOffer, offerType);
        session->peer.createAnswer();
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Failed to negotiate WebRTC session: {}", e.what());
        return false;
    }

    std::unique_lock<std::mutex> lock(session->mutex);
    if (!session->descriptionReady.wait_for(lock, std::chrono::seconds(5), [&session] {
            return !session->localSdp.empty();
        })) {
        SPDLOG_LOGGER_ERROR(webrtc_logger, "Timed out waiting for local SDP answer");
        return false;
    }

    result.type = session->localType;
    result.sdp = session->localSdp;
    result.candidates = session->localCandidates;
    lock.unlock();

    std::lock_guard<std::mutex> sessionsLock(mutex_);
    if (sessions_.size() >= maxSessions_) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected session creation, session limit reached after negotiation: {}", maxSessions_);
        return false;
    }
    result.sessionId = std::to_string(nextSessionId_++);
    SPDLOG_LOGGER_INFO(webrtc_logger, "WebRTC session {} created, answer type: {}, local candidates: {}", result.sessionId, result.type, result.candidates.size());
    sessions_.emplace(result.sessionId, std::move(session));
    return true;
}

bool WebRtcSessionController::addCandidate(const std::string& sessionId, const std::string& candidate, const std::string& mid) {
    std::shared_ptr<Session> session;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(sessionId);
        if (it == sessions_.end()) {
            SPDLOG_LOGGER_WARN(webrtc_logger, "Cannot add ICE candidate, unknown session: {}", sessionId);
            return false;
        }
        session = it->second;
    }
    if (candidate.empty() || mid.empty()) {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Rejected empty ICE candidate for session: {}", sessionId);
        return false;
    }
    SPDLOG_LOGGER_DEBUG(webrtc_logger, "Adding remote ICE candidate for session: {}", sessionId);
    session->peer.addRemoteCandidate(candidate, mid);
    SPDLOG_LOGGER_DEBUG(webrtc_logger,
        "Accepted remote ICE candidate for session {}: mid={}, candidate={}", sessionId, mid, candidate);
    return true;
}

bool WebRtcSessionController::getCandidates(const std::string& sessionId, std::vector<Candidate>& candidates) const {
    std::shared_ptr<Session> session;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(sessionId);
        if (it == sessions_.end()) {
            return false;
        }
        session = it->second;
    }
    std::lock_guard<std::mutex> lock(session->mutex);
    candidates = session->localCandidates;
    return true;
}

bool WebRtcSessionController::removeSession(const std::string& sessionId) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool removed = sessions_.erase(sessionId) != 0;
    if (removed) {
        SPDLOG_LOGGER_INFO(webrtc_logger, "WebRTC session {} closed", sessionId);
    } else {
        SPDLOG_LOGGER_WARN(webrtc_logger, "Cannot close unknown session: {}", sessionId);
    }
    return removed;
}

size_t WebRtcSessionController::sessionCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.size();
}

}  // namespace ovms